"""
A module for executing tasks in a Directed Acyclic Graph (DAG) using NetworkX.

This module provides functionality to create and execute a DAG where each node
represents a task with dependencies. Tasks are executed in topological order,
ensuring that all dependencies of a task are completed before the task itself runs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx
from loguru import logger
from networkx import DiGraph
from networkx.readwrite import json_graph
from pydantic import BaseModel, model_validator
from toolz import curry
from typing_extensions import Protocol

T = TypeVar("T")


class TaskSkipped(Exception):
    pass


class TaskStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submited"
    RUNNING = "running"
    SKIPPED = "skipped"
    DONE = "done"


@runtime_checkable
class Task(Protocol[T]):
    """Protocol defining the interface for a task in the DAG."""

    task_id: str
    tags: list[str]
    status: TaskStatus | str

    def execute(self, exec_context: dict[str, Any]) -> T:
        """Execute the task and return its result."""
        ...


class Scheduling(BaseModel):
    day: int | None = None
    active_days: str = "mon-sun"

    @property
    def periodicity(self) -> str:
        if self.day is not None:
            return "monthly"
        elif self.active_days != "mon-sun":
            return "weekly"
        else:
            return "daily"

    def is_active_day(self, date: datetime) -> bool:
        if self.day is not None:
            return date.day == self.day
        elif self.active_days != "mon-sun":
            return date.weekday() in self.get_numeric_week_days()
        else:
            return True

    def get_numeric_week_days(self) -> list[int]:
        weekday_map = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        tokens = self.active_days.split("-")
        if len(tokens) == 2:
            num_days = range(weekday_map[tokens[0]], weekday_map[tokens[1]] + 1)
        else:
            num_days = [weekday_map[tokens[0]]]

        return num_days


class BaseTaskModel(BaseModel, ABC):
    """Base model for tasks."""

    task_id: str
    tags: list[str] = []
    status: TaskStatus = TaskStatus.PENDING
    scheduling: Scheduling | None = Scheduling()

    @abstractmethod
    def execute(self, exec_context: dict[str, Any]) -> T:
        """Execute the task and return its result."""
        ...


class FunctionTask(BaseTaskModel, Generic[T]):
    """A task that wraps a callable function."""

    func_name: str | None = None
    func: Callable[..., T] | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_func(self):
        """Sets func field as callable"""
        try:
            if self.func_name:
                self.func = globals().get(self.func_name)
            elif self.func:
                self.func_name = self.func.__name__
            else:
                raise ValueError("Function name or function not provided")
        except AttributeError:
            raise ValueError(f"Function '{self.func_name}' not found")
        return self

    def execute(self, exec_context: dict[str, Any]) -> T:
        """Execute the wrapped function with stored arguments."""
        kwargs = self.kwargs or {}
        self.status = TaskStatus.RUNNING
        if not self.scheduling.is_active_day(
            datetime.fromisoformat(exec_context["date"])
        ):
            logger.info(f"Task '{self.task_id}' is not scheduled for today")
            raise TaskSkipped(self.task_id)

        logger.debug(
            f"Executing task '{self.task_id}': "
            f"'{self.func_name}' with args {self.args} and kwargs {self.kwargs}"
            # f"\nExecution context: {exec_context}"
        )
        return self.func(*self.args, **kwargs)


class DAGStorage:
    """Class for saving and loading DAGs to and from a file."""

    def __init__(self, file_path: str):
        self._file_path = file_path

    def save_dag(self, runner: DAGRunner) -> None:
        # Create a copy of the graph to avoid modifying the original
        graph = runner.graph.copy()

        # Add task data as node attributes
        for node_id, data in graph.nodes(data=True):
            # Store task information as node attributes
            task = data.get("task")
            graph.nodes[node_id].update(
                {
                    "task_type": task.__class__.__name__,
                    "task_module": task.__class__.__module__,
                }
            )

            # Add function-specific attributes for FunctionTask
            if isinstance(task, FunctionTask):
                graph.nodes[node_id]["task"] = task.model_dump(exclude={"func"})
            else:
                raise ValueError(f"Task type {task.__class__.__name__} not supported")

        # Convert to node-link format and save as JSON
        # noinspection PyArgumentList
        data = json_graph.node_link_data(graph, edges="edges")
        with open(self._file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_dag(self, max_workers: int = 3) -> DAGRunner:
        with open(self._file_path, "r") as f:
            data = json.load(f)

        # Reconstruct the graph
        graph = json_graph.node_link_graph(data, edges="edges")

        # Create a new DAGRunner
        runner = DAGRunner(max_workers=max_workers)

        # Recreate tasks (simplified - you'll need to handle task recreation properly)
        for node_id, node_data in graph.nodes(data=True):
            # WARNING: This is a simplified example
            # In a real implementation, you'd need proper task reconstruction logic
            cls = globals().get(node_data.get("task_type"))
            task_d = node_data.get("task")
            task = cls(**task_d)
            runner.add_task(node_id, task)

        # Recreate dependencies
        for source, target in graph.edges():
            runner.add_dependency(str(target), [str(source)])

        return runner


class DAGRunner:
    """
    A class for building and executing a Directed Acyclic Graph (DAG) of tasks.

    Tasks are executed in topological order, ensuring that all dependencies
    of a task are completed before the task itself is executed.
    """

    def __init__(self, max_workers: int) -> None:
        """Initialize a new DAGRunner with an empty graph."""
        self.graph = nx.DiGraph()
        self.results: Dict[str, Any] = {}
        self.max_workers = max_workers
        self._pool_workers_semaphore = threading.Semaphore(max_workers)
        self._result_event = threading.Event()
        self._lock = threading.Lock()

    def wait_for_tasks_results(
        self, node_id: str, graph: DiGraph, timeout: float = None
    ) -> Any:
        """Wait for a task to complete and return its result.

        :param node_id: The ID of the node that is waiting for the tasks
        :param graph: The graph to check.
        :param timeout: Maximum time to wait in seconds (None for no timeout)
        :return: The task's result
        :raises TimeoutError: If the timeout is reached
        :raises KeyError: If task_id doesn't exist
        """

        # Wait for any of the events to be set and then check for task results
        while True:
            task_id_lst = nx.ancestors(graph, node_id)
            logger.debug(f"{node_id}: Waiting for task results: {task_id_lst}")
            # * wait for any result event
            if not self._result_event.wait(timeout=timeout):
                raise TimeoutError("Timeout reached")

            if self.are_ancestors_done(node_id=node_id, graph=graph):
                logger.debug(f"{node_id}: Wait end: All tasks done: {task_id_lst}")
                return
            # check if all tasks are done
            logger.debug(f"{node_id}: Tasks pending: {self.pending_ancestors(node_id, graph)}")

    def reset_status(self, tags: list[str] = None):
        for node_id, node in self.graph.nodes(data=True):
            task = self.graph.nodes[node_id].get("task")
            if tags and self.any_tag_in_tags(*task.tags, tags=tags):
                task.status = TaskStatus.PENDING
                self.results.pop(node_id, None)

    @staticmethod
    def are_ancestors_done(node_id: str, graph: DiGraph):
        """Check if all ancestors of a node are done.

        :param node_id: The ID of the node to check.
        :param graph: The graph to check.
        :return: True if all ancestors of the node are done, False otherwise.
        """
        for asc_id in nx.ancestors(graph, node_id):
            task = graph.nodes[asc_id].get("task")
            if task.status != TaskStatus.DONE:
                return False
        return True

    @staticmethod
    def pending_ancestors(node_id: str, graph: DiGraph):
        return [
            asc_id
            for asc_id in nx.ancestors(graph, node_id)
            if graph.nodes[asc_id].get("task").status != TaskStatus.DONE
        ]

    @staticmethod
    @curry
    def sort_generation(
        graph: DiGraph,
        generations: list[list[str]],
        generation: tuple[int, list[str]],
    ) -> tuple[int, list[str]]:
        g, generation = generation
        if g == 0:
            return g, generation
        ancestors_index_d = {}
        node_ancestors_d = {}
        for node in generation:
            ancestors = nx.ancestors(graph, node)
            node_ancestors_d[node] = ancestors
            ancestors_index_d[node] = max(
                [
                    generations[g - 1].index(asc_id)
                    for asc_id in ancestors
                    if asc_id in generations[g - 1]
                ]
            )
            pass

        s_gen = sorted(generation, key=lambda node_id: ancestors_index_d[node_id])
        return g, s_gen

    # noinspection D

    def get_secure_node_generator(
        self, graph: DiGraph, max_node_retries: int = 0
    ) -> Generator[str]:
        """Get a generator that yields nodes in topological order, ensuring that all
        task ancestors of a node are done before the node is yielded.

        :param graph: The graph to yield nodes from.
        :param max_node_retries: The maximum number of retries for a node to be yielded. If 0,
        there is no limit.
        :return: A generator that yields nodes in topological order.
        """
        gen_dict = {
            g: gen for g, gen in enumerate(list(nx.topological_generations(graph)))
        }
        # * No need for generations sort ... topological order is already preserved
        # generations = list(gen_dict.values())
        # gen_dict = toolz.itemmap(
        #     DAGRunner.sort_generation(graph, generations), gen_dict
        # )

        moved_set = set()
        retries = 0
        for g, gen in gen_dict.items():
            logger.debug(f"Generation {g}: {gen}")
            if g == 0:
                logger.debug(f"Yielding generation {g} directly (no dependencies)")
                yield from gen
            else:
                i = 0
                while i < len(gen):
                    node = gen[i]
                    if DAGRunner.are_ancestors_done(node, graph):
                        logger.debug(
                            f"Yielding node {node} (gen {g}, position {i + 1}/{len(gen)})"
                        )
                        yield node
                        retries = 0
                        i += 1
                    elif node not in moved_set and node != gen[-1]:
                        logger.debug(f"Moving node {node} to end of generation {g}")
                        gen.remove(node)
                        gen.append(node)
                        moved_set.add(node)
                        # * process same i index again
                    elif retries < max_node_retries or max_node_retries == 0:
                        retries += 1
                        logger.debug(
                            f"Node {node} already moved or in last position, "
                            f"waiting for ancestors to be done, "
                            f"retries: {retries}"
                        )
                        self.wait_for_tasks_results(node_id=node, graph=graph)
                    else:
                        logger.error(
                            f"Node {node} already moved, and retried {max_node_retries} times ..."
                            f"giving up"
                        )
                        yield node
                        retries = 0
                        i += 1
                        # * process same i index again

    @staticmethod
    def any_tag_in_tags(*args: str, tags: list[str]) -> bool:
        return any([t in tags for t in args])

    def add_task(
        self,
        task_id: str,
        task: Union[Task[T], Callable[..., T]],
        tags: list[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # If task is a callable but not a Task, wrap it in a FunctionTask
        if callable(task) and not isinstance(task, Task):
            task = FunctionTask(
                task_id=task_id, func=task, args=args, kwargs=kwargs, tags=tags or []
            )

        self.graph.add_node(task_id, task=task)

    def add_dependency(self, task_id: str, depends_on: List[str]) -> None:
        if task_id not in self.graph.nodes():
            raise ValueError(f"Task with ID '{task_id}' not found")

        for dep in depends_on:
            if dep not in self.graph.nodes():
                raise ValueError(f"Dependency '{dep}' not found in tasks")
            self.graph.add_edge(dep, task_id)

    def validate(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    @curry
    def done_callback(
        self, future: Future, task_id: str, exec_context: dict[str, Any]
    ) -> None:
        task = self.graph.nodes[task_id].get("task")
        try:
            self.results[task_id] = future.result()
            logger.info(f"{task_id}: done! with exec_context: {exec_context}")
            task.status = TaskStatus.DONE
        except TaskSkipped:
            logger.info(f"{task_id}: not scheduled for today, skipping")
            task.status = TaskStatus.SKIPPED

        except Exception as e:
            self.results[task_id] = e
            logger.error(f"{task_id}: failed with exec_context: {exec_context}")
            task = self.graph.nodes[task_id].get("task")
            task.status = TaskStatus.DONE
        finally:
            with self._lock:
                self._result_event.set()
                self._result_event.clear()
                # release semaphore to allow pool to submit more tasks
                self._pool_workers_semaphore.release()

    @staticmethod
    def warn_for_active_descendant(
        date: datetime, descent: Iterable, node_id: str, working_graph: DiGraph
    ):
        for desc_id in descent:
            task = working_graph.nodes[desc_id].get("task")
            if task.scheduling.is_active_day(date):
                logger.warning(
                    f"{node_id}: removing active descendant {desc_id} from "
                    f"inactive node {node_id}"
                )

    @staticmethod
    def warn_for_with_tag_descendant(
        tags: list[str], descent: Iterable, node_id: str, working_graph: DiGraph
    ):
        for desc_id in descent:
            task = working_graph.nodes[desc_id].get("task")
            if DAGRunner.any_tag_in_tags(*task.tags, tags=tags):
                logger.warning(
                    f"{node_id}: removing descendant {desc_id} with tag {tags} where "
                    f"ascendant {node_id} has no tag {tags}"
                )

    @staticmethod
    def filter_nodes_by_active_day(date: datetime, working_graph: DiGraph) -> DiGraph:
        # Find all inactive nodes and their
        res_graph = working_graph.copy()
        nodes_to_remove = set()
        for node_id, node_data in working_graph.nodes(data=True):
            if not node_data.get("task").scheduling.is_active_day(date):
                # Add the node and all its descendants to the removal set
                descent = nx.descendants(working_graph, node_id)
                DAGRunner.warn_for_active_descendant(
                    date, descent, node_id, working_graph
                )

                nodes_to_remove.update(descent)
                nodes_to_remove.add(node_id)

        # Remove inactive tasks and their descendants
        res_graph.remove_nodes_from(nodes_to_remove)
        return res_graph

    @staticmethod
    def filter_nodes_by_tags(tags: list[str] | None, working_graph: DiGraph) -> DiGraph:
        # filter tasks by tags
        if not tags:
            return working_graph

        res_graph = working_graph.copy()
        logger.info(f"Executing tasks with tags: {tags}")
        non_tag_nodes = [
            node_id
            for node_id, node in working_graph.nodes(data=True)
            if not DAGRunner.any_tag_in_tags(*node.get("task").tags, tags=tags)
        ]
        nodes_to_remove = []
        for node_id in non_tag_nodes:
            nodes_to_remove.append(node_id)
            descent = nx.descendants(working_graph, node_id)
            DAGRunner.warn_for_with_tag_descendant(
                tags, descent, node_id, working_graph
            )
            nodes_to_remove.extend(descent)

        res_graph.remove_nodes_from(nodes_to_remove)
        return res_graph

    def execute(
        self, exec_context: dict[str, Any], tags: list[str] = None
    ) -> Dict[str, Any]:
        if not self.validate():
            raise ValueError("The graph contains cycles and is not a valid DAG")

        # Create a working copy of the graph
        working_graph = self.filter_nodes_by_tags(tags, self.graph)
        working_graph = self.filter_nodes_by_active_day(
            datetime.fromisoformat(exec_context.get("date")), working_graph
        )

        logger.info(f"Final working graph node size: {working_graph.number_of_nodes()}")

        # Execute tasks in topological order using a thread pool to parallelize excution
        # in multiple threads and using graph generations to execute tasks in parallel.

        node_gen = self.get_secure_node_generator(working_graph)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for node_id in node_gen:
                task = working_graph.nodes[node_id].get("task")
                if task.status == "done":
                    logger.info(f"{task.task_id}: already done, skipping")
                    continue
                task.status = "submited"
                self._pool_workers_semaphore.acquire()
                executor.submit(
                    task.execute, exec_context=exec_context
                ).add_done_callback(
                    self.done_callback(task_id=task.task_id, exec_context=exec_context)
                )
                logger.debug(
                    f"Active workers: {self.max_workers - self._pool_workers_semaphore._value}"
                )

        return self.results


def func_task(sleep: int = 0) -> float:
    """A sample function that returns 1."""
    if sleep:
        sleep_t = random.random() * sleep
        time.sleep(sleep_t)
        return sleep_t
    return 0


def example_usage2() -> None:
    """Example usage of the DAGRunner class with separate node and edge creation."""
    # Create a new DAG runner
    storage = DAGStorage("dag.json")

    if Path("dag.json").exists():
        runner = storage.load_dag()

    else:
        runner = DAGRunner(8)
        # Add all tasks first
        for i in range(50):
            task = FunctionTask(
                task_id=f"task_{i}",
                func=func_task,
                args=(2,),
                tags=["main"],
            )

            runner.add_task(f"task_{i}", task)

        # generate ramdom dependencies
        for i in range(50):
            # add random dependencies to node i
            n_of_deps = random.randint(0, min(i, 3))
            deps_list = list(range(0, i))
            deps_nums = random.choices(deps_list, k=n_of_deps)
            for dep_num in deps_nums:
                runner.add_dependency(f"task_{i}", [f"task_{dep_num}"])

    try:
        # Execute the DAG
        tags = ["main"]
        exec_context = {
            "date": "2025-10-01",
            "dry-run": False,
            "test": False,
            "config": {},
        }
        tick = time.perf_counter()
        runner.execute(tags=tags, exec_context=exec_context)
        logger.info(f"Execution time: {time.perf_counter() - tick}")
        logger.info("Execution results:")
        for task_id, result in runner.results.items():
            logger.info(f"{task_id}: {result}")

        logger.info("Execution success! Removing file")
        Path("dag.json").unlink(missing_ok=True)

    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted. Saving DAG...")
        storage.save_dag(runner)
        # force exit of main and all threads
        os._exit(1)
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        Path("dag.json").unlink(missing_ok=True)
        os._exit(1)


def example_usage() -> None:
    """Example usage of the DAGRunner class with separate node and edge creation."""
    # Create a new DAG runner
    storage = DAGStorage("dag.json")

    if Path("dag.json").exists():
        runner = storage.load_dag()

    else:
        runner = DAGRunner(3)
        # Add all tasks first
        task_a = FunctionTask(
            task_id="task_a",
            func=func_task,
            args=(5,),
            tags=["main"],
        )
        task_b = FunctionTask(
            task_id="task_b",
            func=func_task,
            args=(),
            tags=["main"],
            scheduling=Scheduling(active_days="wed"),
        )
        task_c = FunctionTask(
            task_id="task_c",
            func=func_task,
            args=(),
            tags=["main"],
        )

        task_d = FunctionTask(
            task_id="task_d",
            func=func_task,
            args=(),
            tags=["main"],
        )
        task_e = FunctionTask(
            task_id="task_e",
            func=func_task,
            args=(),
            tags=["main"],
        )
        task_f = FunctionTask(
            task_id="task_f",
            func=func_task,
            args=(),
            tags=["main"],
            scheduling=Scheduling(),
        )
        task_g = FunctionTask(
            task_id="task_g",
            func=func_task,
            args=(),
            tags=["sub"],
            scheduling=Scheduling(day=1),
        )
        task_h = FunctionTask(
            task_id="task_h",
            func=func_task,
            args=(),
            tags=["sub"],
        )
        runner.add_task("task_a", task_a)
        runner.add_task("task_b", task_b)
        runner.add_task("task_c", task_c)
        runner.add_task("task_d", task_d)
        runner.add_task("task_e", task_e)
        runner.add_task("task_f", task_f)
        runner.add_task("task_g", task_g)
        runner.add_task("task_h", task_h)
        # Then set up dependencies
        runner.add_dependency("task_c", ["task_a", "task_b"])
        runner.add_dependency("task_d", ["task_a"])
        runner.add_dependency("task_e", ["task_b"])
        runner.add_dependency("task_f", ["task_e"])
        # tag "sub"
        runner.add_dependency("task_h", ["task_g"])

    try:
        # Execute the DAG
        tags = ["main"]
        exec_context = {
            "date": "2025-10-01",
            "dry-run": False,
            "test": False,
            "config": {},
        }
        runner.execute(tags=tags, exec_context=exec_context)
        logger.info("Execution results:")
        for task_id, result in runner.results.items():
            logger.info(f"{task_id}: {result}")

        logger.info("Execution success! Removing file")
        Path("dag.json").unlink(missing_ok=True)

    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted. Saving DAG...")
        storage.save_dag(runner)
        # force exit of main and all threads
        os._exit(1)
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        Path("dag.json").unlink(missing_ok=True)
        os._exit(1)


if __name__ == "__main__":
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="DEBUG")
    example_usage2()
