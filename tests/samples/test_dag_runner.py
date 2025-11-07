"""Tests for the DAG runner module."""

import math
import os
import random
import time
from datetime import datetime, timedelta
from typing import Callable, Literal

import pytest

from samples.dag_runner import (
    FunctionTask,
    Scheduling,
    TaskDAG,
    TaskDAGExecutor,
    TaskDAGFilter,
    TaskStatus,
)


def get_next_weekday(
    weekday: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ],
) -> datetime:
    """
    Get the next occurrence of the specified weekday.

    :param weekday: The target weekday (full name in lowercase)
    :type weekday: Literal['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    :return: A datetime object representing the next occurrence of the specified weekday at midnight
    :rtype: datetime
    """
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    target_weekday = weekday_map[weekday.lower()]
    today = datetime.now().date()
    days_ahead = (target_weekday - today.weekday()) % 7
    # If today is the target day and it's not past midnight yet, stay on today
    if days_ahead == 0 and datetime.now().time() < datetime.min.time():
        days_ahead = 0
    elif (
        days_ahead == 0
    ):  # If today is the target day but past midnight, go to next week
        days_ahead = 7

    next_date = datetime.now() + timedelta(days=days_ahead)
    return next_date.replace(hour=0, minute=0, second=0, microsecond=0)


def test_create_task():
    """Test creating a basic task."""
    task = FunctionTask(
        task_id="test_task",
        func=lambda: "test",
        tags=["test"],
    )
    assert task.task_id == "test_task"
    assert task.status == TaskStatus.PENDING
    assert "test" in task.tags


def test_add_task_to_dag():
    """Test adding a task to the DAG."""
    dag = TaskDAG("test_dag.json")
    task = FunctionTask(task_id="task1", func=lambda: 1)
    dag.add_task("task1", task)

    assert "task1" in dag.graph.nodes
    assert dag.get_task("task1") == task


def test_add_dependency():
    """Test adding a dependency between tasks."""
    dag = TaskDAG("test_dag.json")
    task1 = FunctionTask(task_id="task1", func=lambda: 1)
    task2 = FunctionTask(task_id="task2", func=lambda: 2)

    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_dependency("task2", ["task1"])

    assert "task1" in dag.graph.predecessors("task2")


def test_validate_dag():
    """Test DAG validation."""
    dag = TaskDAG("test_dag.json")
    task1 = FunctionTask(task_id="task1", func=lambda: 1)
    task2 = FunctionTask(task_id="task2", func=lambda: 2)

    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_dependency("task2", ["task1"])

    assert dag.validate()


def test_detect_cycle():
    """Test cycle detection in the DAG."""
    dag = TaskDAG("test_dag.json")
    task1 = FunctionTask(task_id="task1", func=lambda: 1)
    task2 = FunctionTask(task_id="task2", func=lambda: 2)

    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_dependency("task2", ["task1"])
    dag.add_dependency("task1", ["task2"])  # Creates a cycle

    assert not dag.validate()


def test_task_execution():
    """Test executing a simple task."""
    task = FunctionTask(
        task_id="test_task",
        func=lambda x: x * 2,
        args=(2,),
    )
    result = task.execute({"date": datetime.now().isoformat()})
    assert result == 4
    assert task.status == TaskStatus.DONE


def test_filter_by_tags():
    """Test filtering tasks by tags."""
    dag = TaskDAG("test_dag.json")
    task1 = FunctionTask(task_id="task1", func=lambda: 1, tags=["test"])
    task2 = FunctionTask(task_id="task2", func=lambda: 2, tags=["prod"])

    dag.add_task("task1", task1)
    dag.add_task("task2", task2)

    # Filter for test tasks
    filter_ = TaskDAGFilter(dag, datetime.now(), ["test"])
    filtered_dag = filter_.filter()

    assert "task1" in filtered_dag.graph.nodes
    assert "task2" not in filtered_dag.graph.nodes


def test_scheduling():
    """Test task scheduling."""
    # Create a task that only runs on the 1st of the month
    task = FunctionTask(
        task_id="monthly_task",
        func=lambda: "monthly",
        scheduling=Scheduling(day=1),
    )

    # Test on the 1st of the month
    exec_date = datetime(2023, 1, 1)
    assert task.scheduling.is_active_day(exec_date) is True

    # Test on a different day
    exec_date = datetime(2023, 1, 2)
    assert task.scheduling.is_active_day(exec_date) is False


def test_task_dag_executor():
    """Test the DAG executor with multiple tasks."""
    dag = TaskDAG("test_dag.json")

    # Create tasks
    task1 = FunctionTask(task_id="task1", func=lambda: 1)
    task2 = FunctionTask(task_id="task2", func=lambda x: x + 1, args=(1,))

    # Add tasks to DAG
    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_dependency("task2", ["task1"])

    # Execute DAG
    executor = TaskDAGExecutor(dag, max_workers=2)
    results = executor.execute(exec_context={"date": datetime.now().isoformat()})

    # Verify results
    assert results["task1"] == 1
    assert results["task2"] == 2
    assert dag.get_task("task1").status == TaskStatus.DONE
    assert dag.get_task("task2").status == TaskStatus.DONE


def test_save_and_load_dag(tmp_path):
    """Test saving and loading a DAG to/from a file."""
    # Create and save a DAG
    filepath = tmp_path / "test_dag.json"
    dag = TaskDAG(str(filepath))

    task = FunctionTask(task_id="test_task", func=lambda: 1)
    dag.add_task("test_task", task)
    dag.save_dag()

    # Load the DAG
    loaded_dag = TaskDAG.load_dag(str(filepath))

    # Verify the loaded DAG
    assert "test_task" in loaded_dag.graph.nodes
    assert loaded_dag.get_task("test_task").task_id == "test_task"
    assert loaded_dag.get_task("test_task").status == TaskStatus.PENDING


# Fixture for common test DAG
@pytest.fixture
def sample_dag(func_factory: Callable[[int, float], Callable[[], None]]):
    """Create a sample DAG for testing."""
    dag = TaskDAG("test_dag.json")

    # Create tasks
    task1 = FunctionTask(
        task_id="task1",
        func=func_factory(0, 0.0),  # Never fails
        tags=["test"],
    )
    task2 = FunctionTask(
        task_id="task2",
        func=func_factory(0, 1.0),  # Always fails
        args=(1,),
        tags=["test"],
    )
    task3 = FunctionTask(
        task_id="task3",
        func=lambda x: x * 2,
        args=(2,),
        tags=["prod"],
    )

    # Add tasks to DAG
    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_task("task3", task3)

    # Add dependencies
    dag.add_dependency("task2", ["task1"])
    dag.add_dependency("task3", ["task2"])

    return dag


def generate_cpu_load(duration_seconds: float) -> None:
    start_time = time.time()
    while (time.time() - start_time) < duration_seconds:
        # Perform CPU-intensive calculations
        x = 0
        for i in range(1000000):
            x += math.sqrt(i)


@pytest.fixture
def func_factory() -> Callable:
    def get_task_func(max_time: int, exception_prob: float) -> Callable[[], None]:
        def func():
            generate_cpu_load(random.random() * max_time)
            if random.random() < exception_prob:
                raise Exception("Task failed")

        return func

    return get_task_func


@pytest.fixture
def sample_dag_factory(func_factory: Callable) -> Callable[[int, int, float], TaskDAG]:
    """Create a sample dag with a size of n tasks"""

    def _n_size_sample_dag(n: int, max_sleep: int, exception_prob: float) -> TaskDAG:
        dag = TaskDAG("test_dag.json")
        for i in range(n):
            task = FunctionTask(
                task_id=f"task{i}",
                func=func_factory(max_sleep, exception_prob),
                tags=["test"],
            )
            dag.add_task(f"task{i}", task)
        # generate random dependencies with no cycles
        for i in range(n):
            # add random dependencies to node i
            n_of_deps = random.randint(0, min(i, 3))
            deps_list = list(range(0, i))
            deps_nums = random.choices(deps_list, k=n_of_deps)
            for dep_num in deps_nums:
                dag.add_dependency(f"task{i}", [f"task{dep_num}"])

        return dag

    return _n_size_sample_dag


@pytest.fixture
def sample_scheduled_dag(
    func_factory: Callable[[int, float], Callable[[], None]],
) -> TaskDAG:
    """Create a sample DAG for testing."""
    dag = TaskDAG("test_dag.json")

    # Create tasks
    task1 = FunctionTask(
        task_id="task1",
        func=func_factory(1, 0.0),  # Never fails
        tags=["test"],
        scheduling=Scheduling(day=1),
    )
    task2 = FunctionTask(
        task_id="task2",
        func=func_factory(1, 1.0),  # Always fails
        args=(1,),
        tags=["test"],
        scheduling=Scheduling(active_days="wed"),
    )
    task3 = FunctionTask(
        task_id="task3",
        func=func_factory(1, 0.0),  # Never fails
        args=(2,),
        tags=["prod"],
        scheduling=Scheduling(),
    )

    # Add tasks to DAG
    dag.add_task("task1", task1)
    dag.add_task("task2", task2)
    dag.add_task("task3", task3)

    # Add dependencies
    dag.add_dependency("task2", ["task1"])
    dag.add_dependency("task3", ["task2"])

    return dag


def test_dag_execution_order(sample_dag):
    """Test that tasks are executed in the correct order."""
    # Execute DAG
    executor = TaskDAGExecutor(sample_dag, max_workers=1)
    # result is an OrderectDict and maintains the order of execution
    result = executor.execute(exec_context={"date": datetime.now().isoformat()})

    # Verify execution order (task1 -> task2 -> task3)
    assert list(result.keys()).index("task1") < list(result.keys()).index("task2")
    assert list(result.keys()).index("task2") < list(result.keys()).index("task3")


def test_dag_execution_order_with_test_tag(sample_dag):
    """Test that tasks are executed in the correct order."""
    # Execute DAG
    executor = TaskDAGExecutor(sample_dag, max_workers=1)
    # result is an OrderectDict and maintains the order of execution
    result = executor.execute(
        exec_context={"date": datetime.now().isoformat()}, tags=["test"]
    )

    # Verify task3 is not executed (taged with "prod" instead of test)

    assert len(result) == 2


def test_dag_save_and_load(sample_dag):
    """Test saving and loading a DAG to/from a file."""
    # Create and save a DAG
    filepath = "test_dag.json"
    sample_dag.save_dag()

    # Load the DAG
    loaded_dag = TaskDAG.load_dag(filepath)

    assert loaded_dag.graph.number_of_nodes() == sample_dag.graph.number_of_nodes()
    assert loaded_dag.graph.number_of_edges() == sample_dag.graph.number_of_edges()
    assert loaded_dag.get_task("task1").task_id == "task1"
    assert loaded_dag.get_task("task2").task_id == "task2"
    assert loaded_dag.get_task("task3").task_id == "task3"
    # remove file
    os.remove(filepath)


def test_scheduling_periodicity(sample_scheduled_dag):
    assert (
        sample_scheduled_dag.get_task("task1").scheduling.is_active_day(
            datetime.now().replace(day=1)
        )
        is True
    )
    assert (
        sample_scheduled_dag.get_task("task1").scheduling.is_active_day(
            datetime.now().replace(day=2)
        )
        is False
    )
    assert (
        sample_scheduled_dag.get_task("task2").scheduling.is_active_day(
            get_next_weekday("wednesday")
        )
        is True
    )
    assert (
        sample_scheduled_dag.get_task("task2").scheduling.is_active_day(
            get_next_weekday("monday")
        )
        is False
    )

    assert sample_scheduled_dag.get_task("task1").scheduling.periodicity == "monthly"
    assert sample_scheduled_dag.get_task("task2").scheduling.periodicity == "weekly"


@pytest.mark.skip
def test_scheduling_performance(
    sample_dag_factory: Callable[[int, int, float], TaskDAG],
):
    dag = sample_dag_factory(100, 5, 0.1)
    executor = TaskDAGExecutor(dag, max_workers=16)
    start_time = time.perf_counter()
    executor.execute(exec_context={"date": datetime.now().isoformat()})
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time} seconds")
