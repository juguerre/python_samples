import time
from datetime import datetime
from typing import Any, Callable

import pytest

from samples.dag_runner import (
    CallBackPublisher,
    FunctionTask,
    SampleBackendCallableClient,
    Scheduling,
    TaskDAG,
    TaskDAGExecutor,
)


@pytest.fixture
def sample_scheduled_dag_factory() -> Callable[[dict[str, Any]], TaskDAG]:
    def factory(exec_context: dict[str, Any]) -> TaskDAG:
        dag = TaskDAG("test_dag.json")

        func = SampleBackendCallableClient(exec_context=exec_context)

        # Create tasks
        task1 = FunctionTask(
            task_id="task1",
            _cached_func=func,
            scheduling=Scheduling(),
        )
        task2 = FunctionTask(
            task_id="task2",
            _cached_func=func,  # Always fails
            scheduling=Scheduling(),
        )
        task3 = FunctionTask(
            task_id="task3",
            _cached_func=func,  # Never fails
            scheduling=Scheduling(),
        )
        task4 = FunctionTask(
            task_id="task4",
            _cached_func=func,  # Never fails
            scheduling=Scheduling(),
        )

        # Add tasks to DAG
        dag.add_task("task1", task1)
        dag.add_task("task2", task2)
        dag.add_task("task3", task3)
        dag.add_task("task4", task4)

        # Add dependencies
        dag.add_dependency("task2", ["task1"])
        dag.add_dependency("task3", ["task1"])
        dag.add_dependency("task4", ["task1"])

        return dag
    return factory


def test_scheduling_performance(sample_scheduled_dag_factory: Callable[[dict[str, Any]], TaskDAG]):
    exec_context = {"date": datetime.now().isoformat(), "callback_publisher": CallBackPublisher()}
    dag = sample_scheduled_dag_factory(exec_context)
    executor = TaskDAGExecutor(dag, max_workers=2)
    start_time = time.perf_counter()
    executor.execute(exec_context=exec_context)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time} seconds")
