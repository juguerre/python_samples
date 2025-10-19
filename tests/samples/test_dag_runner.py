"""Tests for the DAG runner module."""
import os
from datetime import datetime

import pytest

from samples.dag_runner import (
    FunctionTask,
    Scheduling,
    TaskDAG,
    TaskDAGExecutor,
    TaskDAGFilter,
    TaskStatus,
)


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
def sample_dag():
    """Create a sample DAG for testing."""
    dag = TaskDAG("test_dag.json")

    # Create tasks
    task1 = FunctionTask(task_id="task1", func=lambda: 1, tags=["test"])
    task2 = FunctionTask(
        task_id="task2", func=lambda x: x + 1, args=(1,), tags=["test"]
    )
    task3 = FunctionTask(
        task_id="task3", func=lambda x: x * 2, args=(2,), tags=["prod"]
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
    result = executor.execute(exec_context={"date": datetime.now().isoformat()}, tags=["test"])

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


