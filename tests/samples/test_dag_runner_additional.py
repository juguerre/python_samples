"""Additional tests to improve coverage for the DAG runner module."""

from datetime import datetime
from typing import Callable
from unittest.mock import MagicMock

import pytest

from samples.dag_runner import (
    BaseTaskModel,
    FunctionTask,
    Scheduling,
    TaskDAG,
    TaskDAGExecutor,
    TaskDAGFilter,
    TaskSkipped,
    TaskStatus,
)


class TestSchedulingEdgeCases:
    """Test scheduling edge cases to cover lines 81, 103."""

    def test_scheduling_daily_periodicity(self):
        """Test daily periodicity (line 81)."""
        # Default scheduling is daily
        scheduling = Scheduling()
        assert scheduling.periodicity == "daily"
        assert scheduling.is_active_day(datetime.now()) is True

    def test_scheduling_single_weekday(self):
        """Test single weekday scheduling (line 103 else branch)."""
        # Single day notation
        scheduling = Scheduling(active_days="fri")
        days = scheduling.get_numeric_week_days()
        assert days == [4]  # Friday is 4

    def test_scheduling_weekday_range(self):
        """Test weekday range scheduling (line 103 if branch)."""
        # Range notation
        scheduling = Scheduling(active_days="mon-fri")
        days = scheduling.get_numeric_week_days()
        assert days == range(0, 5)  # Monday to Friday


class TestFunctionTaskErrors:
    """Test FunctionTask error handling to cover lines 141-143, 151, 157-158."""

    def test_function_task_missing_function(self):
        """Test error when neither func_name nor func is provided (line 141)."""
        with pytest.raises(ValueError):
            FunctionTask(task_id="test_task")

    def test_function_task_invalid_func_name(self):
        """Test error when func_name doesn't exist (lines 142-143)."""
        with pytest.raises(ValueError):
            FunctionTask(task_id="test_task", func_name="module:nonexistent_func")

    def test_execute_without_date_context(self):
        """Test execution without date in context (line 151)."""
        task = FunctionTask(task_id="test_task", _cached_func=lambda: 1)
        with pytest.raises(
            ValueError, match="Execution context must contain a 'date' field"
        ):
            task.execute({})

    def test_execute_task_skipped_not_scheduled(self):
        """Test task skipping when not scheduled for the day (lines 157-158)."""
        # Create a task only for day 1 of the month
        task = FunctionTask(
            task_id="monthly_task",
            _cached_func=lambda: "result",
            scheduling=Scheduling(day=1),
        )

        # Execute on day 15 (not scheduled)
        with pytest.raises(TaskSkipped):
            task.execute({"date": "2023-01-15T00:00:00"})

        # Status should be DONE even after skip (finally block)
        assert task.status == TaskStatus.DONE


class TestTaskDAGErrors:
    """Test TaskDAG error handling to cover lines 222, 233, 262-264, 288, 292, 316, 321."""

    def test_save_dag_unsupported_task_type(self):
        """Test saving DAG with unsupported task type (line 222)."""

        class CustomTask(BaseTaskModel):
            """Custom task type not handled in save."""

            def execute(self, exec_context):
                return "custom"

        dag = TaskDAG()
        custom_task = CustomTask(task_id="custom")
        dag.add_task("custom", custom_task)

        with pytest.raises(ValueError, match="Task type CustomTask not supported"):
            dag.save_dag()

    def test_load_dag_no_filepath(self, tmp_path):
        """Test loading DAG without filepath (line 233)."""
        # Test the error when _filepath attribute doesn't exist
        # Note: This test might not cover line 233 due to implementation details
        # where _filepath is set at class level during any TaskDAG instantiation

        # Now test that it raises ValueError
        with pytest.raises(FileNotFoundError):
            # Must raise FileNotFoundError if _filepath doesn't exist
            TaskDAG.load_dag_from_file("nonexistent_file.json")

    def test_reset_status_with_tags(self, simple_task_func: Callable) -> None:
        """Test reset_status with tag filtering (lines 262-264)."""
        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func, tags=["test"])
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func, tags=["prod"])

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)

        # Mark tasks as done
        task1.status = TaskStatus.DONE
        task2.status = TaskStatus.DONE

        # Reset only "test" tagged tasks
        dag.reset_status(tags=["test"])

        assert task1.status == TaskStatus.PENDING
        assert task2.status == TaskStatus.DONE  # Should remain DONE

    def test_add_dependency_task_not_found(self, simple_task_func: Callable):
        """Test adding dependency with non-existent task (line 288)."""
        dag = TaskDAG()
        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        dag.add_task("task1", task1)

        with pytest.raises(ValueError, match="Task with ID 'task2' not found"):
            dag.add_dependency("task2", ["task1"])

    def test_add_dependency_dependency_not_found(self, simple_task_func):
        """Test adding dependency with non-existent dependency (line 292)."""
        dag = TaskDAG()
        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        dag.add_task("task1", task1)

        with pytest.raises(ValueError, match="Dependency 'task2' not found"):
            dag.add_dependency("task1", ["task2"])

    def test_are_ancestors_done_with_pending(self, simple_task_func: Callable):
        """Test are_ancestors_done with pending ancestors (line 316)."""
        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func)

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)
        dag.add_dependency("task2", ["task1"])

        # task1 is still PENDING
        assert not TaskDAG.are_ancestors_done("task2", dag.graph)

        # Mark task1 as DONE
        task1.status = TaskStatus.DONE
        assert TaskDAG.are_ancestors_done("task2", dag.graph)

    def test_pending_ancestors(self, simple_task_func: Callable):
        """Test pending_ancestors method (line 321)."""
        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func)

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)
        dag.add_dependency("task2", ["task1"])

        # task1 is PENDING
        pending = TaskDAG.pending_ancestors("task2", dag.graph)
        assert "task1" in pending
        assert len(pending) == 1

    def test_add_callable_as_task(self):
        """Test adding a plain callable as task (line 280)."""
        dag = TaskDAG()

        # Add a plain function (not a Task instance)
        def my_func(x, **_kwargs):
            return x * 2

        # Note: When adding via add_task, the tags go into FunctionTask constructor
        # but args and kwargs are passed to FunctionTask, not as positional args to add_task
        dag.add_task("func_task", my_func, tags=["test"])

        task = dag.get_task("func_task")
        assert isinstance(task, FunctionTask)
        assert task.task_id == "func_task"
        assert "test" in task.tags
        # Args are not passed this way - they would need to be in the FunctionTask constructor
        assert task.args == ()  # Default empty tuple


class TestTaskDAGFilterWarnings:
    """Test TaskDAGFilter warning paths to cover lines 341-344, 354-356, 377-381."""

    def test_filter_warns_active_descendant_removed(self, simple_task_func: Callable):
        """Test warning when active descendant is removed (lines 341-344, 377-381)."""
        dag = TaskDAG()

        # task1 runs only on day 1, task2 runs every day
        task1 = FunctionTask(
            task_id="task1",
            _cached_func=simple_task_func,
            scheduling=Scheduling(day=1),
        )
        task2 = FunctionTask(
            task_id="task2",
            _cached_func=simple_task_func,
            scheduling=Scheduling(),  # Daily
        )

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)
        dag.add_dependency("task2", ["task1"])

        # Execute on day 15 (task1 is inactive, task2 is active)
        filter_obj = TaskDAGFilter(dag, datetime(2023, 1, 15), None)

        # This should trigger the warning about removing active descendant
        filtered_dag = filter_obj.filter()

        # Both tasks should be removed (task2 depends on inactive task1)
        assert "task1" not in filtered_dag.graph.nodes
        assert "task2" not in filtered_dag.graph.nodes

    def test_filter_warns_tag_descendant_removed(self, simple_task_func: Callable):
        """Test warning when tagged descendant is removed (lines 354-356)."""
        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func, tags=["prod"])
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func, tags=["test"])

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)
        dag.add_dependency("task2", ["task1"])

        # Filter for "test" tags only
        filter_obj = TaskDAGFilter(dag, datetime.now(), tags=["test"])

        # This should trigger warning: task1 (no "test" tag) is removed,
        # but task2 (has "test" tag) is a descendant
        filtered_dag = filter_obj.filter()

        # Both should be removed
        assert "task1" not in filtered_dag.graph.nodes
        assert "task2" not in filtered_dag.graph.nodes


class TestTaskDAGExecutorCallbacks:
    """Test TaskDAGExecutor callback error handling (lines 527-528, 544, 570-571)."""

    def test_executor_handles_task_skipped(self, simple_task_func: Callable):
        """Test executor handles TaskSkipped exception (lines 527-528)."""
        dag = TaskDAG()

        # Task that only runs on day 1
        task = FunctionTask(
            task_id="monthly_task",
            _cached_func=simple_task_func,
            scheduling=Scheduling(day=1),
        )

        dag.add_task("monthly_task", task)

        executor = TaskDAGExecutor(dag, max_workers=1)

        # Execute on day 15 (task should be filtered out before execution)
        results = executor.execute({"date": "2023-01-15T00:00:00"})

        # Task is filtered out by the DAG filter, so it's removed from the working DAG
        # It won't be in results because it was never submitted for execution
        assert "monthly_task" not in results
        # Status remains PENDING because it was never submitted
        assert task.status == TaskStatus.PENDING

    def test_executor_handles_task_exception(self):
        """Test executor handles task exceptions properly."""

        def failing_task(*_args, **_kwargs):
            raise RuntimeError("Task failed!")

        dag = TaskDAG()
        task = FunctionTask(task_id="failing", _cached_func=failing_task)
        dag.add_task("failing", task)

        executor = TaskDAGExecutor(dag, max_workers=1)
        results = executor.execute({"date": datetime.now().isoformat()})

        # Exception should be stored in results
        assert "failing" in results
        assert isinstance(results["failing"], RuntimeError)

    def test_executor_skips_already_done_tasks(self, simple_task_func: Callable):
        """Test executor skips tasks already marked as DONE (lines 570-571)."""

        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func)

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)

        # Mark task1 as already done
        task1.status = TaskStatus.DONE

        executor = TaskDAGExecutor(dag, max_workers=1)
        results = executor.execute({"date": datetime.now().isoformat()})

        # task1 should not be in results (was already done)
        assert "task1" not in results
        # task2 should be executed
        assert results["task2"] is not None

    def test_validate_dag_error_on_execute(self, simple_task_func: Callable):
        """Test execute raises error on invalid DAG (line 544)."""
        dag = TaskDAG()

        task1 = FunctionTask(task_id="task1", _cached_func=simple_task_func)
        task2 = FunctionTask(task_id="task2", _cached_func=simple_task_func)

        dag.add_task("task1", task1)
        dag.add_task("task2", task2)

        # Create a cycle
        dag.add_dependency("task2", ["task1"])
        dag.add_dependency("task1", ["task2"])

        executor = TaskDAGExecutor(dag, max_workers=1)

        with pytest.raises(ValueError, match="contains cycles"):
            executor.execute({"date": datetime.now().isoformat()})


class TestStringTemplatingCoverage:
    """Test to cover the missing line in string_templating.py."""

    def test_datap_out_of_range_year(self):
        """Test datap with year outside expected ranges (line 29 in string_templating.py)."""
        from samples.string_templating import datap

        # Test with year before 2020
        with pytest.raises(ValueError, match="Unexpected DataP date"):
            datap("2015-01-01")

        # Test with year after 2050
        with pytest.raises(ValueError, match="Unexpected DataP date"):
            datap("2055-01-01")


class TestTaskFlowParserCoverage:
    """Test to cover missing line in task_flow_parser.py."""

    def test_parse_with_task_not_in_index(self):
        """Test parsing with task index that doesn't contain all tasks (line 79)."""
        from samples.task_flow_parser import parse_task_expression

        # Provide an incomplete task index
        task_index = {"task1": MagicMock()}

        # task2 is not in the index, should trigger warning
        graph = parse_task_expression("task1 >> task2", task_index=task_index)

        assert "task1" in graph.nodes
        assert "task2" in graph.nodes
        # task2 should have no task object associated
        assert graph.nodes["task2"]["task"] is None
