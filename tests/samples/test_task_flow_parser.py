"""Tests for the task flow parser module."""

import pytest

from samples.dag_runner import FunctionTask
from samples.task_flow_parser import parse_task_expression


def test_object_task_index():
    """Test parsing with object task index."""
    func_task = FunctionTask(task_id="task1", func=lambda x: x, tags=["tag1", "tag2"])
    task_index = {
        "task1": func_task,
        "task2": "task2",
        "task3": "task3",
    }
    expression = "[task1, task2] >> task3"
    graph = parse_task_expression(expression, task_index)

    # Check nodes
    assert set(graph.nodes()) == {"task1", "task2", "task3"}

    # Check edges
    assert set(graph.edges()) == {("task1", "task3"), ("task2", "task3")}
    assert graph.nodes["task1"]["task"] == func_task
    assert graph.nodes["task1"]["task"].tags == ["tag1", "tag2"]
    assert graph.nodes["task2"]["task"] == "task2"
    assert graph.nodes["task3"]["task"] == "task3"


def test_parallel_tasks():
    """Test parsing parallel tasks."""
    expression = "[task1, task2] >> task3"
    graph = parse_task_expression(expression)

    # Check nodes
    assert set(graph.nodes()) == {"task1", "task2", "task3"}

    # Check edges
    assert set(graph.edges()) == {("task1", "task3"), ("task2", "task3")}


def test_sequential_tasks():
    """Test parsing sequential tasks."""
    expression = "task1 >> task2 >> task3"
    graph = parse_task_expression(expression)

    # Check nodes
    assert set(graph.nodes()) == {"task1", "task2", "task3"}

    # Check edges
    assert set(graph.edges()) == {("task1", "task2"), ("task2", "task3")}


def test_mixed_parallel_sequential():
    """Test parsing mixed parallel and sequential tasks."""
    expression = "task1 >> [task2, task3] >> task4"
    graph = parse_task_expression(expression)

    # Check nodes
    assert set(graph.nodes()) == {"task1", "task2", "task3", "task4"}

    # Check edges
    assert set(graph.edges()) == {
        ("task1", "task2"),
        ("task1", "task3"),
        ("task2", "task4"),
        ("task3", "task4"),
    }


def test_multiple_parallel_stages():
    """Test parsing multiple parallel stages."""
    expression = "[task1, task2] >> [task3, task4] >> task5"
    graph = parse_task_expression(expression)

    # Check nodes
    assert set(graph.nodes()) == {"task1", "task2", "task3", "task4", "task5"}

    # Check edges
    assert set(graph.edges()) == {
        ("task1", "task3"),
        ("task1", "task4"),
        ("task2", "task3"),
        ("task2", "task4"),
        ("task3", "task5"),
        ("task4", "task5"),
    }


def test_complex_workflow():
    """Test parsing a complex workflow."""
    expression = "start >> [parallel1, parallel2] >> process >> [end1, end2]"
    graph = parse_task_expression(expression)

    # Check nodes
    expected_nodes = {"start", "parallel1", "parallel2", "process", "end1", "end2"}
    assert set(graph.nodes()) == expected_nodes

    # Check edges
    expected_edges = {
        ("start", "parallel1"),
        ("start", "parallel2"),
        ("parallel1", "process"),
        ("parallel2", "process"),
        ("process", "end1"),
        ("process", "end2"),
    }
    assert set(graph.edges()) == expected_edges


def test_invalid_expressions():
    """Test parsing invalid expressions raises appropriate exceptions."""
    # Missing '>>' operator
    with pytest.raises(ValueError, match="must contain at least one '>>' operator"):
        parse_task_expression("task1 task2")

    # Empty expression
    with pytest.raises(ValueError, match="must contain at least one '>>' operator"):
        parse_task_expression("")




def test_whitespace_handling():
    """Test that whitespace is handled correctly."""
    expression = "  task1  >>  [  task2  ,  task3  ]  >>  task4  "
    graph = parse_task_expression(expression)

    # Check nodes (whitespace should be stripped)
    assert set(graph.nodes()) == {"task1", "task2", "task3", "task4"}

    # Check edges
    assert set(graph.edges()) == {
        ("task1", "task2"),
        ("task1", "task3"),
        ("task2", "task4"),
        ("task3", "task4"),
    }
