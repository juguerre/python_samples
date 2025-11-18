from typing import Any

import loguru
import networkx as nx
from networkx import DiGraph

logger = loguru.logger


# noinspection PyPep8Naming
def parse_task_expression(
    expression: str, task_index: dict[str, Any] | None = None
) -> nx.DiGraph:
    """
    Parse a task dependency expression and return a NetworkX directed graph.

    ::

        Supports syntax like:
        - "[task1, task2] >> task3" - parallel tasks leading to one
        - "task1 >> task2 >> task3" - sequential chain
        - "task1 >> [task2, task3] >> task4" - mixed sequential and parallel
        - Tasks inside [] are parallel (no dependencies between them)
        - Multiple >> operators create sequential stages

    :param expression: Task dependency expression
    :param task_index: Optional task index to resolve task names to task objects
    :returns networkx.DiGraph: Directed graph representing task dependencies
    """
    # noinspection PyPep8Naming
    task_index = task_index or {}
    G = nx.DiGraph()

    # Split by '>>' to get all stages
    stages = [stage.strip() for stage in expression.split(">>")]

    if len(stages) < 2:
        raise ValueError("Expression must contain at least one '>>' operator")

    # Parse each stage to get task groups
    task_groups = []
    for stage in stages:
        tasks = _parse_side(stage)
        if tasks:
            task_groups.append(tasks)

    # Add all tasks as nodes
    G = _add_tasks_to_graph(G, task_groups, task_index)

    # Create edges between consecutive stages
    # Each task in stage N connects to each task in stage N+1
    G = _add_edgest_to_graph(G, task_groups)

    return G


# noinspection PyPep8Naming
def _add_edgest_to_graph(G: DiGraph, task_groups: list[Any]) -> DiGraph:
    gx = G.copy()
    for i in range(len(task_groups) - 1):
        current_group = task_groups[i]
        next_group = task_groups[i + 1]

        for current_task in current_group:
            for next_task in next_group:
                gx.add_edge(current_task, next_task)
    return gx


# noinspection PyPep8Naming
def _add_tasks_to_graph(
    G: DiGraph, task_groups: list[Any], task_index: dict[str, Any]
) -> DiGraph:
    gx = G.copy()
    for group in task_groups:
        for task in group:
            task_obj = task_index.get(task)
            if not task_obj and task_index:
                logger.warning(f"Task '{task}' not found in task index")
            gx.add_node(task, task=task_obj)
    return gx


def _parse_side(side: str) -> list[str]:
    """
    Parse one side of the expression (left or right of >>).

    Returns a list of task ids.
    """
    side = side.strip()

    # Check if it's a list [task1, task2, ...]
    if side.startswith("[") and side.endswith("]"):
        # Remove brackets and split by comma
        content = side[1:-1]
        tasks = [task.strip() for task in content.split(",")]
        return [task for task in tasks if task]  # Filter empty strings
    else:
        # Single task
        return [side] if side else []
