import time

import networkx as nx
from prefect import flow
from prefect.futures import PrefectFuture
from prefect.tasks import task
from prefect.logging import get_run_logger

from samples.task_flow_parser import parse_task_expression


@task
def task_1(prev_result: str) -> str:
    logger = get_run_logger()
    logger.info(f"Task 1: Prev Result {prev_result}")
    time.sleep(5)
    return "task_1 done!"


@task
def task_2(prev_result: str) -> str:
    logger = get_run_logger()
    logger.info(f"Task 2: Prev Result {prev_result}")
    time.sleep(2)
    return "task_2 done!"


@task
def task_3(prev_result: str) -> str:
    logger = get_run_logger()
    logger.info(f"Task 3: Prev Result {prev_result}")
    time.sleep(2)
    return "task_3 done!"


@flow(
    name="dynamic_dag_flow",
    description="A dynamic workflow",
    version="1.0.0",
    retries=3,
    retry_delay_seconds=5,
    timeout_seconds=60,
    log_prints=True,
)
def dynamic_dag_flow(dag_def: str) -> str | None:
    # executing task using globals
    futures: list[PrefectFuture] = []
    previous_res = None
    dag = parse_task_expression(dag_def)

    for generation in nx.topological_generations(dag):
        for task_id in list(generation):
            futures.append(globals()[task_id].submit(previous_res))
        # wait for results of this generation and generate previous_res
        previous_res = ":".join([f.result() for f in futures])

    return previous_res


if __name__ == "__main__":
    dynamic_dag_flow("task_1 >> [task_2, task_3]")
