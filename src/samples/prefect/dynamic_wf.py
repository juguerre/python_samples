import time

from prefect import flow
from prefect.tasks import task


@task
def task_1(prev_result: str) -> str:
    time.sleep(5)
    return "task_1 done!"


@task
def task_2(prev_result: str) -> str:
    time.sleep(2)
    return "task_2 done!"


@flow(
    name="dynamic_flow",
    description="A dynamic workflow",
    version="1.0.0",
    retries=3,
    retry_delay_seconds=5,
    timeout_seconds=60,
    log_prints=True,
)
def dynamic_flow(tasks: list[str]) -> str:
    # executing task using globals
    resps: list[str] = []
    previous_res = None
    for task_name in tasks:
        resps.append(globals()[task_name](previous_res))
        previous_res = resps[-1]
    return "\n".join(resps)


if __name__ == "__main__":
    dynamic_flow(["task_2", "task_1"])
    dynamic_flow(["task_1", "task_2"])
