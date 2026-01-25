import datetime
from functools import wraps
from typing import Callable

from prefect import flow, task

context: dict = {"date": datetime.date.today().isoformat()}


def context_flow(**kwargs):
    """Decorator to add runtime_context to flow"""

    def decor(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs2):
            if kwargs2.get("runtime_context"):
                runtime_context = dict(context, **kwargs2["runtime_context"])
            else:
                runtime_context = context
            kwargs2.pop("runtime_context", None)
            return func(*args, **kwargs2, runtime_context=runtime_context)

        # noinspection PyTypeChecker
        return flow(wrapper, **kwargs)

    return decor


@context_flow(log_prints=True)
def generic_etl_flow(*, runtime_context: dict | None = None):
    print(f"runtime_context: {runtime_context}")


@task
def elastic_extract():
    print("elastic")


@task
def csv_extract():
    pass


@task
def python_extract():
    pass


@task
def python_transform():
    pass


@task
def db_load():
    pass
