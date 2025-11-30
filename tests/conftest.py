from typing import Callable

import pytest


@pytest.fixture(scope="module")
def simple_task_func() -> Callable[..., str]:
    def simple_task(*_args, task_id: str, **_kwargs) -> str:
        print(f"Executing task {task_id}")
        return f"Result for {task_id}!"

    return simple_task
