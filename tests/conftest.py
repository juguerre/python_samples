from collections.abc import Callable

import pytest
from icecream import ic


@pytest.fixture(scope="module")
def simple_task_func() -> Callable[..., str]:
    def simple_task(*_args, task_id: str, **_kwargs) -> str:
        ic(f"Executing task {task_id}")
        return f"Result for {task_id}!"

    return simple_task
