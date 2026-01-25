from prefect import flow
from prefect.logging.loggers import print_as_log


@flow(
    name="dummy_wf",
    description="A dummy workflow",
    version="1.0.0",
    retries=3,
    retry_delay_seconds=5,
    timeout_seconds=60,
    log_prints=True,
)
def my_workflow(name: str = "Andrew") -> str:
    print(f"Hello, world {name}!")
    print_as_log(f"Hello, world {name}!")
    return "Hello, world!"


if __name__ == "__main__":
    my_workflow()
    my_workflow(name="John")
