import asyncio
from prefect.workers.process import ProcessWorker

# read .env with dotenv
from dotenv import load_dotenv
load_dotenv()

# --- Configuration matching the CLI command ---
POOL_NAME = "pool1"
WORKER_NAME = "worker-02"
CONCURRENCY_LIMIT = 5
# ---------------------------------------------


async def start_worker():
    """Starts a ProcessWorker instance configured to poll the specified pool."""
    print(f"Starting Process Worker '{WORKER_NAME}' for pool '{POOL_NAME}'...")

    # 1. Instantiate the specific worker class
    worker = ProcessWorker(
        name=WORKER_NAME,
        work_pool_name=POOL_NAME,
        limit=CONCURRENCY_LIMIT,  # The concurrency limit
        # The 'type' is implicitly 'process' since we instantiated ProcessWorker
    )

    # 2. Start the worker (this is the blocking call that polls the API)
    # This will block the thread/process until interrupted (e.g., Ctrl+C)
    try:
        await worker.start()
    except asyncio.CancelledError:
        print(f"Worker {WORKER_NAME} stopped.")


if __name__ == "__main__":
    # Ensure your PREFECT_API_URL is set in the environment before running
    # Example: export PREFECT_API_URL="http://localhost:4200/api"
    print("Ensure PREFECT_API_URL is set to connect to the Prefect server.")
    asyncio.run(start_worker())
