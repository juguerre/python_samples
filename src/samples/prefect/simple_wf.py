from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from prefect import flow, task


@task(retries=3, retry_delay_seconds=[2, 5, 15])
def fetch_page(page: int, api_base: str, per_page: int) -> list[dict[str, Any]]:
    """Return a list of article dicts for a given page number."""
    url = f"{api_base}/articles"
    params = {"page": page, "per_page": per_page}
    print(f"Fetching page {page} …")
    response = httpx.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@task
def to_dataframe(raw_articles: list[list[dict[str, Any]]]) -> pd.DataFrame:
    """Flatten & normalise JSON into a tidy DataFrame."""
    # Combine pages, then select fields we care about
    records = [article for page in raw_articles for article in page]
    df = pd.json_normalize(records)[
        [
            "id",
            "title",
            "published_at",
            "url",
            "comments_count",
            "positive_reactions_count",
            "tag_list",
            "user.username",
        ]
    ]
    return df


@task
def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist DataFrame to disk then log a preview."""
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows ➜ {path}\n\nPreview:\n{df.head()}\n")


@flow(name="devto_etl", log_prints=True)
def etl(api_base: str, pages: int, per_page: int, output_file: Path) -> None:
    """Run the end-to-end ETL for *pages* of articles."""

    # Extract – simple loop for clarity
    raw_pages: list[list[dict[str, Any]]] = []
    for page_number in range(1, pages + 1):
        raw_pages.append(fetch_page(page_number, api_base, per_page))

    # Transform
    df = to_dataframe(raw_pages)

    # Load
    save_csv(df, output_file)


if __name__ == "__main__":
    # Configuration – tweak to taste
    _api_base = "https://dev.to/api"
    _pages = 3  # Number of pages to fetch
    _per_page = 30  # Articles per page (max 30 per API docs)
    _output_file = Path("devto_articles.csv")

    # etl(api_base=_api_base, pages=_pages, per_page=_per_page, output_file=_output_file)

    etl.serve(
        name="first-deployment",
        cron="* * * * *",
        parameters={
            "api_base": _api_base,
            "pages": _pages,
            "per_page": _per_page,
            "output_file": _output_file,
        },
    )
