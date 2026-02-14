# Python Architecture & Utility Samples

A comprehensive collection of Python samples demonstrating advanced architectural patterns, robust utility implementations, and modern Python best practices.

## 🚀 Overview

This repository serves as a showcase for various software engineering patterns implemented in Python, ranging from complex task orchestration to elegant data pipelines and robust network clients.

## 📦 Key Modules

### 🛠️ Task Orchestration & DAGs

* **[TaskDAGExecutor](./src/samples/dag_runner.py)**: A high-performance executor for Directed Acyclic Graphs (DAGs). It uses `networkx` for dependency management and `ThreadPoolExecutor` for concurrent task execution.
* **[TaskFlowParser](./src/samples/task_flow_parser.py)**: An elegant parser for Airflow-like dependency expressions (e.g., `[task1, task2] >> task3`), allowing for intuitive graph construction.

### 🔗 Fluent Pipelines

* **[FluentPipe](./src/samples/fluent_pipeline.py)**: Provides a chainable, fluent interface for building data processing pipelines. It leverages `toolz.curry` for elegant function composition.
* **[FluentPipeProtocol](./src/samples/protocol_stubs.py)**: Auto-generated protocol that enables full IDE type-hinting and auto-completion for dynamic pipelines.
* **[Stub Generation Script](./scripts/gen_stubs.py)**: A maintenance utility that inspects code to regenerate the `FluentPipeProtocol`.

### 🌐 Networking & APIs

* **[GitHubClient](./src/samples/http_client.py)**: A production-ready HTTP client implementation featuring:
  * Both **Synchronous** and **Asynchronous** support.
  * Robust **Retry logic** with exponential backoff (via `tenacity`).
  * Strict **Type validation** using `Pydantic`.
  * Automated logging and error handling.

### 📝 Advanced Templating

* **[String Templating](./src/samples/string_templating.py)**: A secure, sandboxed `Jinja2` environment specialized for date and string manipulations. Includes a rich set of custom filters for complex date arithmetic and formatting.

### 💾 Data Management

* **[ContentSnapshotStore](./src/samples/content_snapshot.py)**: A utility for managing versioned content snapshots, providing simple load/store operations with automatic cleanup and history limits.

## 🛠️ Getting Started

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

```bash
# Install dependencies
uv sync
```

### Running Samples

Most modules can be run directly as scripts to see their functionality in action:

```bash
# Run the Fluent Pipeline demo
uv run src/samples/fluent_pipeline.py

# Run the GitHub Client demo
uv run src/samples/http_client.py

# Regenerate Protocol stubs
uv run scripts/gen_stubs.py
```

### Running Tests

```bash
uv run pytest
```

## 🧰 Tech Stack

* **[NetworkX](https://networkx.org/)**: Graph algorithms and dependency management.
* **[Pydantic](https://docs.pydantic.dev/)**: Data validation and settings management.
* **[HTTPX](https://www.python-httpx.org/)**: Modern HTTP client for Python.
* **[Jinja2](https://jinja.palletsprojects.com/)**: Powerful templating engine.
* **[Toolz](https://toolz.readthedocs.io/)**: Functional programming utilities.
* **[Tenacity](https://tenacity.readthedocs.io/)**: Retrying library for Python.
* **[Loguru](https://github.com/Delgan/loguru)**: Pleasant logging.
