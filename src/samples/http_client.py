import asyncio
import os
from datetime import datetime
from functools import wraps
from io import StringIO
from typing import Any, Callable, ClassVar, TypeVar

import httpx
import loguru
from dotenv import load_dotenv
from httpx import Response
from pydantic import BaseModel, HttpUrl, ValidationError, field_validator
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# ensure .env is load as env vars
load_dotenv()

# TypeVar for models
T = TypeVar("T", bound=BaseModel)

logger = loguru.logger


def model_as_str(
    model: BaseModel | list[BaseModel] | dict[str, BaseModel], style: str = "simple"
) -> str:
    """Convert a model class to a pretty string

    :param model: The model instance, list of model instances, or dict of model instances.
    :param style: The style to use for the string representation. simple, yaml
    :return: A pretty string representation of the model class.
    """
    if model is None:
        return "None"
    if isinstance(model, list) and model:
        return f"List[{model[0].__class__.__name__}]"
    if isinstance(model, dict):
        s_io: StringIO = StringIO()
        s_io.write("\nDict(\n")
        for key, value in model.items():
            s_io.write(f"  {key}: {value}\n")
        s_io.write(")")
        return s_io.getvalue()
    # for simple instance models get the class name and pydantic field values
    if style == "simple":
        s_io: StringIO = StringIO()
        s_io.write(model.__class__.__name__ + "(")
        for field in model.__class__.model_fields:
            # field name and value
            s_io.write(f"{field}: {model.model_dump().get(field)}, ")
        s_io.write(")")
        return s_io.getvalue()
    elif style == "yaml":
        s: str = model.model_dump_json(indent=2)
        # remove quotes and commas from json
        s = s.replace('"', "").replace(",", "").replace("{", "").replace("}", "")
        s = "\n" + model.__class__.__name__ + "(" + s + ")"
        return s

    else:
        raise ValueError(f"Unknown style: {style}")


def log_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log httpclient requests"""

    @wraps(func)
    def request_wrapper(*args, **kwargs) -> Callable[..., Any]:
        method = func.__name__.replace("_", " ").upper()
        params = kwargs.get("params")
        data = kwargs.get("data")
        url = kwargs.get("url")
        model_class = kwargs.get("model_class").__name__
        logger.debug(
            f"Request: method={method} | url={url} | model_class={model_class} "
            f"| params={params} | data={data}"
        )
        return func(*args, **kwargs)

    return request_wrapper


class HttpClientError(Exception):
    """Base exception for HttpClient"""
    pass


class HttpApiError(HttpClientError):
    """Exception for API errors (non-2xx responses)"""

    def __init__(self, message: str, status_code: int, response_text: str) -> None:
        """Initialize the API error.

        :param message: The error message.
        :param status_code: The HTTP status code returned by the API.
        :param response_text: The raw response text from the API.
        """
        super().__init__(f"{message} (Status: {status_code})")
        self.status_code = status_code
        self.response_text = response_text


class HttpValidationError(HttpClientError):
    """Exception for validation errors"""

    def __init__(
        self, message: str, validation_error: ValidationError, data: Any
    ) -> None:
        """Initialize the validation error.

        :param message: The error message.
        :param validation_error: The underlying Pydantic ValidationError.
        :param data: The raw data that failed validation.
        """
        super().__init__(message)
        self.validation_error = validation_error
        self.data = data


class GitHubUser(BaseModel):
    """User information from GitHub API"""

    login: str
    id: int
    avatar_url: HttpUrl
    html_url: HttpUrl
    name: str | None = None
    company: str | None = None
    blog: str | None = None
    location: str | None = None
    email: str | None = None
    bio: str | None = None
    public_repos: int
    followers: int
    following: int
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> Any:
        """Parse the GitHub datetime string into a datetime object.

        :param v: The value to parse (usually an ISO format string).
        :return: A datetime object.
        """
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class GitHubRepo(BaseModel):
    """Repository information from GitHub API"""

    id: int
    name: str
    full_name: str
    html_url: HttpUrl
    description: str | None = None
    stargazers_count: int
    forks_count: int
    language: str | None = None


class BaseHttpClient:
    """A base HTTP client with retry logic and Pydantic validation."""

    @staticmethod
    def _is_transient_error(exception: BaseException) -> bool:
        """Check if an error is transient and should be retried."""
        if isinstance(exception, HttpValidationError):
            return False
        if isinstance(exception, HttpApiError):
            # Retry on 5xx (Server Errors) or 429 (Too Many Requests)
            return exception.status_code >= 500 or exception.status_code == 429
        # Retry on httpx connection/timeout errors
        return isinstance(exception, (httpx.RequestError, httpx.HTTPStatusError))

    RETRY_CONFIG: ClassVar[dict[str, Any]] = {
        "stop": stop_after_attempt(3),
        "wait": wait_exponential(multiplier=1, min=2, max=10),
        "retry": retry_if_exception(_is_transient_error),
        "reraise": True,
    }

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 10,
        follow_redirects: bool = True,
    ) -> None:
        """Initialize the base HTTP client.

        :param base_url: The base URL for all API requests.
        :param headers: Optional default headers to include in every request.
        :param timeout: Request timeout in seconds.
        :param follow_redirects: Whether to follow HTTP redirects.
        """
        self._base_url = base_url
        self._headers = headers or {}
        self._timeout = timeout
        self._follow_redirects = follow_redirects

        self._client: httpx.Client | None = None
        self._client_async: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or initialize the synchronous HTTP client.

        :return: The synchronous HTTP client instance.
        """
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=self._follow_redirects,
            )
        return self._client

    @property
    def client_async(self) -> httpx.AsyncClient:
        """Get or initialize the asynchronous HTTP client.

        :return: The asynchronous HTTP client instance.
        """
        if self._client_async is None:
            self._client_async = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=self._follow_redirects,
            )
        return self._client_async

    @staticmethod
    def _handle_response_error(response: Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error {e.response.status_code} for {e.request.url}")
            raise HttpApiError(
                message=str(e),
                status_code=e.response.status_code,
                response_text=e.response.text,
            ) from e

    @retry(**RETRY_CONFIG)
    @log_request
    def _get(
        self, *, url: str, model_class: type[T], params: dict[str, Any] | None = None
    ) -> T:
        response = self.client.get(url, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    async def _get_async(
        self, *, url: str, model_class: type[T], params: dict[str, Any] | None = None
    ) -> T:
        response = await self.client_async.get(url, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    def _post(self, *, url: str, data: dict[str, Any], model_class: type[T]) -> T:
        response = self.client.post(url, json=data)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    async def _post_async(
        self, *, url: str, data: dict[str, Any], model_class: type[T]
    ) -> T:
        response = await self.client_async.post(url, json=data)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    def _put(
        self,
        *,
        url: str,
        data: dict[str, Any],
        model_class: type[T],
        params: dict[str, Any] | None = None,
    ) -> T:
        response = self.client.put(url, json=data, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    async def _put_async(
        self,
        *,
        url: str,
        data: dict[str, Any],
        model_class: type[T],
        params: dict[str, Any] | None = None,
    ) -> T:
        response = await self.client_async.put(url, json=data, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    def _patch(
        self,
        *,
        url: str,
        data: dict[str, Any],
        model_class: type[T],
        params: dict[str, Any] | None = None,
    ) -> T:
        response = self.client.patch(url, json=data, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    async def _patch_async(
        self,
        *,
        url: str,
        data: dict[str, Any],
        model_class: type[T],
        params: dict[str, Any] | None = None,
    ) -> T:
        response = await self.client_async.patch(url, json=data, params=params)
        self._handle_response_error(response)
        return self._validate_response(model_class, response.json(), url)

    @retry(**RETRY_CONFIG)
    @log_request
    def _delete(
        self,
        *,
        url: str,
        model_class: type[T] | None = None,
        params: dict[str, Any] | None = None,
    ) -> T | None:
        response = self.client.delete(url, params=params)
        self._handle_response_error(response)
        if model_class:
            return self._validate_response(model_class, response.json(), url)
        return None

    @retry(**RETRY_CONFIG)
    @log_request
    async def _delete_async(
        self,
        *,
        url: str,
        model_class: type[T] | None = None,
        params: dict[str, Any] | None = None,
    ) -> T | None:
        response = await self.client_async.delete(url, params=params)
        self._handle_response_error(response)
        if model_class:
            return self._validate_response(model_class, response.json(), url)
        return None

    @retry(**RETRY_CONFIG)
    @log_request
    def _get_list(
        self, *, url: str, model_class: type[T], params: dict[str, Any] | None = None
    ) -> list[T]:
        response = self.client.get(url, params=params)
        self._handle_response_error(response)
        return [
            self._validate_response(model_class, item, url, i)
            for i, item in enumerate(response.json())
        ]

    @retry(**RETRY_CONFIG)
    @log_request
    async def _get_list_async(
        self, *, url: str, model_class: type[T], params: dict[str, Any] | None = None
    ) -> list[T]:
        response = await self.client_async.get(url, params=params)
        self._handle_response_error(response)
        return [
            self._validate_response(model_class, item, url, i)
            for i, item in enumerate(response.json())
        ]

    @staticmethod
    def _validate_response(
        model_class: type[T],
        data: dict[str, Any],
        url: str,
        index: int | None = None,
    ) -> T:
        try:
            return model_class.model_validate(data)
        except ValidationError as ve:
            idx_msg = f" (at index {index})" if index is not None else ""
            error_msg = (
                f"Validation failed for {model_class.__name__} from {url}{idx_msg}"
            )
            logger.error(f"{error_msg}. Data: {data}")
            raise HttpValidationError(error_msg, ve, data) from ve

    def close(self) -> None:
        """Close the synchronous HTTP client and release resources."""
        if self._client:
            self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client and release resources."""
        if self._client_async:
            await self._client_async.aclose()

    def __enter__(self) -> "BaseHttpClient":
        """Enter the synchronous context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the synchronous context manager and close the client."""
        self.close()

    async def __aenter__(self) -> "BaseHttpClient":
        """Enter the asynchronous context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the asynchronous context manager and close the client."""
        await self.aclose()


class GitHubClient(BaseHttpClient):
    BASE_URL: ClassVar[str] = "https://api.github.com"
    BASE_HEADERS: ClassVar[dict] = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    def __init__(self, token: str | None = None, timeout: int = 10) -> None:
        """Initialize the GitHub API client.

        :param token: Optional GitHub personal access token. If not provided, it looks for
            GITHUB_TOKEN env var.
        :param timeout: Request timeout in seconds.
        """
        # Favor token parameter, then environment variable
        auth_token = token or os.getenv("GITHUB_TOKEN")

        headers = self.BASE_HEADERS.copy()
        if auth_token:
            headers["Authorization"] = f"token {auth_token}"

        super().__init__(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=timeout,
        )

    def get_user(self, username: str) -> GitHubUser:
        """Get GitHub user information by username.

        :param username: The GitHub username to retrieve.
        :return: A GitHubUser model containing the user's data.
        """
        # noinspection PyArgumentList
        return self._get(url=f"/users/{username}", model_class=GitHubUser)

    async def get_user_async(self, username: str) -> GitHubUser:
        """Asynchronously get GitHub user information by username.

        :param username: The GitHub username to retrieve.
        :return: A GitHubUser model containing the user's data.
        """
        # noinspection PyArgumentList
        return await self._get_async(url=f"/users/{username}", model_class=GitHubUser)

    def get_repos(self, username: str) -> list[GitHubRepo]:
        """Get the list of public repositories for a GitHub user.

        :param username: The GitHub username whose repositories to retrieve.
        :return: A list of GitHubRepo models.
        """
        # noinspection PyArgumentList
        return self._get_list(url=f"/users/{username}/repos", model_class=GitHubRepo)

    async def get_repos_async(self, username: str) -> list[GitHubRepo]:
        """Asynchronously get the list of public repositories for a GitHub user.

        :param username: The GitHub username whose repositories to retrieve.
        :return: A list of GitHubRepo models.
        """
        # noinspection PyArgumentList
        return await self._get_list_async(
            url=f"/users/{username}/repos", model_class=GitHubRepo
        )


if __name__ == "__main__":
    # Example usage
    async def main() -> None:
        async with GitHubClient() as my_client:
            try:
                # Sync call
                user = await asyncio.to_thread(my_client.get_user, "juguerre")
                logger.info(f"Sync user: {model_as_str(user, style='yaml')}")

                # Async call
                user_async = await my_client.get_user_async("juguerre")
                logger.info(f"Async user: {model_as_str(user_async.model_dump(), style='simple')}")

                # List repos
                repos = await my_client.get_repos_async("juguerre")
                logger.info(model_as_str(repos, style="yaml"))
                logger.info(f"Found {len(repos)} repositories")
                if repos:
                    logger.info(f"First repo: {repos[0].full_name}")

            except HttpClientError as e:
                logger.error(f"Client error: {e}")

    asyncio.run(main())
