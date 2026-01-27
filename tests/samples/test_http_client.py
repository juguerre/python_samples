"""Tests for the http_client module."""

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from samples.http_client import (
    BaseHttpClient,
    GitHubClient,
    GitHubRepo,
    GitHubUser,
    HttpApiError,
    HttpRateLimitError,
    HttpValidationError,
    wait_for_rate_limit,
)


# Fixtures
@pytest.fixture
def mock_github_user_data() -> dict[str, Any]:
    """Return sample GitHub user data."""
    return {
        "login": "testuser",
        "id": 12345678,
        "avatar_url": "https://avatars.githubusercontent.com/u/12345678?v=4",
        "html_url": "https://github.com/testuser",
        "name": "Test User",
        "company": "Test Company",
        "blog": "https://testuser.dev",
        "location": "Test Location",
        "email": "test@example.com",
        "bio": "Test Bio",
        "public_repos": 42,
        "followers": 100,
        "following": 50,
        "created_at": "2011-01-25T18:44:36Z",
    }


@pytest.fixture
def mock_github_repo_data() -> list[dict[str, Any]]:
    """Return sample GitHub repository data."""
    return [
        {
            "id": 1,
            "name": "repo1",
            "full_name": "testuser/repo1",
            "html_url": "https://github.com/testuser/repo1",
            "description": "Test Repo 1",
            "stargazers_count": 10,
            "forks_count": 5,
            "language": "Python",
        }
    ]


@pytest.fixture
def github_client() -> GitHubClient:
    """Create a GitHubClient instance for testing."""
    return GitHubClient(token="test_token")


# Tests for GitHubUser model
def test_github_user_creation(mock_github_user_data: dict[str, Any]) -> None:
    """Test GitHubUser model creation and validation."""
    user = GitHubUser(**mock_github_user_data)

    # Test some fields
    assert user.login == "testuser"
    assert user.id == 12345678
    assert user.name == "Test User"
    assert user.public_repos == 42
    assert user.created_at == datetime(2011, 1, 25, 18, 44, 36, tzinfo=timezone.utc)


def test_github_user_optional_fields() -> None:
    """Test GitHubUser with optional fields as None."""
    user_data = {
        "login": "testuser",
        "id": 12345678,
        "avatar_url": "https://avatars.githubusercontent.com/u/12345678?v=4",
        "html_url": "https://github.com/testuser",
        "public_repos": 0,
        "followers": 0,
        "following": 0,
        "created_at": "2011-01-25T18:44:36Z",
    }
    user = GitHubUser(**user_data)

    assert user.name is None
    assert user.company is None
    assert user.email is None


# Tests for BaseHttpClient
class TestBaseHttpClient:
    """Tests for BaseHttpClient class."""

    @pytest.fixture
    def base_client(self) -> BaseHttpClient:
        """Create a BaseHttpClient instance for testing."""
        # Create a test instance with mock values
        client = BaseHttpClient(
            base_url="https://api.example.com",
            headers={"Test-Header": "test-value"},
            timeout=10,
            follow_redirects=True,
        )

        # Pre-initialize the clients with mocks to avoid real instantiation
        client._client = MagicMock(spec=httpx.Client)
        client._client_async = MagicMock(spec=httpx.AsyncClient)

        return client

    def test_get_success(
        self, base_client: BaseHttpClient, mock_github_user_data: dict[str, Any]
    ) -> None:
        """Test successful GET request with model validation."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_user_data
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        # Patch the client's get method
        base_client.client.get.return_value = mock_response

        # Call the method under test
        result = base_client._get(url="/test", model_class=GitHubUser)

        # Assertions
        assert isinstance(result, GitHubUser)
        assert result.login == "testuser"
        base_client.client.get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

    def test_get_http_error(self, base_client: BaseHttpClient) -> None:
        """Test GET request that raises an HTTP error."""
        # Setup mock to raise HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )
        base_client.client.get.return_value = mock_response

        # Test that the custom HttpApiError is raised
        with pytest.raises(HttpApiError) as excinfo:
            base_client._get(url="not-found", model_class=GitHubUser)

        assert excinfo.value.status_code == 404

    def test_get_validation_error(self, base_client: BaseHttpClient) -> None:
        """Test GET request with invalid response data."""
        # Setup mock with invalid data (missing required fields)
        mock_response = MagicMock()
        mock_response.json.return_value = {"invalid": "data"}
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        base_client.client.get.return_value = mock_response

        # Test that custom HttpValidationError is raised
        with pytest.raises(HttpValidationError):
            base_client._get(url="/invalid", model_class=GitHubUser)

    def test_get_rate_limit_error_retry_after(self, base_client: BaseHttpClient) -> None:
        """Test 429 error with Retry-After header."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "0.1"}
        mock_response.text = "Too Many Requests"
        base_client.client.get.return_value = mock_response

        with pytest.raises(HttpRateLimitError) as excinfo:
            base_client._get(url="/rate-limit", model_class=GitHubUser)

        assert excinfo.value.status_code == 429
        assert excinfo.value.retry_after == 0.1

    def test_get_rate_limit_error_github_reset(self, base_client: BaseHttpClient) -> None:
        """Test 429 error with GitHub X-RateLimit-Reset header."""
        # Reset time 1 second in the future
        future_reset = datetime.now().timestamp() + 1
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"X-RateLimit-Reset": str(future_reset)}
        mock_response.text = "Rate limit exceeded"
        base_client.client.get.return_value = mock_response

        with pytest.raises(HttpRateLimitError) as excinfo:
            base_client._get(url="/rate-limit", model_class=GitHubUser)

        assert excinfo.value.status_code == 429
        # Should be roughly 1 second
        assert 0 <= excinfo.value.retry_after <= 1.1

    def test_wait_for_rate_limit_logic(self) -> None:
        """Test the custom wait strategy logic directly."""
        mock_fallback = MagicMock()
        mock_fallback.return_value = 1.0
        strategy = wait_for_rate_limit(fallback=mock_fallback)

        # Case 1: Failed with HttpRateLimitError and retry_after
        mock_retry_state = MagicMock()
        error = HttpRateLimitError("Limit", 429, "body", retry_after=5.0)
        mock_retry_state.outcome.failed = True
        mock_retry_state.outcome.exception.return_value = error

        wait_time = strategy(mock_retry_state)
        assert wait_time == 5.5  # 5.0 + 0.5 jitter

        # Case 2: Failed with other error, should use fallback
        mock_retry_state.outcome.exception.return_value = ValueError("Other")
        wait_time = strategy(mock_retry_state)
        assert wait_time == 1.0
        mock_fallback.assert_called_once()


# Tests for GitHubClient
class TestGitHubClient:
    """Tests for GitHubClient class."""

    @pytest.fixture
    def mock_github_client(self) -> GitHubClient:
        """Create a GitHubClient with mocked HTTP client."""
        client = GitHubClient(token="test_token")
        client._client = MagicMock(spec=httpx.Client)
        # Mock headers since property access might return the mock
        client._client.headers = {"Authorization": "token test_token"}
        return client

    def test_init_with_token(self) -> None:
        """Test GitHubClient initialization with a token."""
        client = GitHubClient(token="test_token")
        assert client.client.headers["Authorization"] == "token test_token"

    def test_init_without_token(self) -> None:
        """Test GitHubClient initialization without a token."""
        client = GitHubClient(token="")
        env_token = os.getenv("GITHUB_TOKEN")
        assert (
            "Authorization" not in client.client.headers
            or client.client.headers["Authorization"] == "token " + env_token
        )

    def test_get_user_success(
        self, mock_github_client: GitHubClient, mock_github_user_data: dict[str, Any]
    ) -> None:
        """Test successful get_user call."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_user_data
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_github_client.client.get.return_value = mock_response

        # Call the method under test
        user = mock_github_client.get_user("testuser")

        # Assertions
        assert isinstance(user, GitHubUser)
        assert user.login == "testuser"
        mock_github_client.client.get.assert_called_once()

    def test_get_repos_success(
        self,
        mock_github_client: GitHubClient,
        mock_github_repo_data: list[dict[str, Any]],
    ) -> None:
        """Test successful get_repos call."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_repo_data
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_github_client.client.get.return_value = mock_response

        # Call the method under test
        repos = mock_github_client.get_repos("testuser")

        # Assertions
        assert isinstance(repos, list)
        assert len(repos) == 1
        assert isinstance(repos[0], GitHubRepo)
        assert repos[0].name == "repo1"

    @pytest.mark.anyio
    async def test_async_get_user_success(
        self, mock_github_user_data: dict[str, Any]
    ) -> None:
        """Test successful async get_user call."""
        client = GitHubClient(token="test_token")
        client._client_async = MagicMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_user_data
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        # In httpx 0.28+, get returns a coroutine
        client.client_async.get.return_value = mock_response

        user = await client.get_user_async("testuser")
        assert user.login == "testuser"
