"""GraphQL client module for strongly-typed operations using gql and Pydantic.

This module provides a base client and response classes to handle GraphQL
operations with full validation and type safety.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from gql import Client, gql
from gql.dsl import DSLSchema
from gql.transport.httpx import HTTPXTransport
from graphql import GraphQLError
from loguru import logger
from pydantic import BaseModel, ValidationError

# loguru logger
logger = logger.bind(name="graphql_client")

# Type variables for Pydantic models
T = TypeVar("T", bound=BaseModel)
DataT = TypeVar("DataT")


class BaseGraphQLResponse[DataT](BaseModel):
    """Base class for GraphQL responses containing data and optional errors.

    :ivar data: The validated data from the GraphQL operation.
    :ivar errors: A list of error details from the GraphQL server.
    """

    data: DataT | None = None
    errors: list[dict[str, Any]] | None = None

    @property
    def has_errors(self) -> bool:
        """Check if the response contains any errors.

        :return: True if the response contains errors, False otherwise.
        :rtype: bool
        """
        return self.errors is not None and len(self.errors) > 0


class MyGraphQLError(Exception):
    """Base exception for GraphQL client errors.

    :ivar message: message for this exception
    :ivar errors: A list of error details from the GraphQL server.
    :ivar original_exception: The original exception that triggered this error.
    """

    def __init__(
        self,
        message: str,
        errors: list[GraphQLError] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a string representation of the exception including error details."""
        parts = [self.message]
        if self.errors:
            error_summaries = []
            for e in self.errors:
                summary = f"'{e.message}'"
                if hasattr(e, "path") and e.path:
                    summary += f" at {'.'.join(map(str, str(e.path)))}"
                error_summaries.append(summary)
            parts.append(f"GraphQL Errors: [{', '.join(error_summaries)}]")

        if self.original_exception:
            parts.append(
                f"Original Exception: {type(self.original_exception).__name__}"
                f"({self.original_exception})"
            )

        return " | ".join(parts)

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the exception."""
        return (
            f"MyGraphQLError(message={self.message!r}, "
            f"errors={self.errors!r}, "
            f"original_exception={self.original_exception!r})"
        )


@runtime_checkable
class IGraphQLClient(Protocol):
    """Protocol for the GraphQL client to enable dependency injection and mocking."""

    def execute(
        self,
        query: str,
        response_model: type[T],
        variables: Any = None,
        operation_name: str | None = None,
    ) -> T:
        """Execute a GraphQL operation.

        :param query: The GraphQL query or mutation string.
        :param response_model: The Pydantic model class to validate the response against.
        :param variables: Optional variables for the GraphQL operation.
        :param operation_name: Optional name of the operation to execute.
        :return: An instance of the response_model containing the validated data.
        :rtype: T
        """
        ...

    async def execute_async(
        self,
        query: str,
        response_model: type[T],
        variables: Any = None,
        operation_name: str | None = None,
    ) -> T:
        """Asynchronously execute a GraphQL operation.

        :param query: The GraphQL query or mutation string.
        :param response_model: The Pydantic model class to validate the response against.
        :param variables: Optional variables for the GraphQL operation.
        :param operation_name: Optional name of the operation to execute.
        :return: An instance of the response_model containing the validated data.
        :rtype: T
        """
        ...


class BaseGraphQLClient(IGraphQLClient):
    """A strongly-typed GraphQL client leveraging gql and Pydantic.

    This client provides a convenient way to execute GraphQL queries and mutations
    while ensuring that both inputs and outputs are validated against Pydantic models.
    It supports both raw GraphQL strings and ``gql``'s Domain Specific Language (DSL).

    :param url: The GraphQL endpoint URL.
    :param headers: Optional dictionary of HTTP headers.
    :param verify: Whether to verify SSL certificates. Defaults to True.
    :param timeout: Request timeout in seconds. Defaults to 30.
    :param fetch_schema: Whether to fetch the schema from the server on initialization
        (required for DSL usage).
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        verify: bool = True,
        timeout: int = 30,
        fetch_schema: bool = False,
    ) -> None:
        self.url = url
        self.headers = headers
        self.transport = HTTPXTransport(
            url=url,
            headers=headers,
            verify=verify,
            timeout=timeout,
        )
        self.gql_client = Client(
            transport=self.transport,
            fetch_schema_from_transport=fetch_schema,
        )
        self._dsl_schema: DSLSchema | None = None

    def get_dsl_schema(self) -> DSLSchema:
        """Helper to get the DSL schema for building queries programmatically.

        :return: The DSL schema object.
        :rtype: DSLSchema
        :raises MyGraphQLError: If the client schema is not loaded.
        """
        if not self.gql_client.schema:
            raise MyGraphQLError("Client schema is not loaded. Initialize with fetch_schema=True.")
        if not self._dsl_schema:
            self._dsl_schema = DSLSchema(self.gql_client.schema)
        return self._dsl_schema

    @staticmethod
    def _prepare_variables(variables: dict[str, Any] | BaseModel | None) -> dict[str, Any] | None:
        """Convert variables to a dictionary, supporting Pydantic models.

        :param variables: Variables for the GraphQL operation.
        :type variables: dict[str, Any] | BaseModel | None
        :return: The variables as a dictionary.
        :rtype: dict[str, Any] | None
        :raises TypeError: If the variables are not a dict or a Pydantic BaseModel.
        """
        if variables is None:
            return None
        if isinstance(variables, BaseModel):
            return variables.model_dump(exclude_none=True)
        if isinstance(variables, dict):
            return variables
        raise TypeError("Variables must be a dict or a Pydantic BaseModel")

    def execute(
        self,
        query: str,
        response_model: type[T],
        variables: dict[str, Any] | BaseModel | None = None,
        operation_name: str | None = None,
    ) -> T:
        """Execute a GraphQL operation and parse the response into a Pydantic model.

        :param query: The GraphQL query or mutation string.
        :param response_model: The Pydantic model class to validate the response against.
        :param variables: Optional variables for the GraphQL operation.
        :param operation_name: Optional name of the operation to execute.
        :return: An instance of the response_model containing the validated data.
        :rtype: T
        :raises MyGraphQLError: If the GraphQL server returns errors or if validation fails.
        """
        document = gql(query)
        variable_values = self._prepare_variables(variables)

        try:
            logger.debug(f"Executing GraphQL operation: {operation_name or 'unnamed'}")

            # Using get_execution_result=True to get both data and errors
            execution_result = self.gql_client.execute(
                document,
                variable_values=variable_values,
                operation_name=operation_name,
                get_execution_result=True,
            )

            response_data = {
                "data": execution_result.data,
                "errors": execution_result.errors,
            }

            # If the response model is a BaseGraphQLResponse, we validate the whole payload
            if issubclass(response_model, BaseGraphQLResponse):
                return response_model.model_validate(response_data)

            # Legacy/Simple support: if not using BaseGraphQLResponse,
            # we check for errors and raise if any, then validate just the data part.
            if execution_result.errors:
                raise MyGraphQLError(
                    f"GraphQL operation failed: First error: {execution_result.errors[0].message}",
                    errors=execution_result.errors,
                )

            return response_model.model_validate(execution_result.data)

        except ValidationError as e:
            logger.error(f"Pydantic validation error for {response_model.__name__}: {e}")
            raise MyGraphQLError(
                f"Response validation failed for {response_model.__name__}",
                original_exception=e,
            ) from e
        except MyGraphQLError:
            raise
        except Exception as e:
            msg = str(e)
            logger.error(f"GraphQL execution error: {msg}")
            raise MyGraphQLError(
                f"GraphQL operation failed: {msg}",
                original_exception=e,
            ) from e

    async def execute_async(
        self,
        query: str,
        response_model: type[T],
        variables: Any = None,
        operation_name: str | None = None,
    ) -> T:
        """Asynchronous version of execute.

        :param query: The GraphQL query or mutation string.
        :param response_model: The Pydantic model class to validate the response against.
        :param variables: Optional variables for the GraphQL operation.
        :param operation_name: Optional name of the operation to execute.
        :return: An instance of the response_model containing the validated data.
        :rtype: T
        :raises MyGraphQLError: If the GraphQL server returns errors or if validation fails.
        """
        document = gql(query)
        variable_values = self._prepare_variables(variables)

        try:
            async with self.gql_client as session:
                execution_result = await session.execute(
                    document,
                    variable_values=variable_values,
                    operation_name=operation_name,
                    get_execution_result=True,
                )

                response_data = {
                    "data": execution_result.data,
                    "errors": execution_result.errors,
                }

                if issubclass(response_model, BaseGraphQLResponse):
                    return response_model.model_validate(response_data)

                if execution_result.errors:
                    raise MyGraphQLError(
                        f"GraphQL operation '{operation_name}' failed:"
                        f" errors: {len(execution_result.errors)} ",
                        errors=execution_result.errors,
                    )

                return response_model.model_validate(execution_result.data)

        except ValidationError as e:
            raise MyGraphQLError(
                f"Response validation failed for {response_model.__name__}",
                original_exception=e,
            ) from e
        except MyGraphQLError:
            raise
        except Exception as e:
            raise MyGraphQLError(
                f"GraphQL operation failed: {str(e)}",
                original_exception=e,
            ) from e


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Define models using the new base response class
    class UserProfile(BaseModel):
        id: str
        username: str
        email: str | None = None

    class UserData(BaseModel):
        # Key matches the GraphQL field name
        user: UserProfile

    class UserResponse(BaseGraphQLResponse[UserData]):
        """A typed response that includes both data and potential errors.

        :ivar data: The user data if the request was successful.
        :ivar errors: Any errors reported by the GraphQL server.
        """

    # 2. Define a specialized client
    class MySimpleClient(BaseGraphQLClient):
        """Example specialized client demonstrating how to extend the base client."""

        def get_user(self, user_id: str) -> UserResponse:
            """Fetch a user by their ID.

            :param user_id: The unique identifier for the user.
            :return: A UserResponse object containing the user data and any errors.
            :rtype: UserResponse
            """
            query = """
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                    username
                    email
                }
            }
            """
            return self.execute(query, response_model=UserResponse, variables={"id": user_id})

        async def get_user_async(self, user_id: str) -> UserResponse:
            """Asynchronously fetch a user by their ID.

            :param user_id: The unique identifier for the user.
            :return: A UserResponse object containing the user data and any errors.
            :rtype: UserResponse
            """
            query = """
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                    username
                    email
                }
            }
            """
            return await self.execute_async(
                query, response_model=UserResponse, variables={"id": user_id}
            )

    # 3. Demo usage (conceptual, as we need a real endpoint)
    # client = MySimpleClient(url="https://api.example.com/graphql")
    # try:
    #     response = client.get_user("123")
    #     if response.has_errors:
    #         print(f"Errors encountered: {response.errors}")
    #     if response.data:
    #         print(f"User: {response.data.user.username}")
    # except MyGraphQLError as e:
    #     print(f"Request failed: {e}")

    logger.info("Success: GraphQL client with BaseGraphQLResponse support and sample client ready.")
