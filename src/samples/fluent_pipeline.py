"""Fluent Pipeline Module.

A powerful data processing pipeline that provides a fluent, chainable interface
for composing curried functions. This module enables elegant data transformation
workflows with proper type safety and error handling.

Example::

    @pipeable
    def filter_data(data: list, *, threshold: int) -> list:
        return [x for x in data if x > threshold]

    @pipeable
    def transform_data(data: list, *, multiplier: int) -> list:
        return [x * multiplier for x in data]

    result = (default_registry.create_fluent_pipe([1, 2, 3, 4, 5])
              .filter_data(threshold=3)
              .transform_data(multiplier=2)
              .execute())
    # result: [8, 10]

Example with IDE Support::

    from typing import cast
    from .protocol_stubs import FluentPipeProtocol

    # Explicitly cast to the Protocol for full auto-completion/type checking
    pipe = cast(FluentPipeProtocol, default_registry.create_fluent_pipe(data))
    result = pipe.filter_data(threshold=3).execute()

The module uses toolz.curry for function currying and provides a clean
fluent interface for building data processing pipelines.

**Classes:**

* :class:`PipeStep`: Intermediate object for pipeline steps
* :class:`FluentPipe`: Main pipeline builder class
* :class:`PipelineFunctionsRegistry`: Registry for managing pipeable functions

**Functions:**

* :func:`pipeable`: Decorator for registering pipeline functions
* :func:`sample_filter1`: Sample filter function for testing
* :func:`sample_filter2`: Sample identity function for testing
* :func:`sample_add_to_values`: Sample function for adding values to dict entries
* :func:`fluent_pipe`: Demo function showing pipeline usage
"""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Generic, TypeVar

import toolz
from icecream import ic
from toolz import curry

from samples.protocol_stubs import FluentPipeProtocol

# Type variable for the Protocol interface
P = TypeVar("P")


# noinspection PyProtectedMember
class PipeStep(Generic[P]):
    """Intermediate object that enforces function calls in the pipeline.

    This class acts as a bridge between :class:`FluentPipe` and individual
    pipeline functions. It ensures that functions are called with proper
    keyword arguments and maintains the fluent interface.

    :ivar pipe: The parent FluentPipe instance.
    :ivar func: The curried function to be executed.

    :raises RuntimeError: If positional arguments are passed instead of keywords.
    :raises AttributeError: If trying to access attributes when previous function
        wasn't properly called.
    """

    def __init__(self, pipe: "FluentPipe[P]", func: Callable) -> None:
        """Initialize a PipeStep instance.

        :param pipe: The parent FluentPipe instance.
        :param func: The curried function to be executed in this step.
        """
        self.pipe = pipe
        self.func = func

    def __call__(self, *args, **kwargs) -> P:
        """Execute the pipeline step with provided keyword arguments.

        :param args: Positional arguments (not allowed, will raise RuntimeError).
        :param kwargs: Keyword arguments for the curried function.

        :return: The parent FluentPipe instance cast to the Protocol P.

        :raises RuntimeError: If positional arguments are provided.
        """
        if args:
            raise RuntimeError(
                "All parameters for curried functions must be passed as keywords:"
                f" found arg: '{args}' in func '{self.func.__name__}'"
            )
        self.pipe._add_step(self.func, kwargs)
        return self.pipe

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access with proper error messages.

        :param name: The attribute name being accessed.

        :return: The requested attribute if it exists.

        :raises AttributeError: If the previous function wasn't called properly
            or if the attribute doesn't exist.
        """
        if self.pipe._funcs:
            raise AttributeError(f"'{self.pipe._funcs[-1].__name__}' was not correctly called!")
        raise AttributeError(f"Attribute '{name}' not found")


class FluentPipe(Generic[P]):
    """A fluent data pipeline for composing curried functions.

    This class provides a clean, chainable interface for building data
    processing pipelines using curried functions. It maintains the state
    of the pipeline and ensures proper function execution order.

    The pipeline requires all functions to be properly called before
    execution, and enforces that curried functions have exactly one
    remaining parameter (the data to be processed).


    :ivar _data: The initial data to be processed.
    :ivar _funcs: List of functions in the pipeline.
    :ivar _kwargs: List of keyword arguments for each function.
    :ivar _funcs_namespace: Dictionary of available curried functions.

    :param data: The initial data to process through the pipeline.
    :param funcs_namespace: Dictionary mapping function names to curried functions.
    """

    def __init__(self, data: Any, funcs_namespace: dict[str, Callable]) -> None:
        """Initialize a FluentPipe instance.

        :param data: The initial data to process through the pipeline.
        :param funcs_namespace: Dictionary mapping function names to curried functions.
        """
        self._data = data
        self._funcs: list[Callable] = []
        self._kwargs: list[dict[str, Any]] = []
        self._funcs_namespace = funcs_namespace or {}

    def get_fluent_pipe(self) -> P:
        """Returns self as P protocol for type check convenience."""
        return self

    def __getattr__(self, name: str) -> PipeStep[P]:
        """Dynamically access pipeline functions from the namespace.

        This method enables the fluent interface by returning PipeStep
        objects for registered functions. It validates that the requested
        function exists and is callable.

        :param name: The name of the function to access.

        :return: A PipeStep instance for the requested function.

        :raises AttributeError: If the function is not found in the namespace
            or is not callable.
        """

        func = self._funcs_namespace.get(name)

        if func and callable(func):
            # self._funcs.append(func)
            # return self
            return PipeStep(self, func)
        else:
            raise AttributeError(f"'{name}' attribute not found in namespace or this class")

    def _add_step(self, func: Callable, kwargs: dict[str, Any]) -> None:
        """Add a function step to the pipeline.

        :param func: The curried function to add.
        :param kwargs: The keyword arguments for the function.
        """
        self._funcs.append(func)
        self._kwargs.append(kwargs)

    @staticmethod
    def _check_curried_func(f: Callable) -> None:
        """Validate that a curried function has exactly one remaining parameter.

        This ensures that the function is ready to receive the pipeline data
        as its final argument. The function should have all other parameters
        already provided through currying.

        :param f: The curried function to validate.

        :raises RuntimeError: If the function doesn't have exactly one remaining
            parameter.
        """
        if hasattr(f, "func") and hasattr(f, "args"):
            # It's a toolz.curry partial
            f: partial
            sig = inspect.signature(f.func)
            remaining_params = len(sig.parameters) - len(f.args) - len(f.keywords)

            if remaining_params != 1:
                raise RuntimeError(
                    f"Function '{f.__name__}' should have exactly 1 remaining parameter (data), "
                    f"but has {remaining_params}. Provide all other parameters."
                )

    def execute(self) -> Any:
        """Execute the pipeline and return the processed data.

        This method validates that all functions have been properly called,
        prepares the curried functions, and executes them in sequence using
        toolz.pipe.

        :return: The final processed data after all pipeline steps.

        :raises RuntimeError: If not all functions have been called with their
            required parameters.
        """
        if len(self._funcs) > len(self._kwargs):
            raise RuntimeError(
                "All FluentPipe functions should be called: FluentPipe(data).func1().func2(): "
                f"Previous func '{self._funcs[-1].__name__}' was not called"
            )
        c_funcs = []
        for func, kwargs in zip(self._funcs, self._kwargs, strict=False):
            f = func(**kwargs)
            self._check_curried_func(f)
            c_funcs.append(f)

        return toolz.pipe(self._data, *c_funcs)


class PipelineRegistry(Generic[P]):
    """A registry for managing and providing access to pipeable functions.

    This class encapsulates the function mapping and provides the
    decorator interface for registration.

    This class is implemented as a borg pattern to allow for a single
    instance of the registry to be used across the application.

    :ivar _funcs: Dictionary mapping function names to curried functions.
    :ivar _borg_state: Shared state across all instances of this class.
    """

    _borg_state: dict[str, Any]

    def __init__(self) -> None:
        """Initialize or retrieve the shared borg state."""
        # This allows the class to be used as a borg pattern
        if not hasattr(self, "_borg_state"):
            self._funcs: dict[str, Callable] = {}
            self._borg_state = self.__dict__
        else:
            self.__dict__ = self._borg_state

    def pipeable(self, func: Callable) -> Callable:
        """Decorator to register a function in this registry instance.

        :param func: The function to curry and register.
        :return: The curried version of the function.
        """
        curried = curry(func)
        self._funcs[func.__name__] = curried
        return curried

    @property
    def namespace(self) -> dict[str, Callable]:
        """Returns a copy of the registered functions mapping.

        :return: Dict mapping function names to curried functions.
        """
        return self._funcs.copy()

    def create_fluent_pipe(self, data: Any) -> P:
        """Factory method to create a FluentPipe using this registry.

        :param data: The initial data for the pipeline.
        :return: A new FluentPipe instance cast to the Protocol P.
        """
        return FluentPipe(data, self.namespace)


# Create a default registry instance
pipe_registry = PipelineRegistry[FluentPipeProtocol]()
# Re-export the decorator for convenience
pipeable = pipe_registry.pipeable


@pipeable
def sample_filter1(data: list, mul: int) -> Any:
    """Sample filter function that multiplies list elements.

    This is a demonstration function for the fluent pipeline.
    It creates a new list by repeating the original list 'mul' times.

    :param data: The input list to be processed.
    :param mul: The multiplier for list repetition.

    :return: The multiplied list.

    Example:
        >>> sample_filter1([1, 2], mul=3)
        [1, 2, 1, 2, 1, 2]
    """
    return data * mul


@pipeable
def sample_filter2(data: list) -> Any:
    """Sample identity function for testing pipelines.

    This function simply returns the input data unchanged.
    It's useful for testing pipeline structure and flow.

    :param data: The input data to pass through unchanged.

    :return: The same data that was passed in.

    Example:
        >>> sample_filter2([1, 2, 3])
        [1, 2, 3]
    """
    return data


@pipeable
def sample_add_to_values(data: list[dict], value: str) -> list[dict]:
    """Sample function that adds a suffix to all string values in dictionaries.

    This function iterates through a list of dictionaries and adds the
    provided value to all string values in each dictionary using toolz.valmap.

    :param data: List of dictionaries to process.
    :param value: The string value to append to all string values.

    :return: A new list of dictionaries with modified string values.

    Example:
        >>> sample_add_to_values([{'a': '1', 'b': '2'}], value='_test')
        [{'a': '1_test', 'b': '2_test'}]
    """
    data = [toolz.valmap(lambda x: x + value, d) for d in data]
    return data


def fluent_pipe() -> None:
    """Demonstration function showing FluentPipe usage.

    This function creates sample data and demonstrates how to build
    and execute a fluent pipeline using the registered pipeable functions.
    It serves as both a test and an example of proper usage.

    The pipeline:
    1. Starts with sample dictionary data
    2. Multiplies the list using sample_filter1
    3. Passes through sample_filter2 (identity)
    4. Adds suffix to values using sample_add_to_values
    5. Executes and prints the result using icecream

    :return: None (prints result using ic)
    """
    data = [
        {"key1": "aaaaa"},
        {"key1": "bbbbb"},
    ]

    pipe: FluentPipeProtocol = pipe_registry.create_fluent_pipe(data)
    res1 = pipe.sample_filter1(mul=5).sample_filter2().sample_add_to_values(value="_add!").execute()

    pipe: FluentPipeProtocol = FluentPipe[FluentPipeProtocol](
        data,
        pipe_registry.namespace,
    ).get_fluent_pipe()

    res2 = pipe.sample_add_to_values(value="_Hu!").sample_filter1(mul=2).sample_filter2().execute()

    ic(res1)
    ic(res2)


if __name__ == "__main__":
    fluent_pipe()
