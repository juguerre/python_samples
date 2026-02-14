from typing import Any

import pytest

from samples.fluent_pipeline import (
    FluentPipe,
    PipelineRegistry,
    pipeable,
    sample_add_to_values,
    sample_filter1,
    sample_filter2,
)


def test_basic_pipeline_execution() -> None:
    """Test a simple pipeline with multiple steps."""
    data = [{"a": "1"}, {"a": "2"}]
    # Using the default registry implicitly through sample functions
    pipe: FluentPipe[Any] = FluentPipe(
        data,
        {
            "sample_filter1": sample_filter1,
            "sample_filter2": sample_filter2,
            "sample_add_to_values": sample_add_to_values,
        },
    )

    result = (
        pipe.sample_filter1(mul=2).sample_filter2().sample_add_to_values(value="_test").execute()
    )

    assert len(result) == 4
    assert result[0] == {"a": "1_test"}
    assert result[1] == {"a": "2_test"}
    assert result[2] == {"a": "1_test"}
    assert result[3] == {"a": "2_test"}


def test_positional_arguments_fail() -> None:
    """Test that passing positional arguments to pipeline steps raises RuntimeError."""
    data = [1, 2, 3]
    pipe: FluentPipe[Any] = FluentPipe(data, {"sample_filter1": sample_filter1})

    with pytest.raises(
        RuntimeError, match="All parameters for curried functions must be passed as keywords"
    ):
        pipe.sample_filter1(2).execute()  # type: ignore


def test_missing_function_call_fail() -> None:
    """Test that forgetting to call a function in the chain raises
    AttributeError or RuntimeError."""
    data = [1, 2, 3]
    pipe: FluentPipe[Any] = FluentPipe(
        data, {"sample_filter1": sample_filter1, "sample_filter2": sample_filter2}
    )

    # Instance where we access the attribute but don't call it
    step = pipe.sample_filter1
    with pytest.raises(AttributeError, match="'sample_filter1' was not correctly called!"):
        # Accessing next attribute should fail because sample_filter1 wasn't called
        _ = step.sample_filter2

    # Test execute failing if a function was added but not called
    pipe._funcs.append(sample_filter1)
    with pytest.raises(RuntimeError, match="Previous func 'sample_filter1' was not called"):
        pipe.execute()


def test_invalid_attribute_access() -> None:
    """Test accessing a non-existent function in the pipeline."""
    data = [1, 2, 3]
    pipe: FluentPipe[Any] = FluentPipe(data, {})

    with pytest.raises(AttributeError, match="attribute not found"):
        _ = pipe.non_existent_func()


def test_curry_validation_fail() -> None:
    """Test that a function with the wrong number of remaining arguments fails execution."""

    @pipeable
    def wrong_func(data: Any, arg1: Any, arg2: Any) -> Any:
        return data

    data = [1]
    # We only provide one arg, so 2 remain (data + one other).
    # FluentPipe expects exactly 1 remaining (data).
    pipe: FluentPipe[Any] = FluentPipe(data, {"wrong_func": wrong_func})

    with pytest.raises(RuntimeError, match="should have exactly 1 remaining parameter"):
        pipe.wrong_func(arg1=1).execute()


def test_registry_borg_pattern() -> None:
    """Test that PipelineRegistry follows the Borg pattern (shared state)."""
    # Clear Borg state before testing to ensure clean start if other tests ran
    PipelineRegistry._borg_state = {}
    reg1: PipelineRegistry[Any] = PipelineRegistry()
    reg2: PipelineRegistry[Any] = PipelineRegistry()

    @reg1.pipeable
    def my_custom_func(data: Any) -> Any:
        return data

    assert "my_custom_func" in reg1.namespace
    assert "my_custom_func" in reg2.namespace
    assert reg1.namespace == reg2.namespace


def test_custom_registry_isolation() -> None:
    """Test creating a fresh pipe with specific functions."""
    data = "hello"

    @pipeable
    def reverse(data: str) -> str:
        return data[::-1]

    # We can pass a specific namespace to FluentPipe directly
    pipe: FluentPipe[Any] = FluentPipe(data, {"rev": reverse})
    assert pipe.rev().execute() == "olleh"


def test_empty_pipeline() -> None:
    """Test that a pipeline with no steps returns the original data."""
    data = {"test": 123}
    pipe: FluentPipe[Any] = FluentPipe(data, {})
    assert pipe.execute() == data
