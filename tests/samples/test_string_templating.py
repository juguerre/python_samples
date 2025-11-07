"""Tests for string_templating module."""

from datetime import datetime, timezone

import pytest

from src.samples.string_templating import (
    datap,
    date_delta,
    date_format,
    month_first_day,
    month_last_day,
    process_jinja2_template,
    tpl_timezone,
)


class TestDateFunctions:
    """Test date manipulation functions."""

    @pytest.mark.parametrize(
        "date_str, fmt, expected",
        [
            ("2023-01-15", "%Y-%m", "2023-01"),
            ("2023-01-15T12:00:00", "%Y-%m-%d %H:%M", "2023-01-15 12:00"),
            ("2023-01-15", "isoformat", "2023-01-15T00:00:00"),
        ],
    )
    def test_date_format(self, date_str: str, fmt: str, expected: str) -> None:
        """Test date_format function with different formats."""
        assert date_format(date_str, fmt) == expected

    @pytest.mark.parametrize(
        "date_str, expected",
        [
            ("2023-01-15", "G3115"),
            ("2029-12-31", "G9D31"),
            ("2030-01-01", "H101"),
            ("2040-01-01", "I101"),
        ],
    )
    def test_datap(self, date_str: str, expected: str) -> None:
        """Test datap function with different dates."""
        assert datap(date_str) == expected

    @pytest.mark.parametrize(
        "date_str, element, rel, expected",
        [
            ("2023-01-15", "days", 5, "2023-01-20T00:00:00"),
            ("2023-01-15", "months", -1, "2022-12-15T00:00:00"),
            ("2023-01-15", "years", 2, "2025-01-15T00:00:00"),
        ],
    )
    def test_date_delta(
        self, date_str: str, element: str, rel: int, expected: str
    ) -> None:
        """Test date_delta function with different time deltas."""
        assert date_delta(date_str, element, rel) == expected

    def test_timezone_conversion(self) -> None:
        """Test timezone conversion."""
        # Test with naive datetime
        result = tpl_timezone("2023-01-15T12:00:00", "Europe/Madrid")
        assert (
            "2023-01-15T12:00:00+01:00" in result
        )  # +01:00 or +02:00 depending on DST

        # Test with timezone-aware datetime
        dt = datetime(2023, 1, 15, 12, 0, tzinfo=timezone.utc)
        result = tpl_timezone(dt.isoformat(), "Europe/Madrid")
        assert "2023-01-15T13:00:00+01:00" in result

    def test_month_last_day(self) -> None:
        """Test month_last_day function."""
        assert month_last_day("2023-02-15").startswith("2023-02-28")
        assert month_last_day("2024-02-15").startswith("2024-02-29")  # Leap year
        assert month_last_day("2023-04-30").endswith("30T00:00:00")  # 30-day month

    def test_month_first_day(self) -> None:
        """Test month_first_day function."""
        assert month_first_day("2023-02-15").startswith("2023-02-01")
        assert month_first_day("2023-12-31").startswith("2023-12-01")


class TestTemplateProcessing:
    """Test template processing with Jinja2."""

    def test_basic_template(self) -> None:
        """Test basic template rendering."""
        template = "Hello {{ name }}!"
        context = {"name": "World"}
        assert process_jinja2_template(template, context) == "Hello World!"

    def test_date_manipulation_in_template(self) -> None:
        """Test date manipulation in templates."""
        template = """
        Date: {{ date | date }}
        Year: {{ date | year }}
        Month: {{ date | month }}
        Day: {{ date | day }}
        Next day: {{ date | day_after | date }}
        """
        context = {"date": "2023-01-15"}
        result = process_jinja2_template(template, context).strip()
        assert "Date: 2023-01-15" in result
        assert "Year: 2023" in result
        assert "Month: 01" in result
        assert "Day: 15" in result
        assert "Next day: 2023-01-16" in result

    def test_string_operations(self) -> None:
        """Test string operations in templates."""
        template = """
        Original: "{{ text }}"
        Strip: "{{ text | strip }}"
        Upper: {{ text | upper }}
        Title: {{ text | strip | title }}
        """

        context = {"text": "  hello world  "}
        result = process_jinja2_template(template, context).strip()
        assert 'Original: "  hello world  "' in result
        assert 'Strip: "hello world"' in result
        assert "Upper:   HELLO WORLD  " in result
        assert "Title: Hello World" in result

    def test_timezone_conversion_in_template(self) -> None:
        """Test timezone conversion in templates."""
        template = """
        Original: {{ date }}
        Madrid: {{ date | madrid_tz }}
        UTC: {{ date | utc_tz }}
        """
        context = {"date": "2023-01-15T12:00:00+00:00"}
        result = process_jinja2_template(template, context).strip()
        assert "Original: 2023-01-15T12:00:00+00:00" in result
        assert "+01:00" in result or "+02:00" in result  # Depends on DST

    def test_undefined_variable_raises_error(self) -> None:
        """Test that undefined variables raise an error."""
        with pytest.raises(Exception):  # Should raise UndefinedError from Jinja2
            process_jinja2_template("{{ undefined_var }}", {})

    def test_complex_date_manipulation(self) -> None:
        """Test complex date manipulation in templates."""
        template = """
        Month first day: {{ date | month_first_day | date }}
        Month last day: {{ date | month_last_day | date }}
        Previous month last day: {{ date | month_before | month_last_day | date }}
        """
        context = {"date": "2023-03-15"}
        result = process_jinja2_template(template, context).strip()
        assert "Month first day: 2023-03-01" in result
        assert "Month last day: 2023-03-31" in result
        assert "Previous month last day: 2023-02-28" in result
