from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from jinja2 import BaseLoader, StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from toolz import curry


@curry
def date_format(s: str, frmt: str) -> str:
    d = parse(s)
    if frmt == "isoformat":
        return d.isoformat()
    else:
        return d.strftime(frmt)


def datap(s: str) -> str:
    d = parse(s)
    if 2020 <= d.year < 2030:
        data_p = "G" + str(d.year - 2020)
    elif 2030 <= d.year < 2040:
        data_p = "H"
    elif 2040 <= d.year < 2050:
        data_p = "I"
    else:
        raise ValueError(f"Unexpected DataP date: {d}")

    if d.month < 10:
        data_p += str(d.month)
    else:
        data_p += d.strftime("%b")[0].upper()

    data_p += str(d.day).zfill(2)
    return data_p


@curry
def date_delta(
    s: str, element: Literal["years", "months", "weeks", "days"], rel: int
) -> str:
    """Transforms a string datetime moving a relative number `rel` of `elements`

    :param s: string datetime
    :param element: "years", "months", "weeks", "days"
    :param rel: Positive or negative number of `elements` to move s string date
    :return: new string datetime
    """
    d = parse(s)
    param = {element: rel}
    d = d + relativedelta(**param)
    return d.isoformat()


@curry
def tpl_timezone(s: str, tz: str) -> str:
    d = parse(s)
    if not d.tzinfo:  # naive
        d = d.replace(tzinfo=ZoneInfo(tz))
    else:
        d = d.astimezone(ZoneInfo(tz))
    return d.isoformat()


def month_last_day(s: str) -> str:
    d = parse(s)
    d = d.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
    return d.isoformat()


def month_first_day(s: str) -> str:
    d = parse(s)
    d = d.replace(day=1)
    return d.isoformat()


# Define safe string operations
SAFE_OPERATIONS: dict[str, Callable] = {
    "datap": datap,
    "date_delta": date_delta,
    "date_format": date_format,
    "timezone": tpl_timezone,
    # string filters
    "title": str.title,
    "capitalize": str.capitalize,
    "strip": str.strip,
    "lstrip": str.lstrip,
    "rstrip": str.rstrip,
    # easy access alias methods
    "isoformat": date_format(frmt="isoformat"),
    "year": date_format(frmt="%Y"),
    "month": date_format(frmt="%m"),
    "day": date_format(frmt="%d"),
    "date": date_format(frmt="%Y-%m-%d"),
    "madrid_tz": tpl_timezone(tz="Europe/Madrid"),
    "utc_tz": tpl_timezone(tz="UTC"),
    "year_before": date_delta(element="years", rel=-1),
    "month_before": date_delta(element="months", rel=-1),
    "week_before": date_delta(element="weeks", rel=-1),
    "day_before": date_delta(element="days", rel=-1),
    "year_after": date_delta(element="years", rel=1),
    "month_after": date_delta(element="months", rel=1),
    "week_after": date_delta(element="weeks", rel=1),
    "day_after": date_delta(element="days", rel=1),
    "month_last_day": month_last_day,
    "month_first_day": month_first_day,
    # Simple formats for date parts
    "syear": date_format(frmt="%y"),
    "smonth": date_format(frmt="%b"),
    # Add more safe operations as needed
}

env = ImmutableSandboxedEnvironment(loader=BaseLoader(), undefined=StrictUndefined)
env.filters.update(SAFE_OPERATIONS)


def process_jinja2_template(template: str, context: dict[str, Any]) -> str:
    j_template = env.from_string(template, globals=context)
    result = j_template.render()
    return result
