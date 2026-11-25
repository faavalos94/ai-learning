import pytest

from tools_chat import calculate


@pytest.mark.parametrize(
    "a, b, operation, expected",
    [
        (2, 3, "add", 5),
        (10, 4, "subtract", 6),
        (6, 7, "multiply", 42),
        (9, 2, "divide", 4.5),
    ],
)
def test_operations(a, b, operation, expected):
    assert calculate(a, b, operation) == expected


def test_divide_by_zero():
    assert calculate(5, 0, "divide") == "Error: division by zero"


def test_unknown_operation():
    assert calculate(2, 3, "power") == "Error: unknown operation power"
