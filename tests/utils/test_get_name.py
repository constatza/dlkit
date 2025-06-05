# test_get_name.py
from dlkit.utils.general import get_name


# 1. Top‐level function
def sample_function():
    return "hello"


def test_get_name_top_level_function():
    assert get_name(sample_function) == "sample_function"


# 2. Nested (inner) function
def outer_function():
    def inner_function():
        return "nested"

    return inner_function


def test_get_name_nested_function():
    inner = outer_function()
    # __qualname__ for nested: "outer_function.<locals>.inner_function"
    # But get_name uses __name__, so result is just "inner_function"
    assert get_name(inner) == "inner_function"


# 3. Class
class SampleClass:
    def __init__(self):
        self.attribute = "attribute"

    def method(self):
        return self.attribute


def test_get_name_class():
    assert get_name(SampleClass) == "SampleClass"


# 4. Class instance
def test_get_name_instance():
    instance = SampleClass()
    assert get_name(instance) == "SampleClass"
