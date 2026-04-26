"""Tests for internal _color module."""
from maskgeo._color import _assign_colors


def test_assign_colors_returns_dict_with_input_keys():
    result = _assign_colors(["Tumor", "Stroma"])
    assert set(result.keys()) == {"Tumor", "Stroma"}


def test_assign_colors_first_six_are_distinct_primaries():
    result = _assign_colors(["a", "b", "c", "d", "e", "f"])
    colors = list(result.values())
    assert len(set(map(tuple, colors))) == 6, "first 6 colors must all differ"


def test_assign_colors_returns_rgb_lists():
    result = _assign_colors(["x"])
    assert isinstance(result["x"], list)
    assert len(result["x"]) == 3
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in result["x"])


def test_assign_colors_deterministic():
    a = _assign_colors(["A", "B", "C"])
    b = _assign_colors(["A", "B", "C"])
    assert a == b


def test_assign_colors_handles_more_than_six():
    """For n > 6, must still produce n distinct colors."""
    names = [f"n{i}" for i in range(20)]
    result = _assign_colors(names)
    assert len(result) == 20
    distinct = {tuple(c) for c in result.values()}
    assert len(distinct) == 20, "all 20 colors must be distinct"


def test_assign_colors_empty():
    assert _assign_colors([]) == {}
