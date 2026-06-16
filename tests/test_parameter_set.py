import pytest
import numpy as np
import xarray as xr
from sika.modeling.parameter_set import ParameterSet, AuxiliaryParameterSet
from sika.modeling.params import Parameter
from sika.modeling.priors import Uniform

def _extract_groupby_value(result):
    # result may be a tuple of (selector, value), or (selector, value, ...)
    # We want the first non-selector (non-dict) value, which is usually at index 1 or 2
    if isinstance(result, tuple):
        # skip selector dicts
        for item in result[1:]:
            if isinstance(item, dict):
                return item
            # xarray.Dataset or similar
            if hasattr(item, "__getitem__") and ("p1" in item or "p2" in item):
                return item
        raise TypeError(f"No suitable value found in groupby result tuple: {result}")
    return result

def _extract_vals(vals):
    # vals may be a dict or an xarray.Dataset
    if isinstance(vals, dict):
        return vals
    elif hasattr(vals, "__getitem__") and ("p1" in vals or "p2" in vals):
        out = {}
        for k in ("p1", "p2"):
            if k in vals:
                v = vals[k]
                out[k] = v.item() if hasattr(v, "item") else v
        return out
    else:
        raise TypeError(f"Unexpected groupby value type: {type(vals)}")

def _extract_p1(vals):
    # vals may be a dict or an xarray.Dataset
    if isinstance(vals, dict):
        return vals["p1"]
    elif hasattr(vals, "__getitem__") and "p1" in vals:
        v = vals["p1"]
        return v.item() if hasattr(v, "item") else v
    else:
        raise TypeError(f"Unexpected groupby value type: {type(vals)}")
def test_multiple_params_mixed_coords():
    # p1 varies with coords, p2 does not
    coords = {"night": ["n1", "n2"], "order": [0, 1]}
    values1 = np.array([[1, 2], [3, 4]])
    values2 = 42  # scalar param
    p1 = make_simple_param("p1", (2, 2), coords=coords, values=values1)
    p2 = make_simple_param("p2", (1,), coords={}, values=values2)
    ps = AuxiliaryParameterSet(p1=p1, p2=p2)
    ps.set_coords(coords)
    ds = ps.as_xarray()
    # Check shapes
    assert ds["p1"].shape == (2, 2)
    assert ds["p2"].shape == ()
    # Check values
    np.testing.assert_array_equal(ds["p1"].values, values1)
    np.testing.assert_equal(ds["p2"].values, values2)
    # Check selector iteration
    iter_vals = [(v["p1"], v["p2"]) for _, v in ps]
    # There should be 4 selectors (2x2)
    assert len(iter_vals) == 4
    for i, (p1val, p2val) in enumerate(iter_vals):
        assert p2val == 42
        # p1val should match the flattened order of values1
    np.testing.assert_array_equal([v[0] for v in iter_vals], values1.flatten())

def test_multiple_params_all_coords():
    # Both params vary with coords
    coords = {"night": ["n1", "n2"], "order": [0, 1]}
    values1 = np.array([[10, 20], [30, 40]])
    values2 = np.array([[100, 200], [300, 400]])
    p1 = make_simple_param("p1", (2, 2), coords=coords, values=values1)
    p2 = make_simple_param("p2", (2, 2), coords=coords, values=values2)
    ps = AuxiliaryParameterSet(p1=p1, p2=p2)
    ps.set_coords(coords)
    ds = ps.as_xarray()
    # Check shapes
    assert ds["p1"].shape == (2, 2)
    assert ds["p2"].shape == (2, 2)
    # Check values
    np.testing.assert_array_equal(ds["p1"].values, values1)
    np.testing.assert_array_equal(ds["p2"].values, values2)
    # Check selector iteration
    iter_vals = [(v["p1"], v["p2"]) for _, v in ps]
    assert len(iter_vals) == 4
    np.testing.assert_array_equal([v[0] for v in iter_vals], values1.flatten())
    np.testing.assert_array_equal([v[1] for v in iter_vals], values2.flatten())

def test_multiple_params_groupby():
    # Groupby with mixed params
    coords = {"night": ["n1", "n2"], "order": [0, 1]}
    values1 = np.array([[5, 6], [7, 8]])
    values2 = 99
    p1 = make_simple_param("p1", (2, 2), coords=coords, values=values1)
    p2 = make_simple_param("p2", (1,), coords={}, values=values2)
    ps = AuxiliaryParameterSet(p1=p1, p2=p2)
    ps.set_coords(coords)
    groupby_results = list(ps.groupby(["night", "order"]))
    for result in groupby_results:
        vals = _extract_groupby_value(result)
        vdict = _extract_vals(vals)
        assert set(vdict.keys()) == {"p1", "p2"}
        assert vdict["p2"] == 99
    expected_p1 = values1.flatten()
    actual_p1 = [_extract_vals(_extract_groupby_value(result))["p1"] for result in groupby_results]
    np.testing.assert_array_equal(actual_p1, expected_p1)


def make_simple_param(name, shape, coords=None, values=None):
    prior = Uniform(0, 1)
    if coords is None:
        coords = {f"dim_{i}": list(range(s)) for i, s in enumerate(shape)}
    return Parameter(name, prior_transform=prior, varies_with=list(coords.keys()), coords=coords, values=values)

def test_as_xarray_consistency():
    coords = {"night": ["n1", "n2"], "order": [0, 1, 2]}
    values = np.arange(6).reshape(2, 3)
    p1 = make_simple_param("p1", (2, 3), coords=coords, values=values)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.name = "testset"
    ps.set_coords(coords)
    ds = ps.as_xarray()
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"p1"}
    assert ds["p1"].shape == (2, 3)
    expected = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(ds["p1"].values, expected)

# check that the groupby values are in the expected order
def test_groupby_consistency():
    coords = {"night": ["n1", "n2"], "order": [0, 1, 2]}
    values = np.arange(6).reshape(2, 3)
    p1 = make_simple_param("p1", (2, 3), coords=coords, values=values)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.name = "testset"
    ps.set_coords(coords)
    ds = ps.as_xarray()
    groupby_results = list(ps.groupby(["night", "order"]))
    arr = ds["p1"].values.flatten()
    expected = np.arange(6)
    groupby_vals = [_extract_p1(_extract_groupby_value(result)) for result in groupby_results]
    np.testing.assert_array_equal(groupby_vals, expected)

# check that iterating over selectors matches as_xarray order
def test_selector_iteration_matches_as_xarray():
    coords = {"night": ["n1", "n2"], "order": [0, 1]}
    values = np.array([[10, 20], [30, 40]])
    p1 = make_simple_param("p1", (2, 2), coords=coords, values=values)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.set_coords(coords)
    ds = ps.as_xarray()
    iter_vals = [v["p1"] for _, v in ps]
    arr = ds["p1"].values.flatten()
    np.testing.assert_array_equal(arr, iter_vals)

def test_coord_shape_and_dims():
    coords = {"night": ["n1", "n2", "n3"], "order": [0, 1]}
    p1 = make_simple_param("p1", (3, 2), coords=coords)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.set_coords(coords)
    assert ps.coord_shape == (3, 2)
    assert set(ps.dims) == {"night", "order"}
    assert ps.ndim == 2

def test_set_values_flat_and_direct():
    coords = {"night": ["n1", "n2"], "order": [0, 1]}
    p1 = make_simple_param("p1", (2, 2), coords=coords)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.set_coords(coords)
    flat_vals = np.array([1, 2, 3, 4])
    ps.set_values_flat(flat_vals)
    arr = ps.as_xarray()["p1"].values.flatten()
    np.testing.assert_array_equal(arr, flat_vals)
    ps2 = AuxiliaryParameterSet(p1=p1)
    ps2.set_coords(coords)
    ps2.set_values_direct([np.array([[5, 6], [7, 8]])])
    arr2 = ps2.as_xarray()["p1"].values.flatten()
    np.testing.assert_array_equal(arr2, [5, 6, 7, 8])

def test_flattened_guess_matches_parameter_order():
    coords = {"night": ["n2", "n1"], "order": [1, 0]}
    p1 = Parameter(
        "p1",
        prior_transform=Uniform(0, 1),
        varies_with=["night", "order"],
        coords=coords,
        guess=np.array([[1, 2], [3, 4]]),
    )
    p2 = Parameter("p2", prior_transform=Uniform(0, 1), guess=5)
    ps = AuxiliaryParameterSet(p1=p1, p2=p2)
    ps.set_coords(coords)
    np.testing.assert_array_equal(ps.flattened_guess(), np.array([1, 2, 3, 4, 5]))
    
def test_guess_order_matches_values_order_flattened():
    coords = {"night": ["n2", "n1"], "order": [1, 0]}
    p1 = Parameter(
        "p1",
        prior_transform=Uniform(0, 1),
        varies_with=["night", "order"],
        coords=coords,
        guess=np.array([[1, 2], [3, 4]]),
        values=np.array([[1, 2], [3, 4]]),
    )
    np.testing.assert_array_equal(p1.flattened(), p1.flattened_guess())
    
def test_guess_order_matches_values_order_selected():
    coords = {"night": ["n2", "n1"], "order": [1, 0]}
    p1 = Parameter(
        "p1",
        prior_transform=Uniform(0, 1),
        varies_with=["night", "order"],
        coords=coords,
        guess=np.array([[1, 2], [3, 4]]),
        values=np.array([[1, 2], [3, 4]]),
    )
    for sel in p1.selectors:
        assert p1.values(sel) == p1.guess(sel)

def test_flattened_guess_is_none_if_any_missing_guess():
    coords = {"night": ["n1", "n2"]}
    p1 = Parameter("p1", prior_transform=Uniform(0, 1), varies_with=["night"], coords=coords, guess=[1, 2])
    p2 = Parameter("p2", prior_transform=Uniform(0, 1))
    ps = AuxiliaryParameterSet(p1=p1, p2=p2)
    ps.set_coords(coords)
    assert ps.flattened_guess() is None

def test_all_names_consistency():
    coords = {"night": ["n1", "n2"]}
    p1 = make_simple_param("p1", (2,), coords=coords)
    ps = AuxiliaryParameterSet(p1=p1)
    ps.set_coords(coords)
    names = ps.all_names()
    assert all(ps.name in n for n in names)

def test_empty_parameterset():
    from sika.modeling.parameter_set import EmptyParameterSet
    ps = EmptyParameterSet()
    assert ps.name == "empty"
    assert ps.params == []
    assert ps.unfrozen == []
    assert hasattr(ps, "setup")
