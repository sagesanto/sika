import numpy as np
import pytest

from sika.modeling import Parameter, ParameterSet
from sika.modeling.priors import NullPriorTransform

@pytest.mark.params
def test_0d():
    p = Parameter("A",NullPriorTransform,frozen=True,values=1)
    assert p.values() == 1
    assert p.values({"random":"coordinate"}) == 1

@pytest.mark.params
def test_init_with_coords():
    coords = {"a":["a","b","c","d"]}
    p = Parameter("A", NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,1,2,3], coords=coords)
    assert p.ndim == 1
    assert p.dims == ("a",)
    assert p.shape == (4,)
    assert p.nvals == 4

@pytest.mark.params
def test_set_empty_coords():
    p = Parameter("A",NullPriorTransform,frozen=True,values=1)
    p.set_coords({})
    assert p.values() == 1
    assert p.values({"random":"coordinate"}) == 1

@pytest.mark.params    
def test_set_coords():
    coords = {"a":["a","b","c","d"]}
    p = Parameter("A", NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,1,2,3])
    p.set_coords(coords)
    assert p.ndim == 1
    assert p.dims == ("a",)
    assert p.shape == (4,)
    assert p.nvals == 4

@pytest.mark.params
def test_1d_flattened():
    coords = {"a":["a","b","c","d"]}
    vals = [0,1,2,3]
    p = Parameter("A", NullPriorTransform(), frozen=True, varies_with=["a"], values=vals, coords=coords)
    assert np.array_equal(p.flattened(),vals)
    
@pytest.mark.params
def test_iter_0d():
    p = Parameter("A", NullPriorTransform(), frozen=True, values=1)
    i = [(s,v) for s,v in p]
    assert len(i) == 1
    sel, val = i[0]
    assert val == 1
    assert sel == {}

@pytest.mark.params
def test_iter_1d():
    coord_v = ["a","b","c","d"]
    coords = {"a":coord_v}
    vals = [0,1,2,3]
    p = Parameter("A", NullPriorTransform(), frozen=True, varies_with=["a"], values=vals, coords=coords)
    contents = [(s,v) for s,v in p]
    assert len(contents) == len(coord_v)
    for i, (sel, val) in enumerate(contents):
        assert val == i
        assert sel['a'] == coord_v[i]
        
@pytest.mark.params
def test_values_1d():
    coord_v = ["a","b","c","d"]
    coords = {"a":coord_v}
    vals = [0,1,2,3]
    p = Parameter("A", NullPriorTransform(), frozen=True, varies_with=["a"], values=vals, coords=coords)
    for i, c_v in enumerate(coord_v):
        val = p.values({"a":c_v})
        assert val == i
        
@pytest.mark.params
def test_set_from_flat():
    coord_v = ["a","b","c","d"]
    coords = {"a":coord_v}
    vals = [0,1,2,3]
    p = Parameter("A", NullPriorTransform(), varies_with=["a"], coords=coords)
    p.set_from_flat(vals)
    contents = [(s,v) for s,v in p]
    assert len(contents) == len(coord_v)
    for i, (sel, val) in enumerate(contents):
        assert val == i
        assert sel['a'] == coord_v[i]


if __name__ == "__main__":
    pytest.main()