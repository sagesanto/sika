import numpy as np
import pytest

from sika.modeling import Parameter, ParameterSet, Constraint, ListConstraint, ConstraintError
from sika.modeling.priors import NullPriorTransform

@pytest.mark.constraints
def test_0d_const():
    p = Parameter("A",NullPriorTransform(), frozen=True, values=1)
    c_pass = Constraint(p, 0, lambda a,b: a>b)
    c_fail = Constraint(p, 0, lambda a,b: a<b)
    assert c_pass.validate()
    assert not c_fail.validate()
    
@pytest.mark.constraints
def test_1d_const():
    coords = {"a":["a","b","c","d"]}
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_pass = Constraint(p, 0, lambda a,b: a>b)
    c_fail = Constraint(p, 0, lambda a,b: a<b)
    assert c_pass.validate()
    assert not c_fail.validate()
    
@pytest.mark.constraints
def test_1d_const_mixed():
    coords = {"a":["a","b","c","d"]}
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,0,1,0],coords=coords)
    c_gt = Constraint(p, 0, lambda a,b: a>b)
    c_lt = Constraint(p, 0, lambda a,b: a<b)
    assert not c_gt.validate()
    assert not c_lt.validate()
    
@pytest.mark.constraints
def test_1d_1d():
    coords = {"a":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,0,0,0],coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    assert c_gt.validate()
    assert not c_lt.validate()

@pytest.mark.constraints
def test_1d_2d():
    """test that constraining a 1d param with a 2d param fails in init"""
    coords_1d = {"a":["a","b","c","d"]}
    coords = {"a":["a","b","c","d"],"b":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), varies_with=["a","b"], coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1], coords=coords_1d)
    with pytest.raises(ConstraintError):
        c_gt = Constraint(p, p_comp, lambda a,b: a>b)

@pytest.mark.constraints
def test_2d_1d():
    """test that a 2d param can be constrained by a 1d param with a subset of its coords"""
    coords_1d = {"a":["a","b","c","d"]}
    coords = {"a":["a","b","c","d"],"b":["a","b","c","d"]}
    p = Parameter("A",NullPriorTransform(), varies_with=["a","b"], coords=coords)
    p_comp = Parameter("B",NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,0,0,0], coords=coords_1d)
    
    p.set_from_flat(np.ones(16))
    
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    assert c_gt.validate()
    assert not c_lt.validate()

@pytest.mark.constraints
def test_1d_1d_bad_coords():
    """test that constraining a 1d param with 1d param that varies along a different axis fails in init"""
    coords = {"a":["a","b","c","d"],"b":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), varies_with=["b"], coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1], coords=coords)
    with pytest.raises(ConstraintError):
        _ = Constraint(p, p_comp, lambda a,b: a>b)
        
@pytest.mark.constraints
def test_2d_2d():
    coords = {"a":["a","b","c","d"],"b":["a","b","c","d"]}
    p = Parameter("A",NullPriorTransform(), varies_with=["a","b"], coords=coords)
    p_comp = Parameter("B",NullPriorTransform(), varies_with=["a","b"], coords=coords)
    
    p.set_from_flat(np.ones(16))
    p_comp.set_from_flat(np.zeros(16))
    
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    assert c_gt.validate()
    assert not c_lt.validate()
    
@pytest.mark.constraints
def test_0dchange_const():
    p = Parameter("A",NullPriorTransform(), frozen=True, values=1)
    c_pass = Constraint(p, 0, lambda a,b: a>b)
    c_fail = Constraint(p, 0, lambda a,b: a<b)
    assert c_pass.validate()
    assert not c_fail.validate()
    p.set_from_flat(-1)
    assert not c_pass.validate()
    assert c_fail.validate()
    
@pytest.mark.constraints
def test_1dchange_const():
    coords = {"a":["a","b","c","d"]}
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_pass = Constraint(p, 0, lambda a,b: a>b)
    c_fail = Constraint(p, 0, lambda a,b: a<b)
    assert c_pass.validate()
    assert not c_fail.validate()
    p.set_from_flat([-1,-1,-1,-1])
    assert not c_pass.validate()
    assert c_fail.validate()

@pytest.mark.constraints
def test_1dchange_1d():
    coords = {"a":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,0,0,0],coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    
    p.set_from_flat([-1,-1,-1,-1])
    
    assert not c_gt.validate()
    assert c_lt.validate()

@pytest.mark.constraints
def test_1d_1dchange():
    coords = {"a":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,0,0,0],coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    
    p_comp.set_from_flat([2,2,2,2])
    
    assert not c_gt.validate()
    assert c_lt.validate()
    
@pytest.mark.constraints
def test_1dchange_1dchange():
    coords = {"a":["a","b","c","d"]}
    p_comp = Parameter("B",NullPriorTransform(), frozen=True, varies_with=["a"], values=[0,0,0,0],coords=coords)
    p = Parameter("A",NullPriorTransform(), frozen=True, varies_with=["a"], values=[1,1,1,1],coords=coords)
    c_gt = Constraint(p, p_comp, lambda a,b: a>b)
    c_lt = Constraint(p, p_comp, lambda a,b: a<b)
    
    p.set_from_flat([0,0,-1,-1])
    p_comp.set_from_flat([1,1,0,0])
    
    assert not c_gt.validate()
    assert c_lt.validate()
    
if __name__ == "__main__":
    pytest.main()