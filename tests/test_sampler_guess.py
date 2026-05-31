import numpy as np

from sika.modeling import Dataset, Model, Parameter, Sampler
from sika.modeling.parameter_set import AuxiliaryParameterSet, ParameterSet
from sika.modeling.priors import NullPriorTransform
from sika.product import Product


class MockProduct(Product):
    pass


class MockParameterSet(ParameterSet):
    def __init__(self, name, rv, teff):
        self.name = name
        self.rv = rv
        self.teff = teff
        self.setup()


class MockModel(Model[MockProduct]):
    def make_model(self) -> Dataset[MockProduct]:
        return Dataset(MockProduct(parameters={}, metadata={}))


class MockSampler(Sampler[MockProduct, MockProduct]):
    def _make_model(self) -> Dataset[MockProduct]:
        return Dataset(MockProduct(parameters={}, metadata={}))

    def get_errors_and_residuals(self, modeled_ds: Dataset[MockProduct]):
        return np.array([1.0]), np.array([0.0])


def test_sampler_flattened_guess_matches_param_name_order():
    coords = {"night": ["n3", "n1", "n2"]}
    model_pset = MockParameterSet(
        name="model",
        rv=Parameter("rv", NullPriorTransform(), varies_with=["night"], coords=coords, guess=[10, 11, 12]),
        teff=Parameter("teff", NullPriorTransform(), guess=13),
    )
    aux_pset = AuxiliaryParameterSet(
        name="aux",
        jitter=Parameter("jitter", NullPriorTransform(), guess=14),
    )
    model = MockModel(model_pset)
    sampler = MockSampler("test", ".", Dataset(MockProduct(parameters={}, metadata={})), [model], aux_params=aux_pset)

    np.testing.assert_array_equal(sampler.flattened_guess, np.array([10, 11, 12, 13, 14]))
    assert sampler.param_names == [
        "model: rv (night=n3)",
        "model: rv (night=n1)",
        "model: rv (night=n2)",
        "model: teff",
        "aux: jitter",
    ]


def test_sampler_flattened_guess_is_none_if_any_parameter_guess_is_missing():
    coords = {"night": ["n1", "n2"]}
    model_pset = MockParameterSet(
        name="model",
        rv=Parameter("rv", NullPriorTransform(), varies_with=["night"], coords=coords, guess=[1, 2]),
        teff=Parameter("teff", NullPriorTransform()),
    )
    model = MockModel(model_pset)
    sampler = MockSampler("test", ".", Dataset(MockProduct(parameters={}, metadata={})), [model])

    assert sampler.flattened_guess is None