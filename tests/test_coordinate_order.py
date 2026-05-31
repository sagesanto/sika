import numpy as np

from sika.modeling import Dataset, Parameter, AuxiliaryParameterSet
from sika.modeling.priors import NullPriorTransform
from sika.product import Product

def many_nights(n_nights: int = 60):
    descending_odds = [f"night_{i:03d}" for i in range(n_nights - 1, -1, -2)]
    ascending_evens = [f"night_{i:03d}" for i in range(0, n_nights, 2)]
    return descending_odds + ascending_evens

def dataset_for_nights(nights):
    products = [Product(parameters={}, metadata={"night": night}) for night in nights]
    return Dataset(products, dims=["night"])

def test_dataset_preserves_coordinate_order():
    nights = many_nights()

    for _ in range(30):
        n = list(np.random.choice(nights, 30, replace=False))
        dataset = dataset_for_nights(n)

        assert dataset.coords["night"] == n  # check that coords match
        assert [selector["night"] for selector in dataset.selectors] == n  # check that selectors match
        assert [dataset.values({'night':night}).metadata['night'] for night in n] == n  # check that retrieved products match (dataset_for_nights writes the corresponding night into the metadata of the product)

def test_parameter_order_follows_dataset_order():
    nights = many_nights()
    expected_names = [f"test: rv (night={night})" for night in nights] + ["test: vsini"]
    expected_flat = np.arange(len(nights) + 1)

    for _ in range(30):
        dataset = dataset_for_nights(nights)
        parameter_set = AuxiliaryParameterSet(
            name="test",
            rv=Parameter("rv", NullPriorTransform(), varies_with=["night"]),
            vsini=Parameter("vsini", NullPriorTransform()),
        )
        parameter_set.set_coords(dataset.coords)
        parameter_set.set_values_flat(expected_flat)

        assert parameter_set.all_names() == expected_names
        assert parameter_set.rv.flattened().tolist() == expected_flat[:-1].tolist()
        assert parameter_set.vsini.values() == expected_flat[-1]

        for index, night in enumerate(nights):
            assert parameter_set.rv.values({"night": night}) == index