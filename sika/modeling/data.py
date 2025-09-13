__all__ = ["Dataset", "DataLoader"] #, "ProviderDataLoader"]

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, Dict, TypeVar, Generic, Tuple, Collection, Optional
import numpy as np, xarray as xr
from logging import Logger
import itertools

from sika.product import Product
from sika.provider import Provider
from sika.config import Config
from sika.utils import joint_iter as joint_iter_generic

# this works for rectangular collections of products but not ragged ones

#: Any :py:class:`~sika.product.Product` or :py:class:`~sika.product.Product` subclass
T = TypeVar('T', bound='Product', covariant=True)
class Dataset(Generic[T], ABC):
    """An n-dimensional collection of :py:class:`Products <sika.product.Product>` with labeled dimensions. Each :py:class:`Product's <sika.product.Product>` :py:attr:`~sika.product.Product.metadata` is used to determine where in the grid it should be placed, and so must have a key-value pair for each of the dimensions of the dataset. 
    
    For example, consider 6 :py:class:`~sika.implementations.spectroscopy.spectra.Spectrum` objects, each taken on one of two different nights in one of three different bands: ::
        
        spectra = [
            Spectrum(flux=f1,wlen=w1,parameters={}, metadata={"night":1,"band":"y"}),
            Spectrum(flux=f2,wlen=w2,parameters={}, metadata={"night":1,"band":"j"}),
            Spectrum(flux=f3,wlen=w3,parameters={}, metadata={"night":1,"band":"k"}),
            Spectrum(flux=f4,wlen=w4,parameters={}, metadata={"night":2,"band":"y"}),
            Spectrum(flux=f5,wlen=w5,parameters={}, metadata={"night":2,"band":"j"}),
            Spectrum(flux=f6,wlen=w6,parameters={}, metadata={"night":2,"band":"k"}),
        ]
        
    We can construct a ``Dataset`` from this list and specify that the dimensions are 'night' and 'band': :: 
    
        d = Dataset(data=spectra, dims=["night","band"])
        
    We can see that the Dataset's coordinates have been derived from the spectra that comprise it: ::
    
        d.coords
        >>> {'night': [1, 2], 'band': ['k', 'y', 'j']}
        
    .. hint::
        The names and values of the coordinates are arbitrary (within reason...) and set by the metadata of the products. Keys in a product's metadata that don't correspond to one of the specified dimensions will be safely ignored.
        
    
    We can select one Product from the dataset by specifying the coordinates that we're interested in: ::
    
        d.values({"night": 1, "band": 'y'})
        >>> Spectrum(parameters={}, metadata={'night': 1, 'band': 'y'}, ... )
    
    We can select groups of products by specifying only some of the coordinates and letting the others vary: ::
    
        d.values({"night": 1})
        >>> <xarray.DataArray (band: 3)> Size: 24B
        >>>     array([
        >>>            Spectrum(parameters={}, metadata={'night': 1, 'band': 'k'}, ... ),
        >>>            Spectrum(parameters={}, metadata={'night': 1, 'band': 'y'}, ... ),
        >>>            Spectrum(parameters={}, metadata={'night': 1, 'band': 'j'}, ... )
        >>>         ], dtype=object)
        >>>     Coordinates:
        >>>         night    int64 8B 1
        >>>     * band   (band) <U1 12B 'k' 'y' 'j'    
    
    The dataset's :py:attr:`~sika.modeling.data.Dataset.selectors` property is a list of dictionaries that can be used to index each Product, which is useful for when you just want to flatten a dataset: ::
    
        d.selectors
        >>> [{'night': 1, 'band': 'k'},
        >>>  {'night': 1, 'band': 'y'},
        >>>  {'night': 1, 'band': 'j'},
        >>>  {'night': 2, 'band': 'k'},
        >>>  {'night': 2, 'band': 'y'},
        >>>  {'night': 2, 'band': 'j'}]
        d.values(d.selectors[0])
        >>> Spectrum(parameters={}, metadata={'night': 1, 'band': 'k'}, ... )
        
    For convenience, iterating through a Dataset yields (selector, Product) pairs: ::
    
        for (sel, prod) in d: 
            print(f"{sel}, {prod}")
        >>> {'night': 1, 'band': 'k'}, Spectrum(parameters={}, metadata={'night': 1, 'band': 'k'}, . . . )
        >>> {'night': 1, 'band': 'y'}, Spectrum(parameters={}, metadata={'night': 1, 'band': 'y'}, . . . )
        >>> . . .
        >>> {'night': 2, 'band': 'j'}, Spectrum(parameters={}, metadata={'night': 2, 'band': 'j'}, . . . )

    Datasets can be indexed using selectors that contain keys that are *not* in their coordinates, which is helpful when performing operations on multiple datasets in tandem. 
    For example, consider comparing each Spectrum in ``d`` to a dataset of model spectra, ``models``, that varies by band but not by night: ::

        models.coords
        >>> {'band': ['k', 'y', 'j']}
        models.values({'band': 'j'}) == model.values({'band':'j', 'night':1})  # selecting with extra coordinates has no effect
        >>> True
        residuals = []
        for sel in d.selectors:
            model = models.values(sel)  # get a model using a selector like {'night':1,'band':'j'}, even though night is ignored
            data = d.values(sel) # get data using the same selector
            residuals.append(data.flux-model.flux)
    
    We could achieve the same thing more easily by using :py:meth:`Dataset.joint_iter` : ::

        residuals = []
        for sel, (data, model) in Dataset.joint_iter(d,models):
            residuals.append(data.flux-model.flux)
    
    We can also do broadcasting even when each dataset has dimensions that are not found in the other. 
    Consider comparing each :py:class:`~sika.implementations.spectroscopy.spectra.Spectrum` in ``d`` to a ``Dataset`` of spectra generated by the `Phoenix`_ and `Sonora Elf Owl`_ models: ::
    
        models.coords
        >>> {'band': ['k', 'y', 'j'], 'model': ['phoenix', 'elf owl']}
        residual_spectra = []
        for sel, (data, model) in Dataset.joint_iter(d,models):
            spec = Spectrum(
                parameters = {},
                flux = data.flux - model.flux,
                wlen = data.wlen,
                metadata = sel  # tag this spectrum with the joint selector
            )
            residual_spectra.append(spec)
            
    And then creating a ``Dataset`` from the resulting residual spectra: ::
    
        residuals = Dataset(residual_spectra, dims=['night','band','model'])
        
    ``residuals`` is three-dimensional and contains a Product for each combination of 'night', 'band', and 'model': ::
            
            residuals.shape
            >>> (2,3,2)
            residuals.coords
            >>> {'night': [1, 2], 'band': ['k', 'y', 'j'], 'model': ['phoenix', 'elf owl']}
    
    .. _Phoenix: https://ui.adsabs.harvard.edu/abs/2013A%26A...553A...6H/abstract
    .. _Sonora Elf Owl: https://ui.adsabs.harvard.edu/abs/2024ApJ...963...73M/abstract
    
    """
    def __init__(self, data:Union[Collection[T],T], dims:List[str]=None):
        self.dims = dims if dims is not None else []
        # if data is a single item, make it a list temporarily :(
        try: 
            [_ for _ in data]
        except TypeError:
            data = [data]

        s = self._get_coords(data)
        
        #: dict[str,list[Any]]: The coordinates of this ``Dataset`` - a dictionary of {``dim`` to ``[values along that dim occupied by products]``} for each of this ``Dataset``'s dimensions ``dim``.
        self.coords = s[0]
        
        #: list[dict[str,Any]]: A list of dictionaries specifying the coordinates of each of the Products in the dataset.   
        self.selectors = s[1]
        
        self._data = self._process_data(data)
    
    def _get_coords(self, data: List[T]):
        """
        Get the coordinates of the dataset. Also, generate selectors for each combination of coordinates.
        """
        coords = {k:[] for k in self.dims}
        for i, dim in enumerate(self.dims):
            try:
                coords[dim] += [d.metadata[dim] for d in data]
            except KeyError:
                raise ValueError(f"Dimension {dim} missing from dataset metadata.")
        for k, v in coords.items():
            coords[k] = list(set(v))
            
        c_keys = list(coords.keys())
        c_vals = list(coords.values())
        all_combos = list(itertools.product(*c_vals))
        selectors = [dict(zip(c_keys, combo)) for combo in all_combos]
        
        return coords, selectors

    def _process_data(self, data: List[T]) -> Union[T,xr.DataArray]:
        if self.size==1:
            return data[0]
        assert len(data) == self.size, f"Data size {len(data)} does not match expected size {self.size} based on coordinates."
        
        # make empty xr with coords
        arr = xr.DataArray(
            np.empty(self.shape, dtype=object),
            coords=self.coords,
            dims=self.dims,
        )
        # place the data into the arr
        for d in data:
            selector = tuple(self.coords[dim].index(d.metadata[dim]) for dim in self.dims)
            arr.values[selector] = d
        return arr
    
    def __iter__(self):
        for selector in self.selectors:
            v = self.values(selector)
            yield selector, self.values(selector)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the ``Dataset`` (the length of each of its dimensions)
        """
        return tuple(len(self.coords[dim]) for dim in self.dims) if self.coords else (1,)
    
    @property
    def size(self) -> int:
        """
        The number of :py:class:`Products <sika.product.Product>` in the ``Dataset`` (the product of its :py:attr:`~sika.modeling.data.Dataset.shape`).
        """
        return np.prod(self.shape)
    
    def values(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get the product or xarray of products selected by ``context``, a dictionary that specifies some or all of the coordinates at which the Product(s) of interest are stored.
        """
        if not self.coords or context is None or self.size==1:
            r = self._data
        else:
            selector = {}
            for k in self.coords:
                if k in context:
                    selector[k] = context[k]
            try:
                r = self._data.sel(**selector)
            except Exception as e:
                print(f"selector: {selector}")
                print(f"data coords: {self.coords}")
                print(f"self dims: {self.dims}")
                raise e
        try:
            return r.item()
        except (ValueError, AttributeError):
            return r
        
    # this is so goofy bc Dataset is actually a DataArray
    def as_xarray(self) -> xr.DataArray:
        """Get the underlying :py:class:`xr.DataArray`

        :return: The underlying :py:class:`xr.DataArray`. Be gentle with it!
        :rtype: :py:class:`xr.DataArray`
        """
        return self._data

    @staticmethod
    def joint_iter(*datasets: 'Dataset'):
        """Iterate over the broadcasted coordinates of *n* datasets, yielding the joint selector and a tuple of *n* :py:class:`Products <sika.product.Product>`, one from each dataset 

        :yield: selector, (prod1,prod2,...)
        :rtype: dict[str:Any], tuple(:py:class:`~sika.product.Product`)
        """
        for (x) in joint_iter_generic(*[d.as_xarray() for d in datasets]):
            yield x

def joint_iter(*datasets: Dataset):
    for (x) in joint_iter_generic(*[d.as_xarray() for d in datasets]):
        yield x


# lol yes the dataset doesnt pass the type check but it should work fine
class DataLoader(Generic[T], Provider[Dataset[T]], ABC):
    """
    A :py:class:`~sika.provider.Provider` of a :py:class:`~sika.modeling.data.Dataset` of :py:class:`Products <sika.product.Product>`. Provides an explicit entry point for data into a sampler or model pipeline. 
    """
    
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {}
    
    # overriding 'final' to make the params optional because im sick and twisted
    def __call__(self, parameters: Dict[str, Any]=None) -> Dataset[T]:
        parameters = parameters or {}
        return super().__call__(parameters)


# class ProviderDataLoader(Generic[T], DataLoader[T], ABC):
#     """An interface that allows a modeling pipeline to fill the role of a DataLoader"""
    
#     def __init__(self, dims:List[str], providers: List[Provider[T]], params:List[Dict[str,Any]], metadata: Optional[List[Dict[str,Any]]]=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dims = dims
#         self.providers = providers
#         assert len(self.providers) == len(params), "Number of providers must match number of parameter sets."
#         self.params = params
#         metadata = metadata or [{} for _ in range(len(self.providers))]
#         assert len(self.providers) == len(metadata), "If given, number of metadata sets must match number of providers."
#         self.metadata = metadata
        
#     @property
#     def previous(self):
#         return self.providers
        
#     # def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
#     #     """ Configure with config and logger. """
#     #     for provider in self.providers:
#     #         provider.configure(config, logger)
#     #     super().configure(config, logger)

#     @property
#     def provided_parameters(self) -> Dict[str, List[Any]]:
#         return {}
    
#     def _call(self, parameters: Dict[str, Any]=None) -> Dataset[T]:
#         """
#         Call the provider and return a Dataset. Given parameters will be merged with the parameters for each provider.
#         """
#         parameters = parameters or {}
#         products = []
#         for provider, param, metadata in zip(self.providers, self.params, self.metadata):
#             merged_params = {**param, **parameters}
#             product = provider(merged_params)
#             if not isinstance(product, Product):  # assume its a collection of products
#                 for p in product:
#                     if not isinstance(p, Product):
#                         raise TypeError(f"Expected Product, got {type(p)}")
#                     p.metadata.update(metadata)
#                     products.append(p)
#             else:
#                 product.metadata.update(metadata)
#                 products.append(product)
#         return Dataset(products, dims=self.dims)