__all__ = ["Product", "FileWritableProduct", "ArrayProduct1D", "DFProduct"]

from abc import ABC, abstractmethod
from os.path import join, basename, splitext
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TypeVar, Generic, TypeVarTuple
from .store import FileWritable
import pandas as pd
import numpy as np
from copy import deepcopy

@dataclass
class Product(ABC):
    """Base class for a model or unit of data. Can store key-value information in its :py:attr:`~.parameters` and :py:attr:`~.metadata` fields. Derived from a dataclass, ``Products`` are usually produced by a :py:class:`~sika.provider.Provider` or loaded as input data (ex. from an experiment or observation)."""
    #: a record of the parameters that were used to create this product (or at least that's the intent of this attribute, but really it can be used to store any key-value information).
    parameters: Dict[str, Any]
    #: a store of key-value metadata about this product. when constructing a multi-dimensional :py:class:`~sika.modeling.data.Dataset`, keys in the metadata that line up with known coordinates are used as indices. See :py:class:`~sika.modeling.data.Dataset` for more
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self):
        return deepcopy(self)

@dataclass
class FileWritableProduct(Product, FileWritable, ABC):
    """ A product that can be saved to and loaded from a file. """
    
    @classmethod
    def filename(cls, params:Dict[str, Any]) -> str:
        """Generates a filename for the Product based on its parameters. This name should be unique for each unique set of input parameters.

        :param params: the parameters that describe the Product being stored under this filename
        :type params: Dict[str, Any]
        :return: a filename (not an entire path) under which this Product should be stored
        :rtype: str
        """
        return f"{'_'.join([f'{k}_{v}' for k,v in params.items()])}.{type(cls).__name__.lower()}"
    
    @classmethod
    def parse_filename(cls, filename: str):
        """ Extract model name and parameters from the filename. """
        filename = splitext(basename(filename))[0]  # remove file extension
        parts = filename.split('_')
        parameters = {parts[i]: parts[i + 1] for i in range(0,len(parts), 2)}
        for k,v in parameters.items():
            try:
                parameters[k]=float(v)
            except:
                pass
        return parameters
    
    @property
    def save_params(self) -> Dict[str,Any]:
        """ The subset of this Product's parameters that should be used to generate the filename. """
        return self.parameters


@dataclass
class ArrayProduct1D(FileWritableProduct):
    """ A product that can be saved to and loaded from a file in table format. """
    
    @classmethod
    @abstractmethod
    def value_keywords(cls) -> List[str]:
        """ The keyword names of the values in the array """
    
    def _save(self, path: str, **kwargs) -> None:
        arr = []        
        for k in self.value_keywords():
            arr.append(getattr(self, k))
        arr = np.array(arr)
        np.savetxt(path, arr,header=",".join(self.value_keywords()))

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved Product from a file

        :param path: path to the file
        :type path: str
        :return: the Product stored at this location
        :rtype: :py:class:`~sika.product.Product`
        """
        arr = np.loadtxt(path)
        parameters = cls.parse_filename(path)
        a = dict(zip(cls.value_keywords(),arr))
        print(a)
        return cls(parameters=parameters, **a)
        

@dataclass
class DFProduct(FileWritableProduct):
    """ A :py:class:`~sika.product.Product` that can be saved to and loaded from a file in table format. """
    
    @classmethod
    @abstractmethod
    def cols(cls) -> List[str]:
        """ The names of attributes of the :py:class:`~sika.product.Product` that will be written as columns into the table."""
        
    @classmethod
    def nullable_cols(cls) -> List[str]:
        """ The names of columns that are allowed to be nullable during construction of this product."""
        return []
    
    def _save(self, path: str, **kwargs) -> None:
        df = pd.DataFrame()
        for k in self.cols():
            col = getattr(self, k)
            if col is None:
                continue
            df[k] = getattr(self, k)
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved :py:class:`~sika.product.Product` from a table. 

        :param path: the path to the CSV file
        :type path: str
        :raises ValueError: if a column that is required for constructing this Product is not found in the provided csv file
        :return: a Product, initialized from a CSV
        :rtype: :py:class:`~sika.product.Product`
        """
        df = pd.read_csv(path)
        parameters = cls.parse_filename(path)
        kwargs = {}
        for k in cls.cols():
            if k not in df.columns:
                if k in cls.nullable_cols():
                    kwargs[k] = None
                    continue
                else:
                    raise ValueError(f"Column {k} not found in {path}. Available columns: {df.columns.tolist()}")
            kwargs[k] = df[k].tolist()
            try: 
                kwargs[k] = np.array(kwargs[k])
            except Exception as e:
                pass
        return cls(parameters=parameters, **kwargs)
        