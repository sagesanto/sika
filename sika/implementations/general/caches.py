from sika.provider import Provider, ProviderMiddleware
from sika.product import FileWritableProduct, Product
from sika.store import FileWritable
from typing import Generic, TypeVar, List, Dict, Any, Collection, TypeVarTuple, Callable
from os.path import join, exists
from os import makedirs
import logging

T = TypeVar('T', bound=FileWritableProduct, covariant=True)
class FileCache(ProviderMiddleware[T]):
    
    """A piece of middleware that caches Products that are file-writable (they're of type :py:class:`~sika.product.FileWritableProduct`) on disk for re-use."""
    
    def __init__(self, prev:Provider[T], target_cls: type[FileWritableProduct], savedir: str, save:bool=True, load:bool=True, config=None, logger=None):
        """
        :param target_cls: the class that the product should be read into after load
        :type target_cls: type[FileWritableProduct]
        :param savedir: the directory to save the products in
        :type savedir: str
        :param save: whether to save products, defaults to True
        :type save: bool, optional
        :param load: whether to load products, defaults to True
        :type load: bool, optional
        """
        super().__init__(prev, config, logger)
        self.target_cls = target_cls
        self.savedir = savedir
        self.save = save
        self.load = load
        makedirs(self.savedir, exist_ok=True)

    def product_middleware(self, model: T) -> T:
        return model

    def _call(self, params):
        filepath = join(self.savedir, self.target_cls.filename(params))
        if not exists(filepath) or not self.load:
            product = self.prev(params)
            if self.save:
                self.write_out(f"Saving {self.target_cls.__name__} to {filepath}", level=logging.DEBUG)
                product.save(self.savedir)
            return product
        else:
            self.write_out(f"Loaded {self.target_cls.__name__} from {filepath}", level=logging.DEBUG)
        return self.target_cls.load(filepath)