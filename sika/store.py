__all__ = ["FileWritable"]

from abc import ABC, abstractmethod
from os.path import join, exists
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypeVar, Generic, TypeVarTuple, Callable, Union

T = TypeVar('T', bound='FileWritable')
class FileWritable(ABC):
    """ An object that can be saved to and loaded from a file."""

    @classmethod
    @abstractmethod
    def filename(cls, *args) -> str:
        """ Abstract method for creating a filename from the provided arguments. """
        pass

    @property
    @abstractmethod
    def save_params(self) -> Any:
        """ Abstract property detailing the subset of this object's parameters that should be used when generating a filename """
        pass

    def save(self, save_dir: str, **kwargs) -> None:
        """Save this object in the directory save_dir. The name of the file is derived from :py:attr:`~.save_params` and :py:method:`~.filename` 

        :param save_dir: the directory that this object should be saved into. must be writable
        :type save_dir: str
        """
        path = join(save_dir, self.filename(self.save_params))
        self._save(path, **kwargs)

    @abstractmethod
    def _save(self, path: str, **kwargs) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls: type[T], path: str, **kwargs) -> T:
        """ Abstract method for loading an object of type ``T`` from the given filepath """
        pass