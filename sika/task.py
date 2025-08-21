__all__ = ["Task", "IntermediateTask"]

from typing import TypeVar, Generic, Union, Type, Mapping, Tuple, Callable, Any
from logging import Logger
import logging
import json
import uuid
import matplotlib.pyplot as plt

from .utils import write_out, archive_config, NodeShape, NodeSpec, visualize_graph
from .config import Config

class Task:    
    """ The base class for a single step in a modeling pipeline."""
    def __init__(self, config:Union[None,Config]=None, logger: Union[None,Logger]=None, prev: Union['Task',None] = None):
        """The base class for a single step in a modeling pipeline.

        :param config: a :py:class:`~sika.config.Config` object that should be used to configure this `Task`. If not provided, this Task or a parent Task must use the :py:meth:`~sika.task.Task.configure` method to provide a :py:class:`~sika.config.Config` before executing the ``Task``. defaults to None
        :type config: Union[None,Config], optional
        :param logger: a logging.Logger object that should be used to configure this `Task`, defaults to None
        :type logger: Union[None,Logger], optional
        :param prev: a `Task` that immediately precedes this one, defaults to None
        :type prev: Union[Task,None], optional
        """
        self.config = config
        self.logger = logger
        self.prev = prev
        self.ID = uuid.uuid4().hex  # unique identifier for this task instance
        if config is not None:
            self.configure(config, logger)
            
    @property
    def previous(self):
        """ Tasks that precede this one that should be configured when this one is configured. Defines the topology of the chain of tasks."""
        return [self.prev] if self.prev is not None else []
            
    def traverse_previous(self,func:Callable[["Task",Tuple[Any, ...]],dict[Any,Any]| Any],*args:Tuple[Any, ...],maxdepth:int=-1, current_depth=0, pass_depth=False, **kwargs:Mapping[str,Any]):
        """Recursively apply a function to each of the previous tasks in the precursor tree of this task, collecting and returning its result

        Does a tree-like walk of precursors of this task, calling ``func`` on each and collecting the results in a dictionary of {result : [results of ``func`` called each preceding task in :py:attr`~task.Task.previous`]}

        :param func: the function to apply to each precursor. must take the task as its first argument. Result must be hashable.
        :param *args: additional arguments to pass to ``func``
        :param maxdepth: integer. maximum depth to traverse. any negative number has no max.
        :param **kwargs: additional keyword arguments to pass to ``func``
        :param pass_depth: if True, the current depth of the traversal will be passed to ``func`` as the keyword argument ``current_depth``. This is useful for debugging and visualization.

        :returns: a dictionary of {result of ``func``: list of results of :py:meth:`~.traverse_previous` on precursors}
        """
        if pass_depth:
            kwargs['current_depth'] = current_depth + 1
        res = {}
        if not self.previous or not maxdepth:
            return res
        for t in self.previous:
            res[func(t,*args,**kwargs)] = t.traverse_previous(func, *args, maxdepth=maxdepth-1, pass_depth=pass_depth, **kwargs)
        return res
    
    def node_spec(self) -> NodeSpec:
        """ Return a :py:class:`~.NodeSpec` for this task. Used for model structure visualization."""
        return NodeSpec(
            label=self.__class__.__name__,
            shape=NodeShape.SQUARE,
            color='#71B6F4',
            ID = self.ID
        )
        
    def visualize(self, title=None, **kwargs):
        """Create a diagram that visualizes the dependencies between this `Task` and related ones. kwargs are passed to :py:meth:`~sika.utils.visualize_graph`.

        :param title: A title for the plot, defaults to None
        :type title: str, optional
        :return: the figure and axes objects
        """
        # gather all the node_specs from the previous task
        def get_spec(t):
            # spec = t.node_spec()
            # spec.depth = current_depth
            # return spec
            return t.node_spec()

        spec_dict = self.traverse_previous(get_spec, maxdepth=-1)
        # spec_dict = self.traverse_previous(get_spec, maxdepth=-1, pass_depth=True)
        # add this task's spec
        my_spec = self.node_spec()
        my_spec.edge_weight = 1.25
        my_spec.depth = 0
        nodes = {my_spec: spec_dict}
        return visualize_graph(nodes, title=title, **kwargs)

    @property
    def name(self) -> str:
        """ The name of this Task. Usually inferred from the class name, not supplied directly."""
        return self.__class__.__name__
    
    def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
        """Configure the task with a config object and a logger. This configuration will propagate to the Tasks that precede this one (Task.previous). After configuring previous Tasks, calls :py:meth:`~Task._setup()`

        :param config: a config object that will affect this Task's behavior.
        :type config: Union[None, :py:class:`~sika.config.config.Config` ]
        :param logger: an object to use for logging. If not provided, this task's :py:meth:`~.write_out` method will print to the terminal.
        :type logger: Union[None, :py:class:`.logging.Logger` ]
        """
        self.config = config if config is not None else self.config
        self.logger = logger if logger is not None else self.logger
        for t in self.previous:
            t.configure(config,logger)
        self._setup()
        
    def _setup(self):
        """ Perform onetime setup. you may assume that by this point the config has been provided. """
        pass
    
    def write_out(self,*msg:str,level=logging.INFO):
        """Write a message to the logger if provided, or the console if no logger has been configured.

        :param level: logging level at which to send this message, defaults to logging.INFO
        :type level: a valid logging level, optional
        """
        write_out(*msg, level=level, logger=self.logger)
        
    def args_to_dict(self):
        """ Get a representation of the arguments to this `Task` that should be saved when model specifications are saved """
        args = {}
        for k, v in self.__dict__.items():
            if k in ["config", "logger", "prev"]:
                continue
            args[k] = v.to_dict() if hasattr(v, 'to_dict') else v
        return args
            
    def to_dict(self):
        """ Convert the task to a dictionary representation. """
        return {
            "name": self.__class__.__name__,
            "args": self.args_to_dict(),
            "config": archive_config(self.config) if self.config else None,
            "prev": self.prev.to_dict() if self.prev else None
        }
        
    def json(self):
        """Get a json string representation of this `Task`"""
        def default_serializer(obj):
            try:
                return obj.to_dict()
            except:
                return str(obj)
        return json.dumps(self.to_dict(), indent=4, default=default_serializer)

T = TypeVar('T', bound=Task,covariant=True)

class IntermediateTask(Generic[T],Task):
    """ A :py:class:`~Task` that requires that the immediately previous :py:class:`~Task` ``prev`` be of type ``T`` """
    def __init__(self, prev: T, config=None, logger=None):
        """A :py:class:`~Task` that requires that the immediately previous :py:class:`~Task` ``prev`` be of type ``T``

        :param prev: the previous task in the pipeline. Must be of the type ``T`` specified by this ``IntermediateTask``'s template type argument.
        :type prev: T
        :param config: a :py:class:`~sika.config.Config` object that should be used to configure this `Task`. If not provided, this Task or a parent Task must use the :py:meth:`~sika.task.Task.configure` method to provide a :py:class:`~sika.config.Config` before executing the ``Task``. defaults to None
        :type config: Union[None,Config], optional
        :param logger: a logging.Logger object that should be used to configure this `Task`, defaults to None
        :type logger: Union[None,Logger], optional
        """
        super().__init__(config, logger, prev)
    
    @property
    def name(self) -> str:
        """ The name of this Task """
        return f"{self.prev.name} -> {super().name}"