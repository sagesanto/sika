.. _pipelines:

=========
Pipelines
=========
Components for building modeling pipelines.

:ref:`Products <product_api>`
-----------------------------
Products flow through pipelines and represent a model or observation. Additionally, they carry read and writable metadata.

:ref:`Providers <provider_api>`
-------------------------------
Providers are :py:class:`Tasks <sika.task.Task>` that take parameters and products from other tasks in the pipeline and use them to produce another Product to be passed along the pipeline.

:ref:`Tasks <tasks_api>`
------------------------
The base class of :py:class:`~sika.provider.Provider`


.. toctree::
    :hidden:
    :titlesonly:

    ./product_api.rst
    ./provider_api.rst
    ./task_api.rst
