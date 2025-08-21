sika documentation
===================================
sika is a generic framework for building modeling pipelines that are modular, reproducible, and scalable.

Key features:
   - Reusable components enable easy end-to-end testing and iteration of pipelines
   - Configuration-powered pipeline steps allow for flexible, reproducible workflows   
   - Pipeline runs generate and save self-documentation from code and configuration at runtime, increasing reproducibility
   - Provides built-in tools for visualizing and communicating pipeline architecture
   - Capable of robust, convenient multidimensional parameter fitting
   - Supports file-caching, parallelization and other features that allow pipelines to be applied to large, compute intensive tasks 




Installation
------------

To quickly get started with pip: ::

   pip install sika

If you want to use MPI, visualization, the spectroscopy implementation, or other optional features, please see the :ref:`install guide <install_guide>`.


:ref:`Guides <guides>`
------------------------------
Start here for a walkthrough of sika.

:ref:`API Reference <reference>`
--------------------------------
Module-, class-, and function- level documentation of the framework


.. toctree::
   :hidden:  
   
   guides
   api