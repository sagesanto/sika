.. _install_guide:

============
Installation
============

The base framework can be installed with pip: ::

    pip install sika


Visualization
-------------
`Graphviz <https://graphviz.org/>`__ is required to generate pipeline :py:meth:`visualizations <sika.modeling.Task.visualize>`. See the `installation guide <https://graphviz.org/download/>`__.   


Spectroscopy
------------
To use the :ref:`spectroscopy implementation <spectroscopy>` provided in the package, either install sika with the ``spectroscopy`` option: ::
    
    pip install sika[spectroscopy]

Or simply ::

    pip install astropy petitRADTRANS

MPI
---
sika supports distributed computing using MPI through `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__. To use, install sika with the mpi option: ::
    
    pip install sika[mpi]

and, if necessary, follow the ``mpi4py`` `install instructions <https://mpi4py.readthedocs.io/en/stable/install.html>`__ if you do not already have MPI installed.

PyMultiNest
-----------
sika supports PyMultiNest as a backend for nested sampling. To use, first follow the `PyMultiNest instructions <https://johannesbuchner.github.io/PyMultiNest/install.html>`__ to install MultiNest. 