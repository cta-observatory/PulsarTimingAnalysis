
.. _installation:


************
Installation
************

``ptiming_ana`` requires Python |python_requires|. Its main dependencies are
`lstchain <https://cta-observatory.github.io/cta-lstchain/>`_,
`Gammapy <https://gammapy.org>`_ and
`PINT <https://nanograv-pint.readthedocs.io>`_.

We recommend using the ``mamba`` package manager (see info
`here <https://mamba.readthedocs.io>`_) to create an environment with the
dependencies, using the ``environment.yml`` file in the repository:

.. code-block:: console

   $ git clone https://github.com/cta-observatory/PulsarTimingAnalysis.git
   $ cd PulsarTimingAnalysis
   $ mamba env create -f environment.yml
   $ mamba activate pulsar-lst1

Then install ``ptiming_ana`` in the environment:

.. code-block:: console

   $ pip install .

On Linux, the package and all its dependencies can also be installed only with
``pip`` in any Python environment. On macOS the conda environment is needed,
since some of the dependencies of ``lstchain`` are not available in PyPI for
that platform.

.. note::

   PINT runs with reduced numerical precision on platforms without extended
   precision floating-point numbers (e.g. Apple Silicon). Linux x86-64 is
   recommended to compute the pulsar phases of the final results.


Development version
===================

To modify the code, install the package in editable mode together with the
tools needed to run the tests and build the documentation:

.. code-block:: console

   $ pip install -e ".[all]"
   $ pre-commit install


Testing the installation
========================

.. code-block:: console

   $ pip install ".[test]"
   $ pytest

The tests that need to download data (clock corrections and Solar System
ephemerides used by PINT) are skipped by default. They are run with
``pytest --run-network``.
