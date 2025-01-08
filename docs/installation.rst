
.. _installation:


************
Installation
************


We recommend using the ``mamba`` package manager (see info `here <https://mamba.readthedocs.io>`_).


To install ``ptiming_ana`` we recommend creating first an environment. 
You can use the ``environment.yml`` file in the repository:

.. code-block:: console

   $ mamba env create -n pulsar-lst1 -f environment.yml
   $ conda activate pulsar-lst1

or create an empty one:

.. code-block:: console

   $ mamba create -n pulsar-lst1 python=3.11
   $ conda activate pulsar-lst1

Then install ``ptiming_ana`` in the environment:

.. code-block:: console

   $ pip install ptiming_ana
