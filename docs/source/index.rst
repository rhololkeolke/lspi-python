.. LSPI Python documentation master file, created by
   sphinx-quickstart on Thu Mar 19 04:02:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LSPI Python's documentation!
=======================================

Contents:

.. toctree::
   :maxdepth: 2

   autodoc/modules


This is a Python implementation of the Least Squares Policy Iteration (LSPI) reinforcement learning algorithm.
For more information on the algorithm please refer to the paper

| “Least-Squares Policy Iteration.”
| Lagoudakis, Michail G., and Ronald Parr.
| Journal of Machine Learning Research 4, 2003.
| `<https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf>`_

You can also visit their website where more information and a Matlab version is provided.

`<http://www.cs.duke.edu/research/AI/LSPI/>`_

Overview
--------

When using this library the first thing you must do is collect a set of samples
for LSPI to learn from. Each sample should be an instance of the :class:`Sample`.
These samples are then passed into the :func:`lspi.learn` method. This method
takes in the list of samples, a policy, and a solver. The :class:`Policy` class
provided should not need to be modified. The learn method then continuously
calls the solver on the data samples and policy until the policy converges. Once
the policy has converged the agent can use the policy to find the best action
in every state and execute it.

The Policy class contains the basis function approximation and its associated weights.
Weights can be specified or if left unspecified, randomly generated. The policy
also contains the probability of doing an exploration action, and the discount factor.
The Policy class should not need to be modified when using this library.

The basis functions all inherit from the abstract base class :class:`lspi.basis_functions.BasisFunction`. This
class provides the minimum interface for a basis function. Instances of this class
may contain specialized fields and methods. There are a handful of basis function
classes provided in this package including: :class:`lspi.basis_functions.FakeBasis`, :class:`lspi.basis_functions.ExactBasis`,
:class:`lspi.basis_functions.OneDimensionalPolynomialBasis`, and :class:`lspi.basis_functions.RadialBasisFunction`. See
each class for its respective construction parameters and how the basis is calculated.
You can also implement your own BasisFunctions by inheriting from the BasisFunction class and implementing
all of the abstract methods.

As mentioned the learn method takes in a Solver instance. This instance is responsible
for performing a single policy update step given the current policy and the samples being
learned from. Currently the only implemented Solver is the :class:`lspi.solvers.LSTDQSolver` which implements
the algorithm from Figure 5 of the LSPI paper. There are other variants in the LSPI paper that could
also be implemented. Additionally if a different matrix solving style is needed (e.g. sparse matrix solver)
then a new solver can be implemented. To implement a new Solver simply create a
class that inherits from the :class:`lspi.solvers.Solver` class. You must implement all of the abstract methods.

For testing and demonstration purposes the simple ChainDomain from the LSPI paper is included
in the :mod:`lspi.domains` module. If you wish to implement other domains it is recommended
that you inherit from the :class:`lspi.domains.Domain` class and implement the abstract methods.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
