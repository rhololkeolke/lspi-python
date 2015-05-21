# -*- coding: utf-8 -*-
"""Contains main LSPI method and various LSTDQ solvers."""

import abc


class Solver(object):

    r"""ABC for LSPI solvers.

    Implementations of this class will implement the various LSTDQ algorithms
    with various linear algebra solving techniques. This solver will be used
    by the lspi.learn method. The instance will be called iteratively until
    the convergence parameters are satisified.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve(self, data, policy):
        r"""Return one-step update of the policy weights for the given data.

        Parameters
        ----------
        data:
            This is the data used by the solver. In most cases this will be
            a list of samples. But it can be anything supported by the specific
            Solver implementation's solve method.
        policy: Policy
            The current policy to find an improvement to.

        Returns
        -------
        numpy.array
            Return the new weights as determined by this method.

        """
        pass  # pragma: no cover
