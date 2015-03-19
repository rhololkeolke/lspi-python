# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc


class BasisFunction(object):

    """ABC for basis functions used by LSPI Policies.

    A basis function is a function that takes in a state vector and an action
    index and returns a vector of features. The resulting feature vector is
    referred to as Phi in the LSPI paper (pg 9 of the PDF referenced in this
    package's documentation). The Phi vector is dotted with the weight vector
    of the Policy to calculate the Q-value.

    The dimensions of the state vector are usually smaller than the dimensions
    of the Phi vector. However, the dimensions of the Phi vector are usually
    much smaller than the dimensions of an exact representation of the state
    which leads to significant savings when computing and storing a policy.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def size(self):
        """Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the Phi vector. (Referred to as k in the paper).

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        """Calculate the Phi matrix for the given state-action pair.

        The way this value is calculated depends entirely on the concrete
        implementation of BasisFunction.

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.


        Returns
        -------
        numpy.array
            The Phi vector. Used by Policy to compute Q-value.

        """
        pass  # pragma: no cover
