# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc
import numpy as np


class BasisFunction(object):

    """ABC for basis functions used by LSPI Policies.

    A basis function is a function that takes in a state vector and an action
    index and returns a vector of features. The resulting feature vector is
    referred to as :math:`\phi` in the LSPI paper (pg 9 of the PDF referenced
    in this package's documentation). The :math:`\phi` vector is dotted with
    the weight vector of the Policy to calculate the Q-value.

    The dimensions of the state vector are usually smaller than the dimensions
    of the :math:`\phi` vector. However, the dimensions of the :math:`\phi`
    vector are usually much smaller than the dimensions of an exact
    representation of the state which leads to significant savings when
    computing and storing a policy.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def size(self):
        """Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        """Calculate the :math:`\phi` matrix for the given state-action pair.

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
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        """
        pass  # pragma: no cover


class OneDimensionalPolynomialBasis(BasisFunction):
    """Polynomial features for a state with one dimension.

    Takes the value of the state and constructs a vector proportional
    to the specified degree and number of actions. The polynomial is first
    constructed as [..., 1, value, value^2, ..., value^k, ...]
    where k is the degree. The rest of the vector is 0.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    num_actions: int
        The total number of possible actions

    """

    def __init__(self, degree, num_actions):
        if degree < 0:
            raise ValueError('Degree must be >= 0')
        self.degree = degree

        if num_actions < 1:
            raise ValueError('There must be at least 1 action')
        self.num_actions = num_actions

    def size(self):
        """Calculate the size of the basis function.

        The base size will be degree + 1. This basic matrix is then
        duplicated once for every action. Therefore the size is equal to
        (degree + 1) * number of actions


        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.


        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.size()
        6

        """
        return (self.degree + 1) * self.num_actions

    def evaluate(self, state, action):
        """Calculate :math:`\phi` matrix for given state action pair

        The :math:`\phi` matrix is used to calculate the Q function for the
        given policy.

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
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.

        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.evaluate(np.array([2]), 0)
        array([ 1.,  2.,  4.,  0.,  0.,  0.])

        """

        if action < 0 or action >= self.num_actions:
            raise IndexError("Action index out of bounds")

        if state.shape != (1, ):
            raise ValueError("This class only supports one dimensional states")

        phi = np.zeros((self.size(), ))

        offset = self.size()*action

        value = state[0]

        phi[offset:offset + self.degree + 1] = \
            np.array([pow(value, i) for i in range(self.degree+1)])

        return phi
