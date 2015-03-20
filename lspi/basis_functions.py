# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc

import numpy as np


class BasisFunction(object):

    r"""ABC for basis functions used by LSPI Policies.

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
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix for the given state-action pair.

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

    Raises
    ------
    ValueError
        If degree is less than 0
    ValueError
        If num_actions is less than 1

    """

    def __init__(self, degree, num_actions):
        """Initialize polynomial basis function."""
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
        r"""Calculate :math:`\phi` matrix for given state action pair.

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
            raise IndexError('Action index out of bounds')

        if state.shape != (1, ):
            raise ValueError('This class only supports one dimensional states')

        phi = np.zeros((self.size(), ))

        offset = (self.size()/self.num_actions)*action

        value = state[0]

        phi[offset:offset + self.degree + 1] = \
            np.array([pow(value, i) for i in range(self.degree+1)])

        return phi


class RadialBasisFunction(BasisFunction):

    r"""Gaussian Multidimensional Radial Basis Function (RBF).

    Given a set of k means :math:`(\mu_1 , \ldots, \mu_k)` produce a feature
    vector :math:`(1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
    e^{-\gamma || s - \mu_k ||^2})` where `s` is the state vector and
    :math:`\gamma` is a free parameter. This vector will be padded with
    0's on both sides proportional to the number of possible actions
    specified.

    Parameters
    ----------
    means: list(numpy.array)
        List of numpy arrays representing :math:`(\mu_1, \ldots, \mu_k)`.
        Each :math:`\mu` is a numpy array with dimensions matching the state
        vector this basis function will be used with. If the dimensions of each
        vector are not equal than an exception will be raised. If no means are
        specified then a ValueError will be raised
    gamma: float
        Free parameter which controls the size/spread of the Gaussian "bumps".
        This parameter is best selected via tuning through cross validation.
        gamma must be > 0.
    num_actions: int
        Number of actions. Must be in range [1, :math:`\infty`] otherwise
        an exception will be raised.

    Raises
    ------
    ValueError
        If means list is empty
    ValueError
        If dimensions of each mean vector do not match.
    ValueError
        If gamma is <= 0.
    ValueError
        If num_actions is less than 1.

    Note
    ----

    The numpy arrays specifying the means are not copied.

    """

    def __init__(self, means, gamma, num_actions):
        """Initialize RBF instance."""
        if len(means) == 0:
            raise ValueError('You must specify at least one mean')

        if reduce(RadialBasisFunction.__check_mean_size, means) is None:
            raise ValueError('All mean vectors must have the same dimensions')

        self.means = means

        if gamma <= 0:
            raise ValueError('gamma must be > 0')

        self.gamma = gamma

        if num_actions < 1:
            raise ValueError('num_actions must be > 0')

        self.num_actions = num_actions

    @staticmethod
    def __check_mean_size(left, right):
        """Apply f if the value is not None.

        This method is meant to be used with reduce. It will return either the
        right most numpy array or None if any of the array's had
        differing sizes. I wanted to use a Maybe monad here,
        but Python doesn't support that out of the box.

        Return
        ------
        None or numpy.array
            None values will propogate through the reduce automatically.

        """
        if left is None or right is None:
            return None
        else:
            if left.shape != right.shape:
                return None
        return right

    def size(self):
        r"""Calculate size of the :math:`\phi` matrix.

        The size is equal to the number of means + 1 times the number of
        number actions.

        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.

        """
        return (len(self.means) + 1) * self.num_actions

    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix.

        Matrix will have the following form:

        :math:`[\cdots, 1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
        e^{-\gamma || s - \mu_k ||^2}, \cdots]`

        where the matrix will be padded with 0's on either side depending
        on the specified action index and the number of possible actions.

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

        """
        if action < 0 or action >= self.num_actions:
            raise IndexError('Action index out of bounds')

        if state.shape != self.means[0].shape:
            raise ValueError('Dimensions of state must match '
                             'dimensions of means')

        phi = np.zeros((self.size(), ))
        offset = self.size()*action

        rbf = [RadialBasisFunction.__calc_basis_component(state,
                                                          mean,
                                                          self.gamma)
               for mean in self.means]
        phi[offset] = 1.
        phi[offset+1:offset+1+len(rbf)] = rbf

        return phi

    @staticmethod
    def __calc_basis_component(state, mean, gamma):
        mean_diff = state - mean
        return np.exp(-gamma*np.sum(mean_diff*mean_diff))
