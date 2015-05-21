# -*- coding: utf-8 -*-
"""Contains test for the lspi learn method."""
from unittest import TestCase

import lspi
from lspi.solvers import Solver
from lspi.policy import Policy
from lspi.basis_functions import FakeBasis
import numpy as np

class SolverStub(Solver):
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.num_calls = 0

    def count_calls(self):
        self.num_calls += 1
        if self.num_calls > self.max_iterations:
            raise RuntimeError(("%s was called more than the specified " +
                               "max_iterations: %d") %
                               (self.__class__.__name__, self.max_iterations))

class MaxIterationsSolverStub(SolverStub):
    def __init__(self, max_iterations=10):
        super(MaxIterationsSolverStub, self).__init__(max_iterations)

    def solve(self, data, policy):
        super(MaxIterationsSolverStub, self).count_calls()
        return policy.weights + 100

class EpsilonSolverStub(SolverStub):
    def __init__(self, epsilon, max_iterations=10):
        super(EpsilonSolverStub, self).__init__(max_iterations)
        self.epsilon = epsilon

    def solve(self, data, policy):
        super(EpsilonSolverStub, self).count_calls()
        return policy.weights.copy()

class WeightSolverStub(SolverStub):

    def __init__(self, weights, max_iterations=10):
        super(WeightSolverStub, self).__init__(max_iterations)
        self.weights = weights

    def solve(self, data, policy):
        super(WeightSolverStub, self).count_calls()
        return self.weights

class SolverParamStub(SolverStub):
    def __init__(self, data, policy, max_iterations=10):
        super(SolverParamStub, self).__init__(max_iterations)
        self.data = data
        self.policy = policy

    def solve(self, data, policy):
        super(SolverParamStub, self).count_calls()

        assert id(self.data) == id(data)
        np.testing.assert_array_almost_equal_nulp(self.policy.weights,
                                                  policy.weights)
        assert policy.discount == self.policy.discount
        assert policy.explore == self.policy.explore
        assert policy.basis == self.policy.basis
        assert policy.tie_breaking_strategy == \
               self.policy.tie_breaking_strategy

        return self.policy.weights


class TestLearnFunction(TestCase):
    def test_max_iterations_stopping_condition(self):
        """Test if learning stops when max_iterations is reached."""

        with self.assertRaises(ValueError):
            lspi.learn(None, None, None, max_iterations=0)

        max_iterations_solver = MaxIterationsSolverStub()

        lspi.learn(None,
                   Policy(FakeBasis(1)),
                   max_iterations_solver,
                   epsilon=10**-200,
                   max_iterations=10)

        self.assertEqual(max_iterations_solver.num_calls, 10)

    def test_epsilon_stopping_condition(self):
        """Test if learning stops when distance is less than epsilon."""

        with self.assertRaises(ValueError):
            lspi.learn(None, None, None, epsilon=0)

        epsilon_solver = EpsilonSolverStub(10**-21)

        lspi.learn(None,
                   Policy(FakeBasis(1)),
                   epsilon_solver,
                   epsilon=10**-20,
                   max_iterations=1000)

        self.assertEqual(epsilon_solver.num_calls, 1)

    def test_returns_policy_with_new_weights(self):
        """Test if the weights in the new policy differ and are not the same underlying numpy vector."""

        initial_policy = Policy(FakeBasis(1))

        weight_solver = WeightSolverStub(initial_policy.weights)

        new_policy = lspi.learn(None,
                                initial_policy,
                                weight_solver,
                                max_iterations=1)

        self.assertEqual(weight_solver.num_calls, 1)
        self.assertFalse(np.may_share_memory(initial_policy.weights,
                                             new_policy))
        self.assertNotEquals(id(initial_policy), id(new_policy))
        np.testing.assert_array_almost_equal(new_policy.weights,
                                             weight_solver.weights)

    def test_solver_uses_policy_and_data(self):
        """Test that the solver is passed the data and policy."""

        data = [10]
        initial_policy = Policy(FakeBasis(1))

        solver_stub = SolverParamStub(data, initial_policy)

        lspi.learn(solver_stub.data,
                   solver_stub.policy,
                   solver_stub,
                   max_iterations=1)