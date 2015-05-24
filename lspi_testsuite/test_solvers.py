# -*- coding: utf-8 -*-
"""Contains tests for the various solvers."""
from unittest import TestCase

from lspi.basis_functions import ExactBasis
from lspi.policy import Policy
from lspi.sample import Sample
from lspi.solvers import LSTDQSolver

import numpy as np

class TestLSTDQSolver(TestCase):
    def setUp(self):
        self.data = [Sample(np.array([0]), 0, 1, np.array([0])),
                     Sample(np.array([1]), 0, -1, np.array([1]))]

        self.basis = ExactBasis([2], 1)
        self.policy = Policy(self.basis,
                             .9,
                             0,
                             np.zeros((2, )),
                             Policy.TieBreakingStrategy.FirstWins)

    def test_precondition_value_set(self):
        """Test that precondition value is saved to solver."""
        precondition_value = .3
        solver = LSTDQSolver(precondition_value)
        self.assertEqual(solver.precondition_value, precondition_value)

    def test_solve_method_full_rank_matrix(self):
        """Test that the solver works."""

        solver = LSTDQSolver(precondition_value=0)

        weights = solver.solve(self.data, self.policy)

        expected_weights = np.array([10, -10])

        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_solve_method_singular_matrix(self):
        """Test with singular matrix and no precondition."""

        solver = LSTDQSolver(precondition_value=0)

        weights = solver.solve(self.data[:-1], self.policy)

        expected_weights = np.array([10, 0])

        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_solve_method_singular_matrix_with_preconditiong(self):
        """Test with singluar matrix and preconditioning."""

        solver = LSTDQSolver(precondition_value=.1)

        weights = solver.solve(self.data[:-1], self.policy)

        expected_weights = np.array([5, 0])

        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_solve_method_with_absorbing_sample(self):
        """Test with absorbing sample."""
        solver = LSTDQSolver(precondition_value=0)

        self.data[0].absorb = True
        weights = solver.solve(self.data, self.policy)

        expected_weights = np.array([1, -10])

        np.testing.assert_array_almost_equal(weights, expected_weights)