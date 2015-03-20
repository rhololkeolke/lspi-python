# -*- coding: utf-8 -*-
from unittest import TestCase

from lspi.policy import Policy
from lspi.basis_functions import FakeBasis, OneDimensionalPolynomialBasis
import numpy as np
from copy import copy

class TestPolicy(TestCase):

    def create_policy(self, *args, **kwargs):
        return Policy(FakeBasis(5), *args, **kwargs)

    def setUp(self):
        self.poly_policy = Policy(OneDimensionalPolynomialBasis(1, 2),
                                  weights=np.array([1., 1, 2, 2]))
        self.state = np.array([-3.])

    def test_default_constructor(self):
        policy = self.create_policy()

        self.assertTrue(isinstance(policy.basis, FakeBasis))
        self.assertAlmostEqual(policy.discount, 1.0)
        self.assertAlmostEqual(policy.explore, 0.0)
        self.assertEqual(policy.weights.shape, (1,))
        self.assertEqual(policy.tie_breaking_strategy,
                         Policy.TieBreakingStrategy.RandomWins)

    def test_full_constructor(self):
        policy = self.create_policy(.5, .1, np.array([1.]),
                                    Policy.TieBreakingStrategy.FirstWins)

        self.assertTrue(isinstance(policy.basis, FakeBasis))
        self.assertAlmostEqual(policy.discount, .5)
        self.assertAlmostEqual(policy.explore, 0.1)
        np.testing.assert_array_almost_equal(policy.weights, np.array([1.]))
        self.assertEqual(policy.tie_breaking_strategy,
                         Policy.TieBreakingStrategy.FirstWins)

    def test_discount_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.create_policy(discount=-1.0)

        with self.assertRaises(ValueError):
            self.create_policy(discount=1.1)

    def test_explore_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.create_policy(explore=-.01)

        with self.assertRaises(ValueError):
            self.create_policy(explore=1.1)

    def test_weight_basis_dimensions_mismatch(self):
        with self.assertRaises(ValueError):
            self.create_policy(weights=np.arange(2))

    def test_copy(self):
        orig_policy = self.create_policy()
        policy_copy = copy(orig_policy)

        self.assertNotEqual(id(orig_policy), id(policy_copy))
        self.assertEqual(orig_policy.basis.num_actions,
                         policy_copy.basis.num_actions)
        self.assertEqual(orig_policy.discount, policy_copy.discount)
        self.assertEqual(orig_policy.explore, policy_copy.explore)
        np.testing.assert_array_almost_equal(orig_policy.weights,
                                             policy_copy.weights)

        self.assertNotEqual(id(orig_policy.weights), id(policy_copy.weights))

        # verify that changing a weight in the original doesn't affect the copy
        orig_policy.weights[0] *= -1

        # numpy doesn't have an assert if not equal method
        # so to do the inverse I'm asserting the two arrays are equal
        # and expecting the assertion to fail
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(orig_policy.weights,
                                                 policy_copy.weights)

    def test_calc_q_value_unit_weights(self):
        q_value = self.poly_policy.calc_q_value(self.state, 0)
        self.assertAlmostEqual(q_value, -2.)

    def test_calc_q_value_non_unit_weights(self):
        q_value = self.poly_policy.calc_q_value(self.state, 1)
        self.assertAlmostEqual(q_value, -4.)

    def test_calc_q_value_negative_action(self):
        with self.assertRaises(IndexError):
            self.poly_policy.calc_q_value(self.state, -1)

    def test_calc_q_value_out_of_bounds_action(self):
        with self.assertRaises(IndexError):
            self.poly_policy.calc_q_value(self.state, 2)

    def test_calc_q_value_mismatched_state_dimensions(self):
        with self.assertRaises(ValueError):
            self.poly_policy.calc_q_value(np.ones((2,)), 0)
