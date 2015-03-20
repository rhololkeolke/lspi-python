# -*- coding: utf-8 -*-
from unittest import TestCase

from lspi.policy import Policy
from lspi.basis_functions import FakeBasis, OneDimensionalPolynomialBasis
import numpy as np
from copy import copy

class TestPolicy(TestCase):

    def create_policy(self, *args, **kwargs):
        return Policy(FakeBasis(5), *args, **kwargs)

    @staticmethod
    def list_has_duplicates(list, num_places=4):
        # verify that there are no duplicate q values.
        # round the q_values so that there are not small floating point
        # inconsistencies that lead to no duplicates being detected
        # Then make a set of the list. If there are no duplicates then the
        # cardinality of the set will match the length of the list
        rounded_list = map(lambda x: round(x, 4), list)
        return len(set(rounded_list)) < len(list)

    def setUp(self):
        self.poly_policy = Policy(OneDimensionalPolynomialBasis(1, 2),
                                  weights=np.array([1., 1, 2, 2]))
        self.state = np.array([-3.])
        self.tie_weights = np.ones((4,))

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
        self.assertEqual(orig_policy.num_actions,
                         policy_copy.num_actions)
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

    def test_best_action_no_ties(self):

        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertFalse(TestPolicy.list_has_duplicates(q_values))

        best_action = self.poly_policy.best_action(self.state)
        self.assertEqual(best_action, 0)

    def test_best_action_with_ties_first_wins(self):
        self.poly_policy.weights = self.tie_weights
        self.poly_policy.tie_breaking_strategy = \
            Policy.TieBreakingStrategy.FirstWins

        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertTrue(TestPolicy.list_has_duplicates(q_values))

        best_action = self.poly_policy.best_action(self.state)
        self.assertEqual(best_action, 0)

    def test_best_action_with_ties_last_wins(self):
        self.poly_policy.weights = self.tie_weights
        self.poly_policy.tie_breaking_strategy = \
            Policy.TieBreakingStrategy.LastWins

        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertTrue(TestPolicy.list_has_duplicates(q_values))

        best_action = self.poly_policy.best_action(self.state)
        self.assertEqual(best_action, 1)

    def test_best_action_with_ties_random_wins(self):
        self.poly_policy.weights = self.tie_weights
        self.poly_policy.tie_breaking_strategy = \
            Policy.TieBreakingStrategy.RandomWins

        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertTrue(TestPolicy.list_has_duplicates(q_values))

        # select the best action num_times times
        num_times = 10
        best_actions = [self.poly_policy.best_action(self.state)
                        for i in range(num_times)]

        # This test will fail if all of the actions selected either action 0
        # or action 1. When all action 0 is selected the sum will be
        # equal to 0. When all action 1 is taken the sum will be equal to
        # num_times
        self.assertLess(int(sum(best_actions)), num_times)
        self.assertNotEqual(int(sum(best_actions)), 0)

    def test_best_action_mismatched_state_dimensions(self):
        with self.assertRaises(ValueError):
            self.poly_policy.best_action(np.ones((2,)))

    def test_select_action_random(self):
        # first verify there are no ties
        # this way we know the tie breaking strategy isn't introducing
        # the randomness
        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertFalse(TestPolicy.list_has_duplicates(q_values))

        self.poly_policy.explore = 1.0
        self.poly_policy.tie_breaking_strategy = \
            Policy.TieBreakingStrategy.FirstWins

        # this is set up to evaluate to no tie
        num_times = 10
        best_actions = [self.poly_policy.select_action(self.state)
                        for i in range(num_times)]

        self.assertNotEqual(sum(best_actions), 0)
        self.assertNotEqual(sum(best_actions), num_times)

    def test_select_action_deterministic(self):
        # first verify there are no ties
        # this way we know the tie breaking strategy isn't introducing
        # the randomness
        q_values = [self.poly_policy.calc_q_value(self.state, action)
            for action in range(self.poly_policy.num_actions)]

        self.assertFalse(TestPolicy.list_has_duplicates(q_values))

        self.poly_policy.explore = 0.0
        self.poly_policy.tie_breaking_strategy = \
            Policy.TieBreakingStrategy.FirstWins

        # this is set up to evaluate to no tie
        num_times = 10
        best_actions = [self.poly_policy.select_action(self.state)
                        for i in range(num_times)]
        self.assertEqual(sum(best_actions), 0)

    def test_select_action_mismatched_state_dimensions(self):
        with self.assertRaises(ValueError):
            self.poly_policy.select_action(np.ones((2,)))

    def test_num_actions_getter(self):
        self.assertEqual(self.poly_policy.num_actions,
                         self.poly_policy.basis.num_actions)

        self.poly_policy.basis.num_actions = 10

        self.assertEqual(self.poly_policy.num_actions,
                         self.poly_policy.basis.num_actions)

    def test_num_actions_setter(self):
        self.assertEqual(self.poly_policy.num_actions,
                         self.poly_policy.basis.num_actions)

        self.poly_policy.num_actions = 10

        self.assertEqual(self.poly_policy.num_actions,
                         self.poly_policy.basis.num_actions)