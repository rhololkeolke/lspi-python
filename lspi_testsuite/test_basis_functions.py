# -*- coding: utf-8 -*-
"""Contains unit tests for the basis function module."""
from unittest import TestCase

from lspi.basis_functions import (BasisFunction,
    FakeBasis,
    OneDimensionalPolynomialBasis,
    RadialBasisFunction,
    ExactBasis)
import numpy as np

class TestBasisFunction(TestCase):
    def test_require_size_method(self):
        """Test BasisFunction implementation requires size method."""

        class MissingSizeBasis(BasisFunction):
            def evaluate(self, state, action):
                pass

            @property
            def num_actions(self):
                pass

        with self.assertRaises(TypeError):
            MissingSizeBasis()

    def test_require_evaluate_method(self):
        """Test BasisFunction implementation requires evaluate method."""

        class MissingEvaluateBasis(BasisFunction):
            def size(self):
                pass

            @property
            def num_actions(self):
                pass

        with self.assertRaises(TypeError):
            MissingEvaluateBasis()

    def test_require_num_actions_property(self):

        class MissingNumActionsProperty(BasisFunction):
            def size(self):
                pass

            def evaluate(self, state, action):
                pass

        with self.assertRaises(TypeError):
            MissingNumActionsProperty()

    def test_works_with_both_methods_implemented(self):
        """Test BasisFunction implemention works when all methods defined."""

        class ShouldWorkBasis(BasisFunction):

            def size(self):
                pass

            def evaluate(self, state, action):
                pass

            @property
            def num_actions(self):
                pass

        ShouldWorkBasis()

    def test_validate_num_actions(self):
        self.assertEqual(BasisFunction._validate_num_actions(6), 6)

    def test_validate_num_actions_out_of_bounds(self):
        with self.assertRaises(ValueError):
            BasisFunction._validate_num_actions(0)


class TestFakeBasis(TestCase):
    def setUp(self):
        self.basis = FakeBasis(6)

    def test_num_actions_property(self):
        self.assertEqual(self.basis.num_actions, 6)

    def test_num_actions_setter(self):
        self.basis.num_actions = 10

        self.assertEqual(self.basis.num_actions, 10)

    def test_num_actions_setter_invalid_value(self):
        with self.assertRaises(ValueError):
            self.basis.num_actions = 0

    def test_size(self):
        self.assertEqual(self.basis.size(), 1)

    def test_evaluate(self):
        np.testing.assert_array_almost_equal(self.basis.evaluate(None, 0),
                                             np.array([1.]))

    def test_evaluate_negative_action_index(self):
        with self.assertRaises(IndexError):
            self.basis.evaluate(None, -1)

    def test_evaluate_out_of_bounds_action_index(self):
        with self.assertRaises(IndexError):
            self.basis.evaluate(None, 6)

class TestOneDimensionalPolynomialBasis(TestCase):
    def setUp(self):

        self.basis = OneDimensionalPolynomialBasis(2, 2)

    def test_specify_degree(self):

        self.assertEqual(self.basis.degree, 2)

    def test_specify_actions(self):

        self.assertEqual(self.basis.num_actions, 2)

    def test_num_actions_setter(self):
        self.basis.num_actions = 10

        self.assertEqual(self.basis.num_actions, 10)

    def test_num_actions_setter_invalid_value(self):
        with self.assertRaises(ValueError):
            self.basis.num_actions = 0

    def test_out_of_bounds_degree(self):

        with self.assertRaises(ValueError):
            OneDimensionalPolynomialBasis(-1, 2)

    def test_out_of_bounds_num_action(self):

        with self.assertRaises(ValueError):
            OneDimensionalPolynomialBasis(2, 0)

        with self.assertRaises(ValueError):
            OneDimensionalPolynomialBasis(2, -1)

    def test_size(self):

        self.assertEqual(self.basis.size(), 6)

    def test_evaluate(self):

        phi = self.basis.evaluate(np.array([2]), 1)
        self.assertEqual(phi.shape, (6, ))
        np.testing.assert_array_almost_equal(phi,
                                             np.array([0., 0., 0., 1., 2., 4.]))

    def test_evaluate_out_of_bounds_action(self):

        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([2]), 2)

        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([2]), -1)

    def test_evaluate_incorrect_state_dimensions(self):

        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([2, 3]), 0)

class TestRadialBasisFunction(TestCase):
    def setUp(self):

        self.means = [-np.ones((3, )), np.zeros((3, )), np.ones((3, ))]
        self.gamma = 1
        self.num_actions = 2
        self.basis = RadialBasisFunction(self.means,
                                         self.gamma,
                                         self.num_actions)
        self.state = np.zeros((3, ))

    def test_specify_means(self):

        for mean, expected_mean in zip(self.basis.means, self.means):
            np.testing.assert_array_almost_equal(mean, expected_mean)

    def test_empty_means_list(self):
        with self.assertRaises(ValueError):
            RadialBasisFunction([], self.gamma, self.num_actions)

    def test_mismatched_mean_shapes(self):
        with self.assertRaises(ValueError):
            RadialBasisFunction([np.zeros((3, )),
                                 -np.ones((2, )),
                                 np.ones((3, ))],
                                self.gamma,
                                self.num_actions)

    def test_specify_gamma(self):
        self.assertAlmostEqual(self.gamma, self.basis.gamma)

    def test_out_of_bounds_gamma(self):
        with self.assertRaises(ValueError):
            RadialBasisFunction(self.means, 0, self.num_actions)

    def test_specify_actions(self):

        self.assertEqual(self.basis.num_actions, self.num_actions)

    def test_num_actions_setter(self):
        self.basis.num_actions = 10

        self.assertEqual(self.basis.num_actions, 10)

    def test_num_actions_setter_invalid_value(self):
        with self.assertRaises(ValueError):
            self.basis.num_actions = 0

    def test_out_of_bounds_num_action(self):

        with self.assertRaises(ValueError):
            RadialBasisFunction(self.means, self.gamma, 0)

        with self.assertRaises(ValueError):
            RadialBasisFunction(self.means, self.gamma, -1)

    def test_size(self):

        self.assertEqual(self.basis.size(), 8)

    def test_evaluate(self):

        phi = self.basis.evaluate(self.state, 0)
        self.assertEqual(phi.shape, (8, ))
        np.testing.assert_array_almost_equal(phi,
                                             np.array([1.,
                                                       0.0498,
                                                       1.,
                                                       0.0498,
                                                       0.,
                                                       0.,
                                                       0.,
                                                       0.]),
                                             4)

    def test_evaluate_out_of_bounds_action(self):

        with self.assertRaises(IndexError):
            self.basis.evaluate(self.state, 2)

        with self.assertRaises(IndexError):
            self.basis.evaluate(self.state, -1)

    def test_evaluate_incorrect_state_dimensions(self):

        with self.assertRaises(ValueError):
            self.basis.evaluate(np.zeros((2, )), 0)

class TestExactBasis(TestCase):
    def setUp(self):
        self.basis = ExactBasis([2, 3, 4], 2)

    def test_invalid_num_states(self):
        num_states = np.ones(3)
        num_states[0] = 0

        with self.assertRaises(ValueError):
            ExactBasis(num_states, 2)

    def test_num_actions_property(self):
        self.assertEqual(self.basis.num_actions, 2)

    def test_num_actions_setter(self):
        self.basis.num_actions = 3

        self.assertEqual(self.basis.num_actions, 3)

    def test_num_actions_setter_invalid_value(self):
        with self.assertRaises(ValueError):
            self.basis.num_actions = 0

    def test_size(self):
        self.assertEqual(self.basis.size(), 48)

    def test_evaluate(self):
        phi = self.basis.evaluate(np.array([0, 0, 0]), 0)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[0] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

        phi = self.basis.evaluate(np.array([1, 0, 0]), 0)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[1] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

        phi = self.basis.evaluate(np.array([0, 1, 0]), 0)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[2] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

        phi = self.basis.evaluate(np.array([0, 0, 1]), 0)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[6] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

        phi = self.basis.evaluate(np.array([0, 0, 0]), 1)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[24] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

        phi = self.basis.evaluate(np.array([1, 2, 3]), 1)
        self.assertEqual(phi.shape, (48, ))

        expected_phi = np.zeros((48, ))
        expected_phi[47] = 1

        np.testing.assert_array_almost_equal(phi, expected_phi)

    def test_evaluate_out_of_bounds_action(self):
        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([0, 0, 0]), -1)


        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([0, 0, 0]), 3)

    def test_evaluate_out_of_bounds_state(self):
        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([-1, 0, 0]), 0)


        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0, -1, 0]), 0)


        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0, 0, -1]), 0)


        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([2, 0, 0]), 0)


        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0, 3, 0]), 0)


        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0, 0, 4]), 0)

    def test_evaluate_wrong_size_state(self):
        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0]), 0)

        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([0, 0, 0, 0]), 0)