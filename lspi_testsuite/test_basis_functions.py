# -*- coding: utf-8 -*-
"""Contains unit tests for the basis function module."""
from unittest import TestCase

from lspi.basis_functions import BasisFunction, OneDimensionalPolynomialBasis
import numpy as np

class TestBasisFunction(TestCase):
    def test_require_size_method(self):
        """Test BasisFunction implementation requires size method."""

        class MissingSizeBasis(BasisFunction):
            def evaluate(self, state, action):
                pass

        with self.assertRaises(TypeError):
            MissingSizeBasis()

    def test_require_evaluate_method(self):
        """Test BasisFunction implementation requires evaluate method."""

        class MissingEvaluateBasis(BasisFunction):
            def size(self):
                pass

        with self.assertRaises(TypeError):
            MissingEvaluateBasis()

    def test_works_with_both_methods_implemented(self):
        """Test BasisFunction implemention works when all methods defined."""

        class ShouldWorkBasis(BasisFunction):
            def size(self):
                pass

            def evaluate(self, state, action):
                pass

        ShouldWorkBasis()

class TestOneDimensionalPolynomialBasis(TestCase):
    def setUp(self):

        self.basis = OneDimensionalPolynomialBasis(2, 2)

    def test_specify_degree(self):

        self.assertEqual(self.basis.degree, 2)

    def test_specify_actions(self):

        self.assertEqual(self.basis.num_actions, 2)

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

        phi = self.basis.evaluate(np.array([2]), 0)
        self.assertEqual(phi.shape, (6, ))
        np.testing.assert_array_almost_equal(phi,
                                             np.array([1., 2., 4., 0., 0., 0.]))

    def test_evaluate_out_of_bounds_action(self):

        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([2]), 2)

        with self.assertRaises(IndexError):
            self.basis.evaluate(np.array([2]), -1)

    def test_evaluate_incorrect_state_dimensions(self):

        with self.assertRaises(ValueError):
            self.basis.evaluate(np.array([2, 3]), 0)
