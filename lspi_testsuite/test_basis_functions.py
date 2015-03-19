# -*- coding: utf-8 -*-
"""Contains unit tests for the basis function module."""
from unittest import TestCase

from lspi.basis_functions import BasisFunction


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
