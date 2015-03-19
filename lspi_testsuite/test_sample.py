# -*- coding: utf-8 -*-
"""Tests for emodel.lspi.sample class."""
from unittest import TestCase

from lspi import Sample


class TestSample(TestCase):

    def setUp(self):  # flake8: noqa
        """Set the constructor parameters to test with."""
        self.state = [0, 1]
        self.action = 2
        self.reward = -1.5
        self.next_state = [1, 0]
        self.absorb = True

    def test_full_constructor(self):
        """Construct a Sample."""

        sample = Sample(self.state,
                        self.action,
                        self.reward,
                        self.next_state,
                        self.absorb)

        self.assertEqual(sample.state, self.state)
        self.assertEqual(sample.action, self.action)
        self.assertAlmostEqual(sample.reward, self.reward, 3)
        self.assertEqual(sample.next_state, self.next_state)
        self.assertEqual(sample.absorb, self.absorb)

    def test_default_constructor(self):
        """Construct a Sample with default arguments."""

        sample = Sample(self.state,
                        self.action,
                        self.reward,
                        self.next_state)

        self.assertEqual(sample.state, self.state)
        self.assertEqual(sample.action, self.action)
        self.assertAlmostEqual(sample.reward, self.reward, 3)
        self.assertEqual(sample.next_state, self.next_state)
        self.assertEqual(sample.absorb, False)