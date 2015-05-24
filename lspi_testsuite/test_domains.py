# -*- coding: utf-8 -*-
"""Contains unit tests for the included domains."""
from unittest import TestCase

from lspi.domains import ChainDomain
import numpy as np

class TestChainDomain(TestCase):
    def setUp(self):
        self.num_states = 20
        self.reward_location = ChainDomain.RewardLocation.HalfMiddles
        self.failure_probability = .3
        self.domain = ChainDomain(self.num_states,
                                  self.reward_location,
                                  self.failure_probability)

    def test_minimum_number_of_states(self):
        """Test that domain throws error is num_states < 4."""

        with self.assertRaises(ValueError):
            ChainDomain(3)

    def test_invalid_failure_probability(self):
        """Test that error is raised if failure probability is < 0 or > 1."""

        with self.assertRaises(ValueError):
            ChainDomain(failure_probability=-.1)

        with self.assertRaises(ValueError):
            ChainDomain(failure_probability=1.1)

    def test_init_parameters_are_used(self):
        """Test that init parameters are used."""

        self.assertEquals(self.domain.reward_location,
                          self.reward_location)
        self.assertEquals(self.domain.failure_probability,
                          self.failure_probability)

    def test_num_actions(self):
        """Test ChainDomain num_actions implementation."""

        self.assertEquals(self.domain.num_actions(), 2)

    def test_reset_with_no_specified_state(self):
        """Test reset with no specified state."""

        self.domain.reset() # basically test that no exception is thrown

    def test_reset_with_specified_state(self):
        """Test reset with a valid state specified."""

        new_state = np.array([0])

        self.domain.reset(new_state)

        curr_state = self.domain.current_state()
        self.assertEquals(curr_state[0], 0)

    def test_reset_with_diff_sized_state(self):
        """Test state vector with different sized state."""

        new_state = np.zeros(self.num_states+1)
        new_state[0] = 1

        with self.assertRaises(ValueError):
            self.domain.reset(new_state)

    def test_reset_with_invalid_values(self):
        """Test reset with values in state not equal to 0 or 1."""

        new_state = np.array([-1])

        with self.assertRaises(ValueError):
            self.domain.reset(new_state)

        new_state = np.array([self.num_states])

        with self.assertRaises(ValueError):
            self.domain.reset(new_state)

    def test_action_name(self):
        """Test action_name method."""

        self.assertEquals(self.domain.action_name(0), "left")
        self.assertEquals(self.domain.action_name(1), "right")

    def test_deterministic_left(self):
        """Test deterministic left action."""

        num_states = 10
        starting_state = np.array([2])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([1])

        sample = chain_domain.apply_action(0)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 0)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_deterministic_left_chain_end(self):
        """Test deterministic left action at the end of the chain."""

        num_states = 10
        starting_state = np.array([0])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = starting_state.copy()

        sample = chain_domain.apply_action(0)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 0)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_deterministic_right(self):
        """Test deterministic right action."""

        num_states = 10
        starting_state = np.array([num_states-3])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([num_states-2])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_deterministic_right_chain_end(self):
        """Test deterministic right action at the end of the chain."""

        num_states = 10
        starting_state = np.array([num_states-1])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = starting_state.copy()

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_failed_left(self):
        """Test failing left action."""

        num_states = 10
        starting_state = np.array([1])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   1)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([2])

        sample = chain_domain.apply_action(0)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 0)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_failed_right(self):
        """Test failing right action."""

        num_states = 10
        starting_state = np.array([2])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   1)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([1])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_rewards_at_ends(self):
        """Test rewards at end chain."""

        num_states = 10
        starting_state = np.array([0])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Ends,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = starting_state.copy()

        sample = chain_domain.apply_action(0)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 0)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

        starting_state = np.array([num_states-1])

        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = starting_state.copy()

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_rewards_in_middle(self):
        """Test chain with rewards in the middle."""

        num_states = 10
        starting_state = np.array([num_states/2-1])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.Middle,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([num_states/2])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

        starting_state = expected_state.copy()

        expected_state = np.array([num_states/2+1])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

        starting_state = expected_state.copy()

        expected_state = np.array([num_states/2+2])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

    def test_rewards_in_half_middles(self):
        """Test chain with rewards in the middle."""

        num_states = 10
        starting_state = np.array([num_states/4-1])

        chain_domain = ChainDomain(num_states,
                                   ChainDomain.RewardLocation.HalfMiddles,
                                   0)
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([num_states/4])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

        starting_state = np.array([3*num_states/4-1])
        chain_domain.reset(starting_state)

        np.testing.assert_array_equal(chain_domain.current_state(),
                                      starting_state)

        expected_state = np.array([3*num_states/4])

        sample = chain_domain.apply_action(1)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 1)
        self.assertEquals(sample.reward, 1)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)

        chain_domain.reset(starting_state)

        expected_state = np.array([3*num_states/4-2])

        sample = chain_domain.apply_action(0)
        np.testing.assert_array_equal(sample.state, starting_state)
        self.assertEquals(sample.action, 0)
        self.assertEquals(sample.reward, 0)
        np.testing.assert_array_equal(sample.next_state, expected_state)
        self.assertFalse(sample.absorb)
        np.testing.assert_array_equal(chain_domain.current_state(),
                                      expected_state)


    def test_out_of_bounds_action_application(self):
        """Test that error is raised when action is out of range."""

        with self.assertRaises(ValueError):
            self.domain.apply_action(-1)

        with self.assertRaises(ValueError):
            self.domain.apply_action(self.domain.num_actions())