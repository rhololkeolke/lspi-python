# -*- coding: utf-8 -*-
"""Contains integration test of different learning methods on chain domain."""
from unittest import TestCase

import lspi

import numpy as np

class TestChainDomainLearning(TestCase):
    def setUp(self):
        self.domain = lspi.domains.ChainDomain()

        sampling_policy = lspi.Policy(lspi.basis_functions.FakeBasis(2), .9, 1)

        self.samples = []
        for i in range(1000):
            action = sampling_policy.select_action(self.domain.current_state())
            self.samples.append(self.domain.apply_action(action))

        self.random_policy_cum_rewards = np.sum([sample.reward
                                                 for sample in self.samples])

        self.solver = lspi.solvers.LSTDQSolver()

    def test_chain_polynomial_basis(self):

        initial_policy = lspi.Policy(
            lspi.basis_functions.OneDimensionalPolynomialBasis(3, 2),
            .9,
            0)

        learned_policy = lspi.learn(self.samples, initial_policy, self.solver)

        self.domain.reset()
        cumulative_reward = 0
        for i in range(1000):
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            cumulative_reward += sample.reward

        self.assertGreater(cumulative_reward, self.random_policy_cum_rewards)

    def test_chain_rbf_basis(self):

        initial_policy = lspi.Policy(
            lspi.basis_functions.RadialBasisFunction(
                np.array([[0], [2], [4], [6], [8]]), .5, 2),
            .9,
            0)

        learned_policy = lspi.learn(self.samples, initial_policy, self.solver)

        self.domain.reset()
        cumulative_reward = 0
        for i in range(1000):
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            cumulative_reward += sample.reward

        self.assertGreater(cumulative_reward, self.random_policy_cum_rewards)