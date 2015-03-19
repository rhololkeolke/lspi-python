# -*- coding: utf-8 -*-
"""Contains class representing an LSPI sample."""


class Sample(object):

    """Represents an LSPI sample tuple (s, a, r, s', absorb).

    An LSPI sample is a 5 tuple (s, a, r, s', absorb).
        s     : the state of the environment at the start of the sample.
        a     : the action performed.
        r     : the reward received.
        s'    : the resulting state.
        absorb: True if this action ended the episode. False otherwise

    This class is just a dumb data holder so the types of the different
    fields can be anything convenient for the problem domain. For states
    represented by vectors a numpy array works well.
    Actions can often be represented by an index number.
    Reward should be a double.
    absorb should be a boolean.

    """

    def __init__(self, state, action, reward, next_state, absorb=False):
        """Initialize a new sample.

        Assumes that this is a non-absorbing sample (as the vast majority
        of samples will be non-absorbing).

        Parameters
        -----------

        state : Any
            State of the environment at the start of the sample.
            (The usual type is a numpy array.)
        action: int
            Index of action that was executed.
        reward: float
            Reward received from the environment.
        next_state: Any
            State of the environment after executing the sample's action.
            (The type should match that of state.)
        absorb: bool, optional
            True if this sample ended the episode. False otherwise.
            (The default is False, which implies that this is a
            non-episode-ending sample)

        """

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.absorb = absorb

    def __repr__(self):
        return 'Sample(%s, %s, %s, %s, %s)' % (self.state,
                                               self.action,
                                               self.reward,
                                               self.next_state,
                                               self.absorb)
