# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:13:04 2021

@author: grego
"""


"""
Snakes and Ladders (1-Player) Markov Decision Processes (MDPs).
This implements the game given in http://ericbeaudry.uqam.ca/publications/ieee-cig-2010.pdf

Adapted from gridworld.py

The MDPs in this module are actually not complete MDPs, but rather the
sub-part of an MDP containing states, actions, and transitions (including
their probabilistic character). Reward-function and terminal-states are
supplied separately.

"""

import numpy as np
from itertools import product


class SnakeLadderWorld:
    """
    Basic deterministic grid world MDP.

    The attribute size specifies both widht and height of the world, so a
    world will have size**2 states.

    Args:
        size: Length of the board.
        num_shortcuts: Number of snakes/ladders
        seed: Seed used in random number generators of class

    Attributes:
        n_states: The number of states of this MDP.
        n_actions: The number of actions of this MDP.
        p_transition: The transition probabilities as table. The entry
            `p_transition[from, to, a]` contains the probability of
            transitioning from state `from` to state `to` via action `a`.
        size: The width and height of the world.
        actions: The actions of this world as paris, indicating the
            direction in terms of coordinates.
    """

    def __init__(self, size, shortcut_density):
        ###   ADD NUMPY RANDOM SEED AT SOME POINT?
        self.size = size
        self.shortcut_density = shortcut_density

        self.actions = [0, 1, 2]  

        # Need to decide whether to keep states with universally 0 probability
        self.n_states = self.size 
        self.n_actions = len(self.actions)
        
        self.game_board = self._generate_game()

        self.p_transition = self._transition_prob_table()
        


    def _generate_game(self):
        """
        Builds a board of Snakes and Ladders with (self.size) squares and 
        int(self.size * self.shortcut_density) Snakes/Ladders

        Returns
        -------
        game_board : np.array
            When landing on entry [i] of the game_board A[i] gives the final
            location of the player accounting for Snakes/Ladders.

        """
        
        game_board = np.arange(self.size) 
        num_links = int(self.size * self.shortcut_density)
        
        # Don't let the first/last space be a source/sink
        paired_states = np.random.choice(np.arange(1, self.size - 1),
                                         size=(num_links, 2), replace = False)
        
        for source, sink in paired_states:
            game_board[source] = sink
        
        return game_board

    
    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        s1, a = range(self.n_states), range(self.n_actions)
        for s_from, a in product(s1, a):
            table[s_from, :, a] = self._transition_prob(s_from, a)

        return table

    def _transition_prob(self, s_from, a):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            a: The action via which the target state should be reached.

        Returns:
            A vector containing the transition probability from `s_from` 
            to all states under action `a`.
        """
        transition_probs = np.zeros(self.size)
        
        if a == 0:
            transition_probs[self._protected_move(s_from, 1)] += 1
            
        
        if a == 1:
            for dist in np.arange(1, 7):
                transition_probs[self._protected_move(s_from, dist)] += 1/6
        
        if a==2:
            dice_combinations = [1,2,3,4,5,6,5,4,3,2,1]
            for dist in np.arange(2, 13):
                transition_probs[self._protected_move(s_from, dist)] \
                += dice_combinations[dist-2]/36
        
        return transition_probs
        
        
    def _protected_move(self, s_cur, offset):
        """
        Parameters
        ----------
        s_cur : TYPE
            Current state.
        offset : TYPE
            Number of spaces to move.

        Returns
        -------
        TYPE
            Returns the end state of the move accounting for end of the board
            and Snakes/Ladders.
        """
        
        if s_cur + offset >= self.size-1:
            return self.size - 1
        
        return self.game_board[s_cur + offset]
        
        

    def __repr__(self):
        return "SnakeLadderWorld(size={})".format(self.size)



    def state_features(self):
        """
        Return the feature matrix assigning each state with an individual
        feature (i.e. an identity matrix of size n_states * n_states).
    
        Rows represent individual states, columns the feature entries.
    
        Args:
            world: A GridWorld instance for which the feature-matrix should be
                computed.
    
        Returns:
            The coordinate-feature-matrix for the specified world.
        """
        
        NUM_FEATURES = 3
        features = np.zeros((self.size, NUM_FEATURES))
        features[:, 0] = np.arange(0, self.size)

        for s in range(self.size):
            features[s, 1] = self._next_snake(s)
            features[s, 2] = self._next_ladder(s)
            
        return features
    
    # Other feature ideas...length of snake and ladder links
    def _next_snake(self, s):
        
        for i in range(s, self.size):
            if self.game_board[i] < i:
                return i - s
            
        return self.size + 10
    
    def _next_ladder(self, s):
        
        for i in range(s, self.size):
            if self.game_board[i] > i:
                return i - s
            
        return self.size + 10












