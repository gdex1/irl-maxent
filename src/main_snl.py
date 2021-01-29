# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:31:31 2021

@author: grego
"""


import snake_ladder as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O

import numpy as np
import matplotlib.pyplot as plt


def setup_mdp():
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = W.SnakeLadderWorld(size=20, shortcut_density=0.1)

    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[-1] = 1.0

    # set up terminal states
    terminal = [world.size - 1]

    return world, reward, terminal




def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward


# def policy_generator_1(feature_mat, s):
    
# #####   FINISH LATER  ####
#     ## Trajectories look correct. Overflow error in MaxEnt? From degenerate policy?
    

def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()
    start = [0]

    # generate "expert" trajectories
    policy = lambda x: 1
    trajectories = list(T.generate_trajectories(5, world, policy, start, terminal))

    # return trajectories

    # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(world, terminal, trajectories)

    # maximum casal entropy reinforcement learning (non-causal)
    reward_maxcausal = maxent_causal(world, terminal, trajectories)

    print(reward_maxcausal)

if __name__ == '__main__':
    main()
    
    
    