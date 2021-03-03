# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:14:42 2021

@author: grego
"""


from snake_ladder import SnakeLadderWorld
import maxent
import trajectory 
import optimizer 

import numpy as np


SHORTCUT_DENSITY= 0.1
BOARD_SIZE = 20
NUM_EXACT_TRAJECTORIES = 200
NUM_GAME_SAMPLES = 5
NUM_PLAYER_TESTS = 100



def execute_maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = optimizer.Constant(0.1)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = optimizer.ExpSga(lr=optimizer.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = maxent.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def execute_maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = world.state_features()

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = optimizer.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = optimizer.ExpSga(lr=optimizer.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = maxent.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward




world = SnakeLadderWorld(size=BOARD_SIZE, shortcut_density=SHORTCUT_DENSITY)

policy_1 = world.oso_policy
policy_2 = world._smartish_policy

policies = [policy_1, policy_2]



start = [0]
terminal = [BOARD_SIZE - 1]

trajectories_exact_1 = \
    list(trajectory.generate_trajectories(NUM_EXACT_TRAJECTORIES, world, policy_1, start, terminal))
    
trajectories_exact_2 = \
    list(trajectory.generate_trajectories(NUM_EXACT_TRAJECTORIES, world, policy_2, start, terminal))
    
    
np.seterr(all='raise')    
    
reward_exact_1 = execute_maxent(world, terminal, trajectories_exact_1)
reward_exact_2 = execute_maxent(world, terminal, trajectories_exact_2)

results = []
for i in range(NUM_PLAYER_TESTS):
    
    true_policy_num = 1 if np.random.uniform(0, 1) > 0.5 else 0
    
    trajectories_observed = \
    list(trajectory.generate_trajectories(NUM_GAME_SAMPLES, world, 
                                          policies[true_policy_num], start, terminal))

    
    reward_observed = execute_maxent(world, terminal, trajectories_observed)
    
    dist_0 = np.linalg.norm(reward_observed - reward_exact_1)
    dist_1 = np.linalg.norm(reward_observed - reward_exact_2)
    
    predicted_policy_num =  0 if dist_0 < dist_1 else 1
    
    results.append((true_policy_num, predicted_policy_num))
    





















