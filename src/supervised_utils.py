import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from solver import value_iteration, stochastic_policy_from_value_expectation
from trajectory import Trajectory, generate_trajectory, generate_trajectories, stochastic_policy_adapter


# assumes each policy has same # of trajectories
def trajectory_list_to_xy(trajectories_list):
    num_policies = len(trajectories_list)
    n_trajectories_per_policy = len(trajectories_list[0])
    x_data = []
    for i in range(num_policies):
        x_data.extend([np.matrix(t).tolist() for t in trajectories_list[i]])
    y_data = []
    for i in range(num_policies):
        y_data.extend([i] * n_trajectories_per_policy)
    return x_data, y_data

# shuffle data
def shuffle(x_data, y_data):
    temp = list(zip(x_data,y_data))
    random.shuffle(temp) 
    x_data, y_data = zip(*temp)
    return x_data, y_data

# x to ragged tensor
def x_to_ragged(x_data):
    x_data = tf.ragged.constant(x_data)
    max_seq = int(x_data.bounding_shape()[-2])
    return x_data, max_seq

# splits data given that x_data is a ragged tensor, y_data is a np array
def train_test_split(x_data, y_data, test_prop=.20):
    test_n = int(len(y_data) * test_prop)
    x_test = x_data[:test_n, :, :]
    y_test = y_data[:test_n]
    x_train = x_data[test_n:,:,:]
    y_train = y_data[test_n:]
    return x_train, y_train, x_test, y_test

def compute_class_accuracy(y_test, y_predicted):
    # get confusion matrix
    cm = confusion_matrix(y_test, y_predicted)

    # normalize diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # accuracy by class 
    return cm.diagonal()

def sigmoid(X):
    return 1/(1+np.exp(-X))

def get_fixed_policies(success_prob = .9):
    policies_fixed = []
    for i in range(3):
        def policy(state, action = i):
            if success_prob >= np.random.uniform():
                return action
            else:
                return np.random.choice(3)
        policies_fixed.append(policy)
    return policies_fixed

def get_expert_policy(world,  discount = .7, weighting = lambda x: x):
    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[-1] = 1.0

    value = value_iteration(world.p_transition, reward, discount)
    policy = stochastic_policy_from_value_expectation(world, value)
    policy_exec = stochastic_policy_adapter(policy)
    return policy_exec

def generate_trajectories_from_policy_list(world, policy_list, n_trajectories_per_policy = 100):
    start = [0]
    terminal = [world.size - 1]

    trajectories_list = []
    for i, policy in enumerate(policy_list):
        trajectories = list(generate_trajectories(n_trajectories_per_policy, world, policy_list[i], start, terminal))
        trajectories = [t._t for t in trajectories]
        trajectories_list.append(trajectories)
    return trajectories_list