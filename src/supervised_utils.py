import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import confusion_matrix


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