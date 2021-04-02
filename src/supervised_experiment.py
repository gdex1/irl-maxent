
import numpy as np
from lstm_model import LstmModel
from supervised_utils import *
import datetime
import multiprocessing
import signal
from snake_ladder import SnakeLadderWorld
import os
import pickle

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)



def experiment_train_test(world, n_train_samples, n_test_samples, policies):
    n_train_samples_per_policy = int(n_train_samples / len(policies))
    n_test_samples_per_policy = int(n_test_samples / len(policies))

    # generate training trajectoreies and testing trajectories
    train_trajectories = generate_trajectories_from_policy_list(world, policies, n_trajectories_per_policy= n_train_samples_per_policy)
    test_trajectories = generate_trajectories_from_policy_list(world, policies, n_trajectories_per_policy= n_test_samples_per_policy)

    # convert to x,y data form
    x_train, y_train = trajectory_list_to_xy(train_trajectories)
    x_test, y_test = trajectory_list_to_xy(test_trajectories)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # convert data to ragged tensors / numpy array
    x_train, max_seq_train = x_to_ragged(x_train)
    x_test, max_seq_test = x_to_ragged(x_test)
    max_seq = max(max_seq_train, max_seq_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    lstm_model = LstmModel(max_trajectory_len=max_seq, num_features=3, num_outputs=len(policies))

    # train with large number of epochs, but with early stopping
    lstm_model.train(x_train, y_train, x_test, y_test, epochs = 500, 
                     batch_size=int(n_train_samples_per_policy / 10), early_stopping=True, patience=5)

    y_predicted = lstm_model.predict_classes(x_test)
    class_accuracy = compute_class_accuracy(y_test, y_predicted)
    return class_accuracy
  

# make a version w/ k-fold cross validation

if __name__ == "__main__":
    # number of trials for each num_trajectories
    n_trials = 10

    # test with one world
    # define some consants
    world_size = 30
    shortcut_density = 0.1
  
    # create our worlds
    n_worlds = 10
    worlds = []
    for i in range(n_worlds):
        world = SnakeLadderWorld(size=world_size, shortcut_density=shortcut_density)
        worlds.append(world)
        

    # create our policies   
    fixed_policies = get_fixed_policies()
    expert_policy = get_expert_policy(world)

    # create list of policies
    policies = []
    #policies = fixed_policies
    policies.append(expert_policy) # add expert policy to list
    policies.append(world._smartish_policy)
    n_policies = len(policies)

    n_test_samples = 1000
    def experiment_train_test_wrapper(n_train_samples, world):
        return experiment_train_test(world, n_train_samples, n_test_samples, policies)
    
    # create list of n to try
    num_list = [20, 50, 100, 200, 400, 800, 1000]


    # create 1d arg list
    arg_list = []
    for num_trajectories in num_list:
        for world in worlds:
            for i in range(n_trials):
                arg_list.append((num_trajectories, world))

    num_workers = 6 # number of physical cores
    pool = multiprocessing.Pool(num_workers, init_worker)

    class_accuracies = pool.starmap(experiment_train_test_wrapper, arg_list)

    print(class_accuracies)
    # save to numpy file
    class_accuracies = np.vstack(class_accuracies)
    description = 'binary-expert-smart'
    file_name = f'{description}_policies_{n_policies}_worlds_{n_worlds}_trials_{n_trials}_size_{world_size}_density_{shortcut_density}'
    #np.save(os.path.join('experiments', file_name), class_accuracies)

    #np.split(array, len(num_list),axis=0)
    # convert to dictionary
    accuracy_dictionary = {}
    for index, arr in enumerate(np.split(class_accuracies, len(num_list),axis=0)):
        accuracy_dictionary[num_list[index]] = arr
    # write to file
    pickle.dump(accuracy_dictionary, open(file_name, "wb"))
    
    
