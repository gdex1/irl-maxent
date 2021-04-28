
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



def experiment_train_test(n_train_samples, n_test_samples, world_size, shortcut_density, num_worlds):

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(num_worlds):
        world = SnakeLadderWorld(size=world_size, shortcut_density=shortcut_density)

        # create our policies   
        fixed_policies = get_fixed_policies()
        expert_policy = get_expert_policy(world)

        # create list of policies
        policies = []
        #policies = fixed_policies
        policies.append(expert_policy) # add expert policy to list
        policies.append(world._smartish_policy)

        
        n_train_samples_per_policy = int(n_train_samples / len(policies))
        n_test_samples_per_policy = int(n_test_samples / len(policies))


        # generate training trajectoreies and testing trajectories
        train_trajectories = generate_trajectories_from_policy_list(world, policies, n_trajectories_per_policy= n_train_samples_per_policy)
        test_trajectories = generate_trajectories_from_policy_list(world, policies, n_trajectories_per_policy= n_test_samples_per_policy)
        # convert to x,y data form
        _x_train, _y_train = trajectory_list_to_xy(train_trajectories)
        _x_test, _y_test = trajectory_list_to_xy(test_trajectories)
        # append to data
        x_train.extend(_x_train)
        x_test.extend(_x_test)
        y_train.extend(_y_train)
        y_test.extend(_y_test)




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
    num_worlds_per_size = 30

    # test with one world
    # define some consants
    shortcut_density = 0.1
    n_policies = 2

    n_test_samples = 500
    n_train_samples = 200
    def experiment_train_test_wrapper(world_size):
        return experiment_train_test(n_train_samples, n_test_samples, world_size, shortcut_density, num_worlds_per_size) 
    
    # create list of n to try
    #num_list = [20, 50, 100, 200, 400, 800, 1000]
    world_sizes = [10, 20, 30]

    # create 1d arg list
    arg_list = []
    for size in world_sizes:
        for i in range(n_trials):
            arg_list.append((size))

    num_workers = 6 # number of physical cores
    pool = multiprocessing.Pool(num_workers, init_worker)

    #class_accuracies = pool.starmap(experiment_train_test_wrapper, arg_list)
    class_accuracies = pool.map(experiment_train_test_wrapper, arg_list)


    print(class_accuracies)
    # save to numpy file
    class_accuracies = np.vstack(class_accuracies)
    description = 'binary-expert-smart-sizes'
    world_str = str(world_sizes).replace(' ','')
    file_name = f'{description}_policies_{n_policies}_worlds_{world_str}_numWorlds_{num_worlds_per_size}_trials_{n_trials}_density_{int(shortcut_density*100)}'
    #np.save(os.path.join('experiments', file_name), class_accuracies)

    #np.split(array, len(num_list),axis=0)
    # convert to dictionary
    accuracy_dictionary = {}
    for index, arr in enumerate(np.split(class_accuracies, len(world_sizes),axis=0)):
        accuracy_dictionary[world_sizes[index]] = arr
    # write to file
    pickle.dump(accuracy_dictionary, open(os.path.join('./experiments', file_name), "wb"))


    # # write worlds to file
    # world_file = open(os.path.join('experiments', f'worlds_{file_name}'),'w')
    # for world in worlds:
    #     world_file.write(str(world.game_board))
    # world_file.close()
    
    
