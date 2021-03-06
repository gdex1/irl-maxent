from lstm_model import LstmModel
import numpy as np
from trajectory import Trajectory, generate_trajectory, generate_trajectories, stochastic_policy_adapter
from solver import value_iteration, stochastic_policy_from_value_expectation
from snake_ladder import SnakeLadderWorld
from supervised_utils import trajectory_list_to_xy, shuffle, x_to_ragged, train_test_split, compute_class_accuracy, sigmoid, get_fixed_policies, get_expert_policy, generate_trajectories_from_policy_list
import tensorflow as tf
from lstm_model import LstmModel
import plotly.express as px
import datetime
from importance import calc_instance_score, instance_importance_plot

def main():
    # define some consants
    world_size = 20
    shortcut_density = 0.1
  
    # create our world
    world = SnakeLadderWorld(size=world_size, shortcut_density=shortcut_density)   

    # game board
    print("board: ")
    print(world.game_board)


    # create "fixed" policies which each execeute one of the three actions w/ prob p (success_prob)
    # randomly sample from all actions w/ prob 1 - p
    # so excute one action with prob p + 1/3(1 - p) and others with 1/3(1 -  p)
    fixed_policies = get_fixed_policies

    # get policy using value iteration
    expert_policy = get_expert_policy(world)


    # create list of policies
    policies = []
    #policies = policies_fixed
    policies.append(expert_policy) # add expert policy to list
    policies.append(world._smartish_policy)
    num_policies = len(policies)

    # generate trajectories for all policies
    # each index of list contains array of corresponding policy trajectories
    n_trajectories_per_policy = 1000
    trajectories_list = generate_trajectories_from_policy_list(world, policies,n_trajectories_per_policy=n_trajectories_per_policy)

    # print an example trajectory
    # a trajectory from policy 0
    print(trajectories_list[0][0])

    # seperate trajectories into x,y data
    x_data, y_data = trajectory_list_to_xy(trajectories_list)
    x_data, y_data = shuffle(x_data, y_data)

    # convert data to ragged tensor
    # max_seq contains length of longest trajectory
    x_data, max_seq = x_to_ragged(x_data)

    y_data = np.array(y_data)

    # do a simple train/test split
    x_train, y_train, x_test, y_test = train_test_split(x_data,y_data, test_prop =.20)

    # create lstm model
    lstm_model = LstmModel(max_trajectory_len=max_seq, num_features=3, num_outputs=num_policies)
    print(lstm_model.model.summary())


    # train model
    lstm_model.train(x_train, y_train, x_test, y_test, log_dir="./logs/fit/", 
                     epochs = 500, batch_size=int(n_trajectories_per_policy / 10), early_stopping=True,patience=10)

    # compute accuracy by class
    y_predicted = lstm_model.predict_classes(x_test)
    print(compute_class_accuracy(y_test, y_predicted))

    # create instance importance plot
    for i in range(5):
        trajectory_index = i
        fig = instance_importance_plot(x_test, y_test, trajectory_index, lstm_model, scale_constant=10)
        fig.show()
    


if __name__ == "__main__":
    main()















