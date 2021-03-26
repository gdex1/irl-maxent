import plotly.express as px
import numpy as np
from supervised_utils import sigmoid
import tensorflow as tf


def calc_instance_score(model, sequence, occlusion_weight=.7, label = 1, scale_constant=4, scale_function=np.tanh):
    """
    calculates the instance importance scorces for a specific sequence
    where each instance_score = scale_function(scale_constant * [w * occlusion_score + (1 - w) * outvar_score])
    where occlusion_score = prediction(sequence) - prediction(sequence \ i)
    where outvar_socre = prediction(sequence_to_i) = prediction(sequence_to_i-1)

    Args:
        model: A tf keras rnn to calculate instance scores on 
        sequence: A ragged tensor with bouding shape [1, sequence_length, number_o_features]
            ex: <tf.RaggedTensor [[[0, 2, 4], [4, 2, 9]]]>
        occlusion_weight: = [0,1], weight of occlusion score, (outvar_weight = 1 - occlusion_weight)
        label: label or class which we want to calculate instance scores on
        scale_constant: scalar to multiple instance scores before passing to scale function
        scale_function: function to bound input and provide non-linearity (such as tanh)
    Returns:
        instance_scores: an array of length sequence_length with instance_scores for each timestep
    
    """
    # length of sequence
    n = sequence.bounding_shape().numpy()[1]

    instance_scores = np.zeros(n)

    prediction = model.predict(sequence)[0][label]

    # increment instance score by occlusion_weight * occlusion_score to each instance score
    for i in range(n):
        # get and predict on sequence without instance i 
        seq_minus_i = tf.concat([sequence[0:1, 0:i], sequence[0:1, (i+1):]], axis=1)
        pred_minus_i = model.predict(seq_minus_i)[0][label]
        # calculate instance score
        instance_scores[i] = occlusion_weight * (prediction - pred_minus_i)

    pred_to_prev = None
    # increment instance score by (1 - occlusion_weight * outvar score)
    for i in range(1, n):
        # get sequence up until i and predict
        seq_to_i = sequence[0:1, :(i + 1)]
        pred_to_i = model.predict(seq_to_i)[0][label]
        # only calculate outvar on i >= 1
        if pred_to_prev is not None:
            outvar_score = (1 - occlusion_weight) * (pred_to_i - pred_to_prev)
            instance_scores[i] += outvar_score
        # update prev
        pred_to_prev = pred_to_i
    # scale and apply function
    instance_scores = scale_function(scale_constant * instance_scores)
    return instance_scores

def instance_importance_plot(data, y_test, seq_index, model, scale_constant=100):
    seq = data[seq_index:seq_index+1]
    y =  calc_instance_score(model, seq, occlusion_weight=.7, scale_constant=scale_constant, scale_function=sigmoid)
    x = np.arange(len(y))
    fig = px.imshow([y], zmin=0, zmax=1, color_continuous_scale='RdBu')
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=f'Instance Scores for Trajectory {seq_index}',
        xaxis_title='Timesteps'
    )
    print("sequence:", seq)
    print("prediction: ", model.predict(seq),"label: ", y_test[seq_index])
    return fig
