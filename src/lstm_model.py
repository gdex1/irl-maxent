from supervised_model import SupervisedModel
import tensorflow as tf
import datetime
import os


class LstmModel(SupervisedModel):
    def __init__(self, max_trajectory_len, num_features, num_outputs, architecture = None ):
        self.num_features = num_features
        self.max_trajectory_len = max_trajectory_len
        self.num_outputs = num_outputs

        if architecture is None:
            self.model = default_network(max_trajectory_len, num_features, num_outputs)
        else:
            self.model = architecture
        
        metrics = ['accuracy']

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=metrics)

    def train(self, x_train, y_train, x_test, y_test, log_dir = None, epochs = 100, batch_size = 32):
        if log_dir is not None:
            log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        else:
            tensorboard_callback = None
        
        self.model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size, 
                       validation_data=(x_test,y_test), callbacks=[tensorboard_callback])

    # both functions expect x to be TensorShape([# samples, None, None]), so 1 trajectory should be 3 dimensional TensorShape([1, None, None])
    def predict(self, x):
        return self.model.predict(x)

    def predict_classes(self, x):
        return self.model.predict_classes(x)


def default_network(max_trajectory_len, num_features, num_outputs, dropout=.5, recurrent_nodes = 64, dense_nodes = 64):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_trajectory_len, num_features), dtype=tf.float32, ragged=True),
        tf.keras.layers.LSTM(recurrent_nodes),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dense_nodes, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='softmax')
    ])
    return model