from tensorflow import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras_tuner import HyperModel


# define the class for the LSTM model, containing 2 LSTM layers
class LSTMStacked(HyperModel):
    def __init__(self, input_shape,n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    # build function that takes hyperparameters as an input
    def build(self, hp):

        model = keras.Sequential()

        # define the number of units so that it can be optimized in the hyperparameters tuning step
        model.add(LSTM(
                        units=hp.Int(
                            'Units_LSTM_1',
                            min_value=32,
                            max_value=512,
                            step=32,
                            default=128
                        ),
                       input_shape=self.input_shape,
                       return_sequences = True
                    )
                )

        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                    )
                  )
                )
        # define the number of units so that it can be optimized in the hyperparameters tuning step
        model.add(LSTM(
                        units=hp.Int(
                            'Units_LSTM_2',
                            min_value=32,
                            max_value=512,
                            step=32,
                            default=128
                        )
                    )
        )

        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
                    )
            )
                )

        # define the number if units so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )

        model.add(Dense(self.n_outputs, activation='softmax'))

        # define the learning rate so that it can be optimized in the hyperparameters tuning step
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# define the class for the ConvLSTM model
class ConvLSTM(HyperModel):
    def __init__(self, input_shape, n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    # build function that takes hyperparameters as an input
    def build(self, hp):

        model = keras.Sequential()

        # define the the number of output filters so that it can be optimized in the hyperparameters tuning step
        model.add(
            ConvLSTM2D(
                filters=hp.Choice(
                        'num_filters',
                        values=[16, 32, 64],
                        default=16,
                ),
                kernel_size=(1, 3),
                activation='relu',
                input_shape=self.input_shape
            )
        )

        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        # # remove all of the dimensions of the inout tensor, except for one
        model.add(Flatten())

        # define the number if units and the activation function
        # so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )

        model.add(Dense(self.n_outputs, activation='softmax'))

        # define the learning rate so that it can be optimized in the hyperparameters tuning step
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# define the class for the CNNLSTM model
class CNNLSTM(HyperModel):
    def __init__(self, input_shape,n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    def build(self, hp):

        model = keras.Sequential()

        model.add(TimeDistributed(
            Conv1D(
                filters=hp.Choice(
                        'num_filters_1',
                        values=[16, 32, 64],
                        default=64,
                ),
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape,
            )
          )   
        )
        # define the the number of output filters so that it can be optimized in the hyperparameters tuning
        # TimeDistributed layer apply the same layer to several inputs.
        model.add(TimeDistributed(
            Conv1D(
                filters=hp.Choice(
                        'num_filters_2',
                        values=[16, 32, 64],
                        default=64,
                ),
                kernel_size=3,
                activation='relu'
            )
          )   
        )

        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(TimeDistributed(Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            )
           )
          )
        )

        # MaxPooling1D downsamples the input representation by taking
        # the maximum value over a spatial window of size pool_size

        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

        # remove all of the dimensions of the inout tensor, except for one
        model.add(TimeDistributed(Flatten()))

        # define the number if units so that it can be optimized in the hyperparameters tuning step
        model.add(
            LSTM(
                units=hp.Int(
                    'units_LSTM',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                )
            )
        )

        # define the dropout rate so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )

        # define the number if units and the activation function
        # so that it can be optimized in the hyperparameters tuning step
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(Dense(self.n_outputs, activation='softmax'))

        # define the learning rate so that it can be optimized in the hyperparameters tuning step
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
