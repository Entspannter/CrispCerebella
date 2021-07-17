class LSTMStacked(HyperModel):
    def __init__(self, input_shape,n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    def build(self, hp):

        model = keras.Sequential()

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
        


        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=64,
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

class ConvLSTM(HyperModel):
    def __init__(self, input_shape,n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    def build(self, hp):


        model = keras.Sequential()

        model.add(
            ConvLSTM2D(
                filters=hp.Choice(
                        'num_filters',
                        values=[16, 32, 64],
                        default=64,
                ),
                kernel_size=(1,3),
                activation='relu',
                input_shape=self.input_shape
            )
        )

        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
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
      
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

        model.add(TimeDistributed(Flatten()))

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