import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

class LSTMController:

    def __init__(self,
                 x_scaler: MinMaxScaler,
                 y_scaler: MinMaxScaler,
                 input_shape,
                 units = 100,
                 dropout_rate = 0.1,
                 bounds = None,
                 h5 = None):
        self.x_scaler: MinMaxScaler = x_scaler
        self.y_scaler: MinMaxScaler = y_scaler
        self.lower = None
        self.upper = None
        self.verbose = "auto"

        if bounds:
            self.set_bounds(min(bounds), max(bounds))

        if h5:
            self.model: Sequential = load_model(h5)

        else:
            # Initialize the Keras model
            self.model = Sequential()

            # Create the first layer (an LSTM layer)
            self.model.add(LSTM(units = units, return_sequences = True, input_shape = input_shape))
            
            # Add a dropout layer to prevent overfitting
            self.model.add(Dropout(rate = dropout_rate))

            # Add another LSTM and Dropout layer
            self.model.add(LSTM(units = units))
            self.model.add(Dropout(rate = dropout_rate))

            # Finally add a Dense layer to return the prediction
            self.model.add(Dense(1))

            # Compile the model
            self.model.compile(optimizer = "adam", loss = "mean_squared_error")

    def __call__(self, set_point, process_var):
        error = set_point - process_var

        # Prepare input for LSTM
        X = np.vstack((set_point, error)).T
        Xs = self.x_scaler.transform(X)
        Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

        # Predict output
        ys = self.model.predict(Xs, verbose=self.verbose)
        y = self.y_scaler.inverse_transform(ys)

        # Clip output if needed
        if self.lower and y < self.lower:
            y = self.lower
        elif self.upper and y > self.upper:
            y = self.upper
        
        return y

    def set_bounds(self, lower = None, upper = None):
        self.lower = lower
        self.upper = upper
    
    def train(self, X, y, validation_split = 0.2):
        result = self.model.fit(X, y, verbose = 0,
                                validation_split = validation_split,
                                batch_size = 100,
                                epochs = 300)
        
        return result
    
    def save(self, filepath):
        self.model.save(filepath)

    def set_output_level(self, value = "auto"):
        self.verbose = value