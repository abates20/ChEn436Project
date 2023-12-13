import os
from project import PID, LSTMController as LSTM, utils

generate_data = True

# Create a PID controller to generate training data
pid = PID(0, 10, 60, 0)
pid.set_bounds(0, 100)

# Generate the training data, plot it, and save it
if generate_data or not "lstm_training_data.csv" in os.listdir("data"):
    data = utils.generate_tclab_data(pid)
    utils.plot_tclab_data(data, to_file="plots/train_data.png")
    data.to_csv("data/lstm_training_data.csv", sep=",", index=False)
else:
    import pandas as pd
    data = pd.read_csv("data/lstm_training_data.csv", sep=",")

# Pull out the features from the data and analyze them
features = data[["SP1", "T1"]].copy()
features["Error"] = features["SP1"] - features["T1"]

utils.analyze_features(features, data["Q1"].values, to_file="plots/features.png")

# Based on feature analysis, we'll use set point (SP1) and error

# Prepare the data for training the LSTM model
X = features[["SP1", "Error"]].values
y = data[["Q1"]].values
x_scaler, y_scaler, Xtrain, ytrain = utils.prepare_data(X, y)

# Create the model and train it (training takes several minutes)
window = Xtrain.shape[1] # window size
n_features = Xtrain.shape[2] # number of features
input_shape = (window, n_features)
model = LSTM(x_scaler, y_scaler, input_shape)

result = model.train(Xtrain, ytrain)
utils.plot_loss(result, to_file="plots/training_loss.png")

# Save the trained model
model.save("data/lstm_control.h5")