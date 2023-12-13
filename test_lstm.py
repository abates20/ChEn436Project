from project import PID, utils
import pandas as pd
import numpy as np
import os
import tclab

generate_data = False
make_gif = True

# Create a PID controller to generate testing data
pid = PID(0, 10, 60, 0)
pid.set_bounds(0, 100)

# Generate data for testing
if generate_data or not "lstm_testing_data.csv" in os.listdir("data"):
    data = utils.generate_tclab_data(pid, minutes=30)
    data.to_csv("data/lstm_testing_data.csv", index=False)
else:
    data = pd.read_csv("data/lstm_testing_data.csv")

# Load the LSTM model
h5_file = "data/lstm_control.h5"
train_data = pd.read_csv("data/lstm_training_data.csv")
train_data["Error"] = train_data["SP1"] - train_data["T1"]

X = train_data[["SP1", "Error"]].values
y = train_data[["Q1"]].values
x_scaler, y_scaler, Xtrain, ytrain = utils.prepare_data(X, y)

window = Xtrain.shape[1] # window size
n_features = Xtrain.shape[2] # number of features
input_shape = (window, n_features)
lstm = utils.load_lstm_from_h5(h5_file, x_scaler, y_scaler, input_shape)

# Initial test of the LSTM on generated data
Q1 = data["Q1"].values
T1 = data["T1"].values
SP1 = data["SP1"].values
Q1_pred = np.zeros_like(Q1)
tm = data["Time"].values

for i in range(window, len(tm)):
    SP1_input = SP1[i - window:i]
    T1_input = T1[i - window:i]
    Q1_pred[i] = lstm(SP1_input, T1_input)

utils.plot_lstm_test(data, Q1_pred, to_file="plots/lstm_check.png")


# Full test of the LSTM as the primary controller
TCLab = tclab.TCLab

# Uncomment this line to use the simulated tclab
# TCLab = tclab.setup(connected = False, speedup = 30)

with TCLab() as lab:
    start_T = lab.T1

    # Set the T1 set point values
    n = 15 * 60
    T_setpoint = np.zeros(n) + start_T
    T_setpoint[20:] = 60
    T_setpoint[350:] = 35
    T_setpoint[700:] = 45

    # Control the temperature using the provided PID controller
    Q = np.zeros(n)
    T = np.zeros(n)
    tm = np.zeros(n)

    for i, t in enumerate(tclab.clock(n - 1)):
        tm[i] = t

        if i > window:
            SP1_input = T_setpoint[i - window:i]
            T1_input = T[i - window:i]
            Q[i] = lstm(SP1_input, T1_input)
        
        T[i] = lab.T1
        lab.Q1(Q[i])

# Trim the arrays to just the values got filled in
tm = tm[:i]
Q = Q[:i]
T = T[:i]
T_setpoint = T_setpoint[:i]

# Store the data in a pandas dataframe
data = pd.DataFrame(np.vstack((tm, Q, T, T_setpoint)).T, columns=["Time", "Q1", "T1", "SP1"])
data.to_csv("data/lstm_test_results.csv", index=False)

# Plot the results
utils.plot_tclab_data(data, to_file="plots/lstm_test.png")

# Make a GIF if make_gif is set to True
if make_gif:
    import imageio

    images = []

    for i in range(1, data.shape[0]):
        temp_data = data[:i]
        utils.plot_tclab_data(temp_data, to_file=f"plots/gif/plot{i}.png")
        images.append(imageio.imread(f"plots/gif/plot{i}.png"))

    for _ in range(10):
        images.append(imageio.imread(f"plots/gif/plot{i}.png"))

    imageio.mimsave("plots/lstm_test.gif", images, fps=10)