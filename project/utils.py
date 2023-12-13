import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tclab

from .pid import PID

def generate_tclab_data(pid: PID, minutes = 120, speedup = 200):
    "Generate data from the TCLab for training an LSTM model."
    # Get the TCLab model
    TCLab = tclab.setup(connected = False, speedup = speedup)

    # Start set point at room temperature
    with TCLab() as lab:
        start_T = lab.T1

    # Generate a random series of temperature set point steps
    n = minutes * 60
    T_low = 30; T_hi = 70
    T_setpoint = np.zeros(n) + start_T
    j = 30
    while j <= n:
        i = j
        j += random.randint(4 * 60, 10 * 60)
        T_setpoint[i:j] = random.randint(T_low, T_hi)

    # Control the temperature using the provided PID controller
    Q = np.zeros(n)
    T = np.zeros(n)
    tm = np.zeros(n)

    with TCLab() as lab:
        for i, t in enumerate(tclab.clock(n - 1)):
            tm[i] = t
            dt = 0
            if i > 0:
                dt = tm[i] - tm[i - 1]
            
            T[i] = lab.T1
            Q[i] = pid(T_setpoint[i], T[i], dt)
            lab.Q1(Q[i])
    
    # Trim the arrays to just the values got filled in
    tm = tm[:i]
    Q = Q[:i]
    T = T[:i]
    T_setpoint = T_setpoint[:i]

    # Store the data in a pandas dataframe
    data = pd.DataFrame(np.vstack((tm, Q, T, T_setpoint)).T, columns=["Time", "Q1", "T1", "SP1"])
    return data


def plot_tclab_data(data: pd.DataFrame, to_file = None):
    "Plots data from a pandas DataFrame."
    tm = data["Time"].values
    Q1 = data["Q1"].values
    T1 = data["T1"].values
    T1_setpoint = data["SP1"].values

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(tm / 60, T1_setpoint, "k-", label="Set point")
    plt.plot(tm / 60, T1, "r.", label="T1")
    plt.ylabel("Temperature (°C)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tm / 60, Q1, "r-", label="Q1")
    plt.ylabel("Heater (%)")
    plt.xlabel("Time (min)")
    plt.legend()

    if to_file:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()


def analyze_features(X: pd.DataFrame, y: np.ndarray, plot = True, to_file = None):
    """Use the `SelectKBest` object to
    determine the scores for feature selection.
    """
    from sklearn.feature_selection import SelectKBest, f_regression

    selector = SelectKBest(f_regression, k="all")
    fit = selector.fit(X, y)

    if plot:
        plt.figure()
        plt.bar(X.columns, fit.scores_)

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.show()
    
    return fit


def prepare_data(X, y, window = 15):
    """Prepare data so that it can be used to
    train the LSTM model.
    
    This includes scaling the data, formatting
    it, and splitting into training and test
    sets.
    """
    from sklearn.preprocessing import MinMaxScaler

    x_scaler = MinMaxScaler()
    Xs = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    ys = y_scaler.fit_transform(y)

    # Prepare the input and output arrays
    inputs = []
    outputs = []
    n = len(y)
    i = window
    while i < n:
        inputs.append(Xs[i - window:i])
        outputs.append(ys[i])
        i += 1

    # Reshape the inputs and outputs
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    # Return the scalers and the sets
    return x_scaler, y_scaler, inputs, outputs


def plot_loss(train_result, to_file = None):
    "Plot the loss from training the LSTM model."
    plt.figure()
    plt.semilogy(train_result.history["loss"], label = "Training")
    plt.semilogy(train_result.history["val_loss"], label = "Validation")
    plt.ylabel("Loss")
    plt.legend()

    if to_file:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()


def load_lstm_from_h5(filepath, x_scaler, y_scaler, input_shape):
    from .lstm import LSTMController
    lstm = LSTMController(x_scaler, y_scaler, input_shape, h5=filepath)
    return lstm


def plot_lstm_test(tclab_data: pd.DataFrame, Q1_pred, to_file = None):
    "Plots data from a pandas DataFrame."
    tm = tclab_data["Time"].values
    Q1 = tclab_data["Q1"].values
    T1 = tclab_data["T1"].values
    T1_setpoint = tclab_data["SP1"].values

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(tm / 60, T1_setpoint, "k-", label="Set point")
    plt.plot(tm / 60, T1, "r.", label="T1")
    plt.ylabel("Temperature (°C)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(tm / 60, Q1, "r-", label="PID")
    plt.plot(tm / 60, Q1_pred, "b--", label="LSTM")
    plt.ylabel("Heater (%)")
    plt.xlabel("Time (min)")
    plt.legend()

    if to_file:
        plt.savefig(to_file)
        plt.close()
    else:
        plt.show()