import numpy as np
import tclab
import os

# Get the physics-based model
from project import PhysicsModel

# Initialize the model with some initial guess values
U = 10 # W/m2/K
epsilon = 0.9
A = 1.2e-3 # m2
mcp = 0.004 * 500 # J/K
model = PhysicsModel(U, epsilon, A, mcp)


# Measure the step change on the TCLab and save the data
minutes = 10
n = minutes * 60
tm = np.zeros(n + 1)
Q = np.zeros(n + 1)
Q[10:] = 80
Q[200:] = 20
Q[400:] = 60

T1_meas = np.zeros(n + 1)
i = 0

if not "dynamics_data.csv" in os.listdir("data"):

    TCLab = tclab.TCLab

    # Uncomment the following line to use the TCLab model instead
    TCLab = tclab.setup(connected=False, speedup=20)

    with TCLab() as lab:
        for t in tclab.clock(n):
            tm[i] = t
            lab.Q1(Q[i])
            T1_meas[i] = lab.T1
            i += 1

    data = np.vstack((tm, Q, T1_meas)).T
    np.savetxt("data/dynamics_data.csv", data, delimiter=",", header="Time (s), Heater (%), T1 (°C)")

else:
    data = np.loadtxt("data/dynamics_data.csv", skiprows=1, delimiter=",").T
    tm = data[0]
    Q = data[1]
    T1_meas = data[2]


# Compare the initial guess to the measured data
model.plot_with_data(tm, Q, T1_meas, "plots/initial_guess.png")

# Optimize the Physics-based model using the measured data
model.fit(tm, Q, T1_meas)

print("\nRegressed parameters:")
print(f"U = {model.U} W/m2/K")
print(f"epsilon = {model.epsilon}")
print(f"A = {model.A} m2")
print(f"mcp = {model.mcp} J/K")

# Compare the fitted model to the measure data
model.plot_with_data(tm, Q, T1_meas, "plots/optimized_model.png")