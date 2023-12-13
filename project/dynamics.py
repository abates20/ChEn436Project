import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the Physics-based Model
class PhysicsModel:

    def __init__(self, U, epsilon, A, mcp):
        # Model parameters
        # U = heat transfer coefficient
        # epsilon = emissivity
        # A = surface area
        # mcp = heat capacity

        self.U = U
        self.epsilon = epsilon
        self.A = A
        self.mcp = mcp

    def __call__(self, T, tm, Q, Ta):
        sigma = 5.67e-8 # W/m2/K4
        alpha = 0.01 # W/%

        # Model calculations
        convection = self.U * self.A * (Ta - T)
        radiation = self.epsilon * sigma * self.A * (Ta ** 4 - T ** 4)
        generation = alpha * Q

        dTdt = (convection + radiation + generation) / self.mcp
        return dTdt
    
    def simulate(self, tm, Q, T0 = 23):
        T1 = np.zeros(len(tm))
        T1[0] = T0

        for i in range(1, len(tm)):
            time_step = tm[i - 1:i + 1]
            T1[i] = odeint(self, T1[i - 1], time_step, args=(Q[i], T0))[-1]
        
        return T1
    
    def fit(self, tm, Q, T1):
        from scipy.optimize import minimize

        def objective(params, tm, Q, T1_meas):
            model = PhysicsModel(*params)
            T1 = model.simulate(tm, Q, T1_meas[0])
            squared_error = (T1_meas - T1) ** 2
            SSE = sum(squared_error)

            # Penalty if epsilon is greater than 1 or less than 0
            if params[1] > 1 or params[1] < 0:
                SSE += 1e6
            return SSE
        
        initial_params = (self.U, self.epsilon, self.A, self.mcp)
        print(f"\nInitial objective = {objective(initial_params, tm, Q, T1)}")
        params = minimize(objective, initial_params, args=(tm, Q, T1)).x
        print(f"Final objective = {objective(params, tm, Q, T1)}")

        self.U, self.epsilon, self.A, self.mcp = params
        return self
    
    def plot_with_data(self, tm, Q, T1, to_file = None):
        T1_sim = self.simulate(tm, Q, T1[0])

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(tm, T1_sim, color="blue", linestyle="-", label="T1 Simulated")
        plt.plot(tm, T1, ".", color="red", label="T1 Measured")
        plt.ylabel("Temperature (°C)")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(tm, Q, color="black", linestyle="--", label="Q1")
        plt.ylabel("Heater (%)")
        plt.legend()
        
        plt.xlabel("Time (s)")

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.show()