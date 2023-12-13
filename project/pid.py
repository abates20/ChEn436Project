# PID controller class
class PID:
    """A PID controller.

    Initialize by providing values for `ubias`, `Kc`, `tauI`, `tauD`.
    Additional optional parameters are `Kff` for feed foward control and
    `bounds` to set a minimum and maximum for the output u value.

    Example
    -------
    >>> # An example controller for a TCLab (where Q is between 0 and 100%)
    >>> pid = PID(ubias = 0, Kc = 5, tauI = 50, tauD = 0)
    >>> pid.set_bounds(0, 100)
    >>>    
    >>> # Calculate u by calling the controller
    >>> u = pid(set_point = 40, process_var = 23, dt = 1)
    >>> print(u)
    86.7
    """

    def __init__(self, ubias, Kc, tauI, tauD, Kff = 0, bounds = None) -> None:
        """Initialize the PID controller.

        Parameters
        ----------
        ubias: `float`
            The steady state value of u.

        Kc: `float`
            The gain parameter for the PID controller.

        tauI: `float`
            The integral time parameter for the PID controller.

        tauD: `float`
            The derivative time parameter for the PID controller.

        Kff: `float`
            An optional parameter to specify the gain for feed forward control.

        bounds: `tuple`
            Lower and/or upper bounds on u.
        """
        self.ubias = ubias
        self.Kc = Kc
        self.tauI = tauI
        self.tauD = tauD
        self.Kff = Kff
        self.integral_error = 0
        if bounds:
            self.set_bounds(min(bounds), max(bounds))
        else:
            self.set_bounds(None, None)
        
        # Lists to hold values as PID is called
        self.set_point = []
        self.process_var = []
        self.u = []
        self.time = []
        self.error = []

    def __call__(self, set_point, process_var, dt, process_var2 = 0) -> float:
        """Calculate u based on the current set point and value of the process
        variable.
        
        Parameters
        ----------
        set_point: `float`
            The current set point for the process variable.

        process_var: `float`
            The current value of the process variable.

        dt: `float`
            The time delta.

        process_var2: `float`
            An optional argument for a second process variable if feed forward
            control is being used.

        Returns
        -------
        `float`
            The calculated u value.
        """
        error = set_point - process_var
        self.integral_error += error * dt

        dpv = 0 if len(self.process_var) == 0 else process_var - self.process_var[-1]
        dpvdt = dpv / dt if dt > 0 else 0

        # Calculate u
        P = self.Kc * error
        I = self.Kc / self.tauI * self.integral_error
        D = -self.Kc * self.tauD * dpvdt
        u = self.ubias + P + I + D + self.Kff * (process_var - process_var2)

        # Check bounds
        if not self.upper_bound is None and u > self.upper_bound:
            u = self.upper_bound
            self.integral_error -= error * dt
        if not self.lower_bound is None and u < self.lower_bound:
            u = self.lower_bound
            self.integral_error -= error * dt

        # Update lists
        self.set_point.append(set_point)
        self.process_var.append(process_var)
        self.u.append(u)
        self.error.append(error)

        if len(self.time) == 0:
            self.time.append(0)
        else:
            self.time.append(self.time[-1] + dt)

        return u
    
    def set_bounds(self, lower = None, upper = None):
        """Set the minimum and maximum for u.

        Parameters
        ----------
        lower: `float`
            The minimum value (lower bound) for u.

        upper: `float`
            The maximum value (upper bound) for u.

        Returns
        -------
        `None`
        """
        self.lower_bound = lower
        self.upper_bound = upper
    
    def reset(self):
        """Reset the PID controller.

        This clears out the recorded values for the set point, process 
        variable, u, time, and error. It also resets the integral error
        to 0.
        """
        self.integral_error = 0
        self.set_point = []
        self.process_var = []
        self.u = []
        self.time = []
        self.error = []