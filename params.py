
import numpy as np

# spatial grid 
N_QUBITS = 5          # 2^5 = 32 spatial grid points
N        = 2**N_QUBITS
L        = 1.0        # domain length: x in [0, L]
dx       = L / N      # grid spacing = 0.03125
x        = np.linspace(0, L, N, endpoint=False)

# time stepping 
T_FINAL  = 0.3        # total simulation time
N_STEPS  = 300       # number of timesteps
dt       = T_FINAL / N_STEPS   # = 0.001


NU       = 0.05       # default kinematic viscosity
NU_SWEEP = [0.10, 0.05, 0.02, 0.01]  # for the viscosity study

# initial condition: Gaussian pulse centred at x = 0.5
def u_init(x):
    return np.exp(-50.0 * (x - 0.5)**2)
