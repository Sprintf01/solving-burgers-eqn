import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ==========================================
# 1. Parameters (from your setup)
# ==========================================
N_QUBITS = 5          # 2^5 = 32 spatial grid points
N        = 2**N_QUBITS
L        = 1.0        # domain length: x in [0, L]
dx       = L / N      # grid spacing = 0.03125
x        = np.linspace(0, L, N, endpoint=False)

T_FINAL  = 0.3        # total simulation time
N_STEPS  = 300        # number of timesteps
dt       = T_FINAL / N_STEPS   # = 0.001

NU       = 0.05       # default kinematic viscosity

# Initial condition
def u_init(x):
    return np.exp(-50.0 * (x - 0.5)**2)

# ==========================================
# 2. Linearized Crank-Nicolson Solver
# ==========================================
def solve_burgers_crank_nicolson(u0, nu=NU):
    u = u0.copy()
    history = [u.copy()]
    
    # Pre-calculate diffusion parameters
    alpha = (nu * dt) / (2.0 * dx**2)
    
    # We solve for interior points: index 1 to N-2 (size N-2)
    # Boundaries are fixed: u[0] = 0.0 and u[-1] = 0.0 (Dirichlet)
    M = N - 2  
    
    for _ in range(N_STEPS):
        # 1. Set up the Tridiagonal Matrix Banded form (required by scipy solve_banded)
        # Row 0: Upper diagonal (C_i)
        # Row 1: Main diagonal  (B_i)
        # Row 2: Lower diagonal (A_i)
        ab = np.zeros((3, M))
        
        # Define spatial advection scale variable dynamically at current timestep
        # We use upwind linearization based on the local direction of u^n
        u_interior = u[1:-1]
        beta = (u_interior * dt) / (2.0 * dx)
        
        # Center diagonal (B_i)
        ab[1, :] = 1.0 + beta + 2.0 * alpha
        
        # Upper diagonal (C_i) -> coefficient for u_{i+1}^{n+1}
        ab[0, 1:] = -alpha                  # C_i is constant except at boundaries
        
        # Lower diagonal (A_i) -> coefficient for u_{i-1}^{n+1}
        ab[2, :-1] = -beta[:-1] - alpha     # A_i depends on local velocity
        
        # 2. Construct Right Hand Side (D_i)
        d = np.zeros(M)
        for i in range(1, N - 1):
            idx = i - 1  # interior index mapping
            
            # Explicit parts (time-step n)
            adv_n = u[i] * (u[i] - u[i-1]) / dx
            diff_n = nu * (u[i+1] - 2.0*u[i] + u[i-1]) / dx**2
            
            # Crank-Nicolson Right Hand Side combination
            d[idx] = u[i] + (dt / 2.0) * (-adv_n + diff_n)
            
        # 3. Apply Dirichlet Boundary Conditions (u[0] = u[N-1] = 0)
        # Boundary adjustments to LHS equation system are zero since boundary values are 0.0.
        
        # 4. Solve the Tridiagonal System for interior points
        u_new = np.zeros(N)
        u_new[1:-1] = solve_banded((1, 1), ab, d)
        
        # Force boundaries explicitly
        u_new[0]  = 0.0
        u_new[-1] = 0.0
        
        u = u_new
        history.append(u.copy())
        
    return np.array(history)

# ==========================================
# 3. Execution & Visualization
# ==========================================
u0 = u_init(x)
u_hist = solve_burgers_crank_nicolson(u0, NU)

plt.figure(figsize=(9, 4))
for t_idx in [0, 50, 150, 300]:
    plt.plot(x, u_hist[t_idx], label=f't = {t_idx*dt:.3f}')
    
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title(f'Classical Burgers — Crank-Nicolson (nu = {NU})')
plt.legend()
plt.tight_layout()
plt.show()