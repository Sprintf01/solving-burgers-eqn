# classical_v2
"""
- upwind advection is only 1st-order accurate -> adds numerical diffusion
    of order O(dx) that is NOT part of the true PDE
- explicit Euler time-stepping is only 1st-order accurate in dt
Fix here:
- 4th-order central differences for both u_x and u_xx (removes the O(dx)
    upwind bias)
- classical RK4 time integration (removes the O(dt) Euler bias)

Same interface as classical.py: produces `u_hist` of shape (N_STEPS+1, N)
using the same params.py (N, dx, dt, N_STEPS, NU, u_init, x)
"""

from params import *
import numpy as np

u0 = u_init(x)

CFL_adv = np.max(np.abs(u0)) * dt / dx
CFL_diff = NU * dt / dx**2
print(f'CFL advection : {CFL_adv:.4f}')
print(f'CFL diffusion : {CFL_diff:.4f}')
assert CFL_adv < 1.0,  'Increase N_STEPS to reduce dt'  #if it evaluates to False, Python immediately halts the program and raises an AssertionError
assert CFL_diff < 0.9, 'Increase N_STEPS to reduce dt'


def dudx_4th(u):
    """4th-order central first derivative, one-sided 2nd/3rd-order stencils
    at the two nearest-boundary points so nothing needs ghost cells."""
    d = np.zeros_like(u)
    d[2:-2] = (-u[4:] + 8*u[3:-1] - 8*u[1:-3] + u[:-4]) / (12*dx)
    d[1]    = (u[2] - u[0]) / (2*dx)                       # 2nd-order
    d[-2]   = (u[-1] - u[-3]) / (2*dx)
    d[0]    = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)            # one-sided
    d[-1]   = (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx)
    return d

def d2udx2_4th(u):
    """4th-order central second derivative, 2nd-order one-sided near boundary."""
    d = np.zeros_like(u)
    d[2:-2] = (-u[4:] + 16*u[3:-1] - 30*u[2:-2] + 16*u[1:-3] - u[:-4]) / (12*dx**2)
    d[1]    = (u[2] - 2*u[1] + u[0]) / dx**2                # 2nd-order
    d[-2]   = (u[-1] - 2*u[-2] + u[-3]) / dx**2
    d[0]    = (u[2] - 2*u[1] + u[0]) / dx**2                # reuse neighbour stencil
    d[-1]   = (u[-3] - 2*u[-2] + u[-1]) / dx**2
    return d

def rhs(u, nu):
    """du/dt = -u u_x + nu u_xx  (non-conservative form, matches classical.py)"""
    return -u * dudx_4th(u) + nu * d2udx2_4th(u)

def apply_bc(u):
    u[0] = 0.0
    u[-1] = 0.0
    return u

def solve_burgers_v2(u0, nu=NU):
    u = u0.copy()
    history = [u.copy()]
    for _ in range(N_STEPS):
        k1 = rhs(u, nu)
        k2 = rhs(apply_bc(u + 0.5*dt*k1), nu)
        k3 = rhs(apply_bc(u + 0.5*dt*k2), nu)
        k4 = rhs(apply_bc(u + dt*k3), nu)
        u = u + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        u = apply_bc(u)
        history.append(u.copy())
    return np.array(history)

u_hist = solve_burgers_v2(u_init(x))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4))
    for t_idx in [0, 50, 150, 300]:
        if t_idx < len(u_hist):
            plt.plot(x, u_hist[t_idx], label=f't = {t_idx*dt:.3f}')
    plt.xlabel('x'); plt.ylabel('u(x, t)')
    plt.title(f'Classical Burgers — 4th-order central + RK4 (nu = {NU})')
    plt.legend(); plt.tight_layout()
    plt.show()

    # compare against old FTCS for a sanity check
    from classical import u_hist as u_hist_ftcs
    l2 = np.sqrt(np.mean((u_hist[-1] - u_hist_ftcs[-1])**2))
    print(f"L2 difference vs old FTCS solver: {l2:.2e}")

'''
OLD CLASSICAL, ONLY FTCS  

from params import *

u0 = u_init(x)

CFL_adv  = np.max(np.abs(u0)) * dt / dx
CFL_diff = NU * dt / dx**2
print(f'CFL advection : {CFL_adv:.4f}  ') #(must be < 0.5)
print(f'CFL diffusion : {CFL_diff:.4f}')
 
assert CFL_adv  < 0.5, 'Increase N_STEPS to reduce dt'
assert CFL_diff < 0.5, 'Increase N_STEPS to reduce dt'
def solve_burgers(u0, nu=NU):
    u = u0.copy()
    history = [u.copy()]
    for _ in range(N_STEPS):
        u_new = u.copy()
        for i in range(1, N - 1):
            adv  = u[i] * (u[i] - u[i-1]) / dx       # upwind
            diff = nu * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
            u_new[i] = u[i] + dt * (-adv + diff)
        u_new[0]  = 0.0    # Dirichlet: u = 0 at left wall
        u_new[-1] = 0.0   # Dirichlet: u = 0 at right wall
        u = u_new
        history.append(u.copy())
    return np.array(history)    # shape: (N_STEPS+1, N)
 
u_hist = solve_burgers(u_init(x))
import matplotlib.pyplot as plt
 
plt.figure(figsize=(9, 4))
for t_idx in [0, 50, 150, 300]:
    plt.plot(x, u_hist[t_idx], label=f't = {t_idx*dt:.3f}')
plt.xlabel('x'); plt.ylabel('u(x, t)')
plt.title(f'Classical Burgers — FTCS  (nu = {NU})')
plt.legend(); plt.tight_layout()
#plt.savefig('phase1_result.png', dpi=150); plt.show()
'''