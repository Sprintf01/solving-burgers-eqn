
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
