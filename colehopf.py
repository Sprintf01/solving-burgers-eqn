from scipy.integrate import cumulative_trapezoid
from params import *
from classical import *

def cole_hopf_forward(u0, nu=NU):
    """
    Forward Cole-Hopf: u0 → psi0 = exp(-(1/2nu) * integral_0^x u0 ds)
    Uses trapezoidal rule — 2nd order, matches FD solver accuracy.
    """
    integral = cumulative_trapezoid(u0, dx=dx, initial=0.0)
    psi0 = np.exp(-integral / (2.0 * nu))
    return psi0

def solve_heat(psi0, nu=NU):
    psi = psi0.copy()
    history = [psi.copy()]
    coeff = nu * dt / dx**2

    for _ in range(N_STEPS):
        psi_new = psi.copy()
        # Update interior points only
        psi_new[1:-1] = psi[1:-1] + coeff * (psi[2:] - 2*psi[1:-1] + psi[:-2])
        # Dirichlet BC: hold boundary values fixed at their t=0 values
        psi_new[0] = psi_new[1]
        psi_new[-1] = psi_new[-2]   # fixed at initial value
        psi = psi_new
        history.append(psi.copy())
    return np.array(history)

def cole_hopf_inverse(psi, nu=NU):
    """
    Inverse Cole-Hopf: psi → u = -2nu * dpsi/dx / psi
    Uses np.gradient (central differences interior, one-sided at boundary).
    """
    logpsi = np.log(np.maximum(psi,1e-14))
    dlogpsi = np.gradient(logpsi, dx,edge_order=2)

    return -2.0*nu*dlogpsi
#dpsi_dx = np.gradient(psi, dx)
    #return -2.0 * nu * dpsi_dx / (psi + 1e-14) 
    #dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dx)
    #dpsi_dx[0]  = (psi[1] - psi[0]) / dx     # one-sided at left
    #dpsi_dx[-1] = (psi[-1] - psi[-2]) / dx   # one-sided at right

# check 
psi0     = cole_hopf_forward(u_init(x))
#print(f"psi0 range: [{psi0.min():.4f}, {psi0.max():.4f}]")

# Roundtrip check at t=0 (must be < 1e-3 before running heat solver)
u0_rt = cole_hopf_inverse(psi0)
l2_rt = np.sqrt(np.mean((u_init(x) - u0_rt)**2))
#print(f"Roundtrip L2 at t=0: {l2_rt:.2e} ")

psi_hist  = solve_heat(psi0)
u_ch_final = cole_hopf_inverse(psi_hist[-1])

l2_full = np.sqrt(np.mean((u_hist[-1] - u_ch_final)**2))
l2_int  = np.sqrt(np.mean((u_hist[-1][2:-2] - u_ch_final[2:-2])**2))
print(f"L2 error :     {l2_full:.2e}") #fulldomain
#print(f"L2 error interior only:   {l2_int:.2e}")