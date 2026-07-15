# compare.py

"""
Single entry point: runs all four solvers on the same params.py grid and
prints/plots a side-by-side comparison.

  1. Classical FTCS                 (classical.py)
  2. Classical Crank-Nicolson       (cranknicolson.py)
  3. Cole-Hopf + exact exp(Tt)     (same logic as quantum.py)
  4. Cole-Hopf + LCU exp(Tt)       (lcu.py)
  5. Pure PINN                     (pinn.py)

Run this last, after confirming each individual file runs on its own.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from params import *
from classical import u_hist
from cranknicolson import solve_burgers_crank_nicolson, u_hist as u_hist_cn  # Import from your Crank-Nicolson file
from colehopf import cole_hopf_forward, cole_hopf_inverse, psi0
from hamiltonian import H, T_matrix
from lcu import lcu_apply_segmented
from pinn import train_pinn, evaluate_pinn

T_FINAL = N_STEPS * dt
u_ref = u_hist[-1]
#u_cn = u_hist_cn[-1]  # Grab the final timestep for comparison


def l2(u):
    return np.sqrt(np.mean((u_ref - u) ** 2))


# --- 1. classical is already computed (u_hist) ---

# --- 2. Crank-Nicolson is imported (u_cn) ---

# --- 3. exact quantum (Cole-Hopf + matrix exponential) ---
psi_norm = np.linalg.norm(psi0)
psi0_unit = psi0 / psi_norm
psi_exact_unit = expm(T_matrix * T_FINAL) @ psi0_unit
u_quantum_exact = cole_hopf_inverse(psi_exact_unit * psi_norm, nu=NU)

# --- 4. LCU quantum ---
psi_lcu_unit, A, p_succ, n_terms = lcu_apply_segmented(H, T_FINAL, psi0_unit, K=6)
u_quantum_lcu = cole_hopf_inverse(psi_lcu_unit * psi_norm, nu=NU)

# --- 5. PINN ---
print("Training PINN (a minute or two on CPU)...")
pinn_model = train_pinn(n_epochs=1000)
u_pinn = evaluate_pinn(pinn_model, x, T_FINAL)


# --- report ---
print("\n=== L2 error vs classical FTCS (reference) ===")
#print(f"{'Crank-Nicolson':30s}: {l2(u_cn):.3e}")
print(f"{'Cole-Hopf + exact exp(Tt)':30s}: {l2(u_quantum_exact):.3e}")
print(f"{'Cole-Hopf + LCU (K=6)':30s}: {l2(u_quantum_lcu):.3e}   ")
      #f"[A={A:.2f}, success_prob={p_succ:.2e}, {n_terms} unitary terms]")
print(f"{'PINN':30s}: {l2(u_pinn):.3e}")


# --- plot ---
plt.figure(figsize=(9, 5))
plt.plot(x, u_ref,            'k-',  lw=2.5, label='Classical FTCS (Ref)')
#plt.plot(x, u_cn,             'm--', lw=2,   label='Classical Crank-Nicolson')
plt.plot(x, u_quantum_exact,  'b:',  lw=2,   label='Cole-Hopf + exact exp(Tt)')
plt.plot(x, u_quantum_lcu,    'g-.', lw=2,   label='Cole-Hopf + LCU')
plt.plot(x, u_pinn,            'r:',  lw=2.5, label='Pure PINN')

plt.xlabel('x')
plt.ylabel(f'u(x, T={T_FINAL:.3f})')
plt.title('Classical, Quantum (Exact & LCU), and Neural Network comparison — viscous Burgers')
plt.legend()
plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
plt.show()
