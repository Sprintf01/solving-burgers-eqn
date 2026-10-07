from params import *
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def build_diffusion_hamiltonian(n_qubits, dx, nu):
    """
    Build the FD Laplacian as SparsePauliOp.
    T[i,i]   = -2 * nu/dx^2
    T[i,i+1] = T[i+1,i] = nu/dx^2
    Neumann BC: corner elements are -nu/dx^2 (one neighbour only).
    """
    n = 2**n_qubits
    c = nu / dx**2           # off-diagonal coefficient

    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = -2.0 * c
        if i > 0:     T[i, i-1] = c
        if i < n-1:   T[i, i+1] = c
    # Neumann BC: boundary nodes have only one interior neighbour
    T[0, 0]   = -c
    T[-1, -1] = -c

    # Decompose into Pauli basis
    H = SparsePauliOp.from_operator(T)
    H = H.simplify(atol=1e-10)   # drop negligible terms
    return H, T
 
H, T_matrix = build_diffusion_hamiltonian(N_QUBITS, dx, NU)
 
# Verify: H.to_matrix() must reproduce T_matrix
H_recovered = H.to_matrix().real
max_err = np.max(np.abs(H_recovered - T_matrix))
print(f'Hamiltonian reconstruction error: {max_err:.2e}')  # expect < 1e-12
print(f'Number of Pauli terms: {len(H)}') 
