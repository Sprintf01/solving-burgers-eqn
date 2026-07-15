# lcu.py
"""
LCU (Linear Combination of Unitaries) solver for the heat/diffusion sub-step
of the Cole-Hopf quantum method.

This file implements a segmented LCU solver to approximate exp(T * t) using a 
truncated Taylor series applied across multiple small time segments. 
This prevents the LCU 1-norm (A) from exploding and keeps the success 
probability physically viable.
"""

from params import *
from hamiltonian import H, T_matrix
from colehopf import cole_hopf_forward, cole_hopf_inverse
from classical import u_hist
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def taylor_lcu_terms(H_pauli: SparsePauliOp, t_step, K):
    """
    Return list of (coeff, pauli_label) for exp(H t_step) ~ sum_k (t_step^k/k!) H^k.
    """
    terms = []
    n_q = H_pauli.num_qubits
    Hk = SparsePauliOp.from_list([("I" * n_q, 1.0)])   # H^0 = I
    fact = 1.0
    for k in range(K + 1):
        coeff_k = (t_step ** k) / fact
        for label, c in zip(Hk.paulis.to_labels(), Hk.coeffs):
            terms.append((coeff_k * c, label))
        if k < K:
            Hk = (Hk @ H_pauli).simplify(atol=1e-12)
            fact *= (k + 1)
    return terms


def lcu_apply_segmented(H_pauli, t_final, psi_in, K=6, n_segments=30):
    """
    Emulate the PREPARE-SELECT-PREPARE^+ block encoding of exp(H t) applied to psi_in
    using segmented time-stepping to maintain physical stability.

    Returns:
        psi_out       -- the (unnormalised) output vector, ~ exp(H t) @ psi_in
        total_A       -- product of LCU 1-norms across all segments (overall amplitude scaling)
        total_p_succ  -- joint probability of measuring all ancillas as |0> across steps
        n_terms       -- number of unitary terms in the decomposition per segment
    """
    dt_seg = t_final / n_segments
    psi_current = psi_in.copy().astype(complex)
    
    # Generate the Taylor LCU terms for a single small time step
    terms = taylor_lcu_terms(H_pauli, dt_seg, K)
    A_seg = sum(abs(c) for c, _ in terms)
    
    # Pre-build matrices to optimize simulation speed
    U_matrices = [SparsePauliOp(label).to_matrix() for _, label in terms]
    coeffs = [c for c, _ in terms]
    
    total_A = 1.0
    total_p_succ = 1.0
    
    # Step sequentially through each time segment
    for step in range(n_segments):
        out = np.zeros_like(psi_current, dtype=complex)
        for c, U in zip(coeffs, U_matrices):
            out += c * (U @ psi_current)
            
        norm_out = np.linalg.norm(out)
        
        # Track simulated success probability for this segment
        # p_succ = (||U_approx |psi>|| / A_seg)^2
        p_succ_seg = (norm_out / A_seg) ** 2
        total_p_succ *= p_succ_seg
        total_A *= A_seg
        
        # Update current state for the next step (normalized)
        psi_current = out / norm_out

    # Rescale the final normalized state vector back to the original physical scale
    # using the cumulative product of segment norms
    final_norm = np.linalg.norm(psi_in) * (total_A * np.sqrt(total_p_succ))
    psi_out = psi_current * final_norm
    
    return psi_out, total_A, total_p_succ, len(terms)


if __name__ == "__main__":
    # 1. Forward Cole-Hopf transform on the initial condition
    psi0 = cole_hopf_forward(u_init(x))
    psi_norm = np.linalg.norm(psi0)
    psi0_unit = psi0 / psi_norm
    T_FINAL = N_STEPS * dt

    # 2. Run Segmented LCU solver
    # K=6 (Taylor order), n_segments=30 (keeps ||H * dt_seg|| ~ 0.25)
    psi_final_unit, A, p_succ, n_terms = lcu_apply_segmented(
        H, T_FINAL, psi0_unit, K=6, n_segments=30
    )
    psi_final = psi_final_unit * psi_norm

    # 3. Inverse Cole-Hopf transform back to velocity space u(x, T)
    u_lcu = cole_hopf_inverse(psi_final.real, nu=NU)

    # 4. Compare against classical FTCS ground truth
    l2 = np.sqrt(np.mean((u_hist[-1] - u_lcu) ** 2))
    
    print("================ LCU SOLVER RESULTS ================")
    print(f"LCU terms per segment     : {n_terms}")
    print(f"Segmented LCU 1-norm (A)  : {A:.4e}")
    print(f"LCU joint success prob    : {p_succ:.4e}")
    print(f"L2 error vs classical     : {l2:.2e}")
    print("====================================================")