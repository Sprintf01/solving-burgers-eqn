# lcu.py
"""
Real, gate-level LCU solve of the Cole-Hopf heat sub-step, replacing the
exact expm(T*T_FINAL) in quantum.py with N_STEPS genuine Aer circuit
executions (one per dt), using the verified core in lcu_core.py.

WHY PER-STEP, NOT ONE SHOT:
A truncated Taylor series is only a good approximation to exp(Ht) when |Ht|
is small. quantum.py can legally call expm(T * T_FINAL) once because expm()
is exact -- there's no truncation. LCU here uses a K-term Taylor series, so
it has to be applied once per dt (small step), N_STEPS times, exactly like a
Trotterized time evolution.

WHY WE CAN RENORMALIZE FREELY BETWEEN STEPS:
cole_hopf_inverse(psi) = -2*nu * d/dx[log(psi)]. Multiplying psi by any
overall (even step-dependent) positive scalar shifts log(psi) by a constant,
which vanishes under d/dx. So only psi's *shape* matters for recovering u --
we can renormalize to a unit vector after every step without affecting the
final answer. That's what lets us feed each step's output straight back in
as the next step's input state.
"""

import numpy as np
from params import *
from hamiltonian import H
from colehopf import cole_hopf_forward, cole_hopf_inverse
from classical import u_hist
from lcu_core_trial import build_lcu_circuit, run_lcu_on_aer

K_TAYLOR = 1          # Taylor order per step; raise for tighter per-step accuracy
FINAL_STEP_SHOTS = 20000   # real Aer shots, just for the final reported success prob


def solve_heat_lcu(psi0, H_pauli, n_steps, dt, K=K_TAYLOR, verbose=True):
    psi_current = psi0 / np.linalg.norm(psi0)
    history = [psi_current.copy()]
    p_succ_last = None

    for step in range(n_steps):
        is_last = (step == n_steps - 1)
        qc, qr_anc, qr_sys, A, n_terms = build_lcu_circuit(H_pauli, dt, psi_current, K=K)
        shots = FINAL_STEP_SHOTS if is_last else 0
        out, p_theory, p_emp, _ = run_lcu_on_aer(qc, qr_anc, qr_sys, A, shots=shots)

        psi_current = out / np.linalg.norm(out)
        history.append(psi_current.copy())

        if is_last:
            p_succ_last = (p_theory, p_emp)
        if verbose and (step % max(1, n_steps // 10) == 0):
            print(f"  step {step+1:4d}/{n_steps}  "
                  f"ancilla qubits={len(qr_anc)}  A={A:.3f}  p_succ(theory)={p_theory:.3%}")

    return np.array(history), p_succ_last


if __name__ == "__main__":
    psi0 = cole_hopf_forward(u_init(x))

    print(f"Running LCU heat solve: {N_STEPS} steps, K={K_TAYLOR} Taylor order per step...")
    psi_hist_lcu, (p_theory_last, p_emp_last) = solve_heat_lcu(psi0, H, N_STEPS, dt, K=K_TAYLOR)

    u_lcu = cole_hopf_inverse(psi_hist_lcu[-1], nu=NU)

    l2 = np.sqrt(np.mean((u_hist[-1] - u_lcu) ** 2))
    print("\n================ LCU RESULT ================")
    print(f"Final-step theoretical success prob : {p_theory_last:.3%}")
    print(f"Final-step empirical success prob   : {p_emp_last:.3%}  "
          f"({FINAL_STEP_SHOTS} real Aer shots)")
    print(f"L2 error vs classical (classical_v2) : {l2:.3e}")
    print("==============================================")
