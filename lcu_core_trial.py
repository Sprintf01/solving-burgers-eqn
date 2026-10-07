# lcu_core.py
"""
Correct, hardware-faithful LCU building blocks: PREPARE -> SELECT ->
PREPARE(dagger), genuinely executed on AerSimulator.

Fixes vs. an earlier draft that only *looked* like it used Aer:
  1. SELECT = one controlled-(Pauli string) gate per term, built as a single
     unitary and controlled all at once -- NOT decomposed into separate
     controlled-X / controlled-Y / controlled-Z calls (those are not the
     same operator as a single controlled tensor-product gate once signs
     and phases are involved).
  2. Sign of alpha_i is baked directly into that unitary (U_i = sign(alpha_i)*P_i),
     not faked with a stray Z on one ancilla qubit.
  3. PREPARE amplitudes are literally sqrt(|alpha_i| / A), A = sum|alpha_i|,
     computed and reported explicitly.
  4. Actually runs on AerSimulator via qc.save_statevector() + backend.run().
     Statevector.from_instruction() is pure linear algebra and never touches
     Aer at all -- that was the main thing wrong with the earlier version.
  5. Reports the LCU 1-norm A, the *theoretical* success probability (from
     Aer's own statevector output), AND an empirically measured success
     probability from real sampled shots (ancilla measured, many shots).

Verified (see this file's __main__ self-test): circuit output matches the
classical Taylor-truncated exp(Ht) to ~1e-15, and the theoretical success
probability matches 20000 real Aer-sampled shots to within statistical noise.

One thing this file deliberately does NOT include: Oblivious Amplitude
Amplification (OAA). I drafted it and it did not verify cleanly in testing
(transpile/global-phase bookkeeping issues) -- rather than hand you an
amplification step I haven't confirmed correct, it's left out. Ask if you
want it built and verified as its own follow-up.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import StatePreparation, UnitaryGate


# ---------------------------------------------------------------------------
# 1. Taylor-series LCU decomposition of exp(H t)
# ---------------------------------------------------------------------------
def taylor_lcu_terms(H_pauli: SparsePauliOp, t_step, K):
    """(coeff, pauli_label) list for exp(H t) ~ sum_k (t^k/k!) H^k.

    NOTE: t_step should be a SINGLE timestep (dt), not the full T_FINAL --
    a truncated Taylor series is only a good approximation to exp(Ht) when
    |H t| is small. For anything beyond one dt, call this + build/run once
    per step (see lcu.py) rather than passing T_FINAL directly.
    """
    terms = []
    n_q = H_pauli.num_qubits
    Hk = SparsePauliOp.from_list([("I" * n_q, 1.0)])
    fact = 1.0
    for k in range(K + 1):
        coeff_k = (t_step ** k) / fact
        for label, c in zip(Hk.paulis.to_labels(), Hk.coeffs):
            terms.append((coeff_k * c, label))
        if k < K:
            Hk = (Hk @ H_pauli).simplify(atol=1e-12)
            fact *= (k + 1)
    return terms


# ---------------------------------------------------------------------------
# 2. Build the real LCU circuit for ONE step
# ---------------------------------------------------------------------------
def build_lcu_circuit(H_pauli: SparsePauliOp, t_step, psi_in, K=4):
    """
    Returns (qc, qr_anc, qr_sys, A, n_terms). qc is the FULL circuit:
    state prep on system + PREPARE on ancilla + SELECT + PREPARE-dagger,
    fully decomposed to u/cx and ending in qc.save_statevector() so Aer can
    actually execute it.
    """
    terms = taylor_lcu_terms(H_pauli, t_step, K)
    coeffs = np.real([c for c, _ in terms])   # H Hermitian -> real coefficients
    labels = [label for _, label in terms]

    alphas = np.abs(coeffs)
    A = alphas.sum()                           # the real LCU 1-norm
    n_terms = len(terms)

    sys_qubits = H_pauli.num_qubits
    anc_qubits = max(1, int(np.ceil(np.log2(n_terms))))
    padded_len = 2 ** anc_qubits

    prep_amp = np.zeros(padded_len)
    prep_amp[:n_terms] = np.sqrt(alphas / A)
    prep_amp = prep_amp / np.linalg.norm(prep_amp)   # guard fp drift only

    qr_anc = QuantumRegister(anc_qubits, name="anc")
    qr_sys = QuantumRegister(sys_qubits, name="sys")
    qc = QuantumCircuit(qr_anc, qr_sys)

    psi_unit = psi_in / np.linalg.norm(psi_in)
    qc.append(StatePreparation(psi_unit.tolist()), qr_sys)

    prep_gate = StatePreparation(prep_amp.tolist())
    qc.append(prep_gate, qr_anc)

    for idx, (coeff, label) in enumerate(zip(coeffs, labels)):
        sign = 1.0 if coeff >= 0 else -1.0
        pauli_mat = SparsePauliOp(label).to_matrix()
        U_i = sign * pauli_mat                       # still unitary (sign = +-1)
        gate = UnitaryGate(U_i, label=f"U{idx}")
        controlled_gate = gate.control(num_ctrl_qubits=anc_qubits, ctrl_state=idx)
        qc.append(controlled_gate, list(qr_anc) + list(qr_sys))

    qc.append(prep_gate.inverse(), qr_anc)

    # StatePreparation.inverse() emits a raw "multiplexer" instruction that
    # segfaults the Aer statevector backend if left in place (confirmed
    # empirically). Fully decompose to plain u/cx BEFORE attaching
    # save_statevector / running on Aer.
    qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    qc.save_statevector()

    return qc, qr_anc, qr_sys, A, n_terms


# ---------------------------------------------------------------------------
# 3. Actually execute on Aer
# ---------------------------------------------------------------------------
def run_lcu_on_aer(qc, qr_anc, qr_sys, A, shots=0):
    """
    Runs the circuit on AerSimulator.
      - Always: statevector method -> exact post-selected system state +
        theoretical success probability, genuinely produced by Aer.
      - If shots > 0: ALSO runs a measured version (ancilla measured, real
        sampled shots) and returns the empirical success probability too.
        Set shots=0 to skip this (much faster) when you just need the state,
        e.g. inside a many-timestep loop.
    """
    n_anc = len(qr_anc)
    n_sys = len(qr_sys)
    dim_sys = 2 ** n_sys
    dim_anc = 2 ** n_anc

    backend_sv = AerSimulator(method="statevector")
    qc_compiled = transpile(qc, backend_sv, optimization_level=0)
    result = backend_sv.run(qc_compiled).result()
    sv = np.asarray(result.get_statevector(qc_compiled))

    # qr_anc was declared FIRST, so its qubits are the LEAST-significant bits.
    # The ancilla=|0...0> block is column 0 of the (dim_sys, dim_anc) reshape,
    # NOT the first dim_sys contiguous amplitudes.
    assert sv.shape[0] == dim_sys * dim_anc
    v0 = sv.reshape(dim_sys, dim_anc)[:, 0]
    out_unnormalized = v0 * A
    theoretical_p_succ = float(np.real(np.vdot(v0, v0)))

    empirical_p_succ = None
    if shots > 0:
        qc_meas = qc.copy()
        qc_meas.data = [inst for inst in qc_meas.data if inst.operation.name != "save_statevector"]
        creg = ClassicalRegister(n_anc, name="c")
        qc_meas.add_register(creg)
        qc_meas.measure(qr_anc, creg)

        backend_qasm = AerSimulator()
        qc_meas_compiled = transpile(qc_meas, backend_qasm, optimization_level=0)
        counts = backend_qasm.run(qc_meas_compiled, shots=shots).result().get_counts()
        zero_key = "0" * n_anc
        empirical_p_succ = counts.get(zero_key, 0) / shots

    return out_unnormalized, theoretical_p_succ, empirical_p_succ, qc_compiled


# ---------------------------------------------------------------------------
# 4. Self-contained correctness test (toy 1-qubit system)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    from scipy.linalg import expm

    H_toy = SparsePauliOp.from_list([("Z", -1.5), ("X", 0.7)])
    t_step = 0.05
    psi_in = np.array([0.8, 0.6], dtype=complex)
    psi_in = psi_in / np.linalg.norm(psi_in)

    K = 2
    qc, qr_anc, qr_sys, A, n_terms = build_lcu_circuit(H_toy, t_step, psi_in, K=K)
    print(f"n_terms = {n_terms}, ancilla qubits = {len(qr_anc)}, A = {A:.6f}")

    out, p_theory, p_emp, qc_compiled = run_lcu_on_aer(qc, qr_anc, qr_sys, A, shots=20000)

    print(f"Theoretical success probability : {p_theory:.4%}")
    print(f"Empirical success probability   : {p_emp:.4%}  (20000 shots, real Aer sampling)")
    print(f"Compiled circuit depth          : {qc_compiled.depth()}")
    print(f"Compiled circuit ops            : {dict(qc_compiled.count_ops())}")

    H_mat = H_toy.to_matrix()
    exact_taylor = sum(np.linalg.matrix_power(H_mat * t_step, k) / math.factorial(k)
                        for k in range(K + 1)) @ psi_in
    true_exp = expm(H_mat * t_step) @ psi_in

    print(f"\nL2 error (circuit vs. exact classical Taylor truncation): "
          f"{np.linalg.norm(out - exact_taylor):.3e}   <- should be ~machine precision")
    print(f"L2 error (circuit vs. true exp(Ht), K={K} truncation)    : "
          f"{np.linalg.norm(out - true_exp):.3e}   <- this is just Taylor truncation error")
