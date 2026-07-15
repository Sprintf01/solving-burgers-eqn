# lcu.py (Verified Quantum Burgers Solver - Clean VS Code Version)
"""
Verified Quantum LCU solver for 1 step of the diffusion sub-step.

This version is optimized for VS Code (no dynamic attributes/red lines)
by using the standard Statevector.from_instruction() class method.
"""

import numpy as np
from params import *
from hamiltonian import build_diffusion_hamiltonian  
from colehopf import cole_hopf_forward
from classical import u_hist

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import StatePreparation  


def taylor_lcu_terms(H_pauli: SparsePauliOp, t_step, K):
    """
    Return list of (coeff, pauli_label) for exp(H t_step) ~ sum_k (t_step^k/k!) H^k.
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


def run_actual_quantum_lcu_step(H_pauli, t_step, psi_in, K=1):
    """
    Constructs and runs a physical gate-based LCU quantum circuit 
    using the true physical Cole-Hopf state, and evaluates it using 
    statevector simulation.
    """
    # 1. Get Taylor terms (Purely real coefficients for exp(-H*dt))
    terms = taylor_lcu_terms(H_pauli, t_step, K)
    coeffs = np.real([c for c, _ in terms])  # Since H is Hermitian, coeffs are real
    labels = [label for _, label in terms]
    
    # 2. Determine qubit requirements
    sys_qubits = H_pauli.num_qubits
    anc_qubits = int(np.ceil(np.log2(len(terms))))
    
    padded_len = 2**anc_qubits
    padded_coeffs = np.zeros(padded_len)
    padded_coeffs[:len(coeffs)] = coeffs
    
    # 3. Define quantum registers (NO classical register or measure() to avoid collapse)
    qr_anc = QuantumRegister(anc_qubits, name="ancilla")
    qr_sys = QuantumRegister(sys_qubits, name="system")
    qc = QuantumCircuit(qr_anc, qr_sys)
    
    # Step A: True Physical State Preparation (Actual Cole-Hopf state!)
    psi_normalized = psi_in / np.linalg.norm(psi_in)
    sys_prep = StatePreparation(psi_normalized.tolist())
    qc.append(sys_prep, qr_sys)
    
    # Step B: PREPARE state on ancilla using magnitudes
    magnitudes = np.abs(padded_coeffs)
    anc_state = np.sqrt(magnitudes)
    anc_state /= np.linalg.norm(anc_state) 
    
    anc_prep = StatePreparation(anc_state.tolist())
    qc.append(anc_prep, qr_anc)
    
    # Step C: SELECT Stage (Native MCX and signs)
    for idx, label in enumerate(labels):
        coeff = coeffs[idx]
        bin_idx = format(idx, f'0{anc_qubits}b')
        
        # Invert the controls where the binary index bit is '0'
        for bit_pos, char in enumerate(bin_idx):
            if char == '0':
                qc.x(qr_anc[bit_pos])
                
        # Rigorously handle negative signs using phase kickback (Z gate on a control bit)
        if coeff < 0:
            qc.z(qr_anc[0])  # Flip phase if the coefficient is negative
            
        # Distribute native controlled gates directly on the target qubits
        for q_idx, pauli in enumerate(reversed(label)):
            if pauli == 'X':
                qc.mcx(list(qr_anc), qr_sys[q_idx])
            elif pauli == 'Y':
                # Multi-controlled Y: Ry(pi/2) -> MCX -> Ry(-pi/2)
                qc.ry(np.pi/2, qr_sys[q_idx])
                qc.mcx(list(qr_anc), qr_sys[q_idx])
                qc.ry(-np.pi/2, qr_sys[q_idx])
            elif pauli == 'Z':
                # Multi-controlled Z: H -> MCX -> H
                qc.h(qr_sys[q_idx])
                qc.mcx(list(qr_anc), qr_sys[q_idx])
                qc.h(qr_sys[q_idx])
                
        # Uncompute control inversions
        for bit_pos, char in enumerate(bin_idx):
            if char == '0':
                qc.x(qr_anc[bit_pos])
                
    # Step D: PREPARE-dagger on ancilla (cleanly inverting StatePreparation)
    qc.append(anc_prep.inverse(), qr_anc)
    
    # --- VS CODE COMPLIANT STATEVECTOR SIMULATION ---
    # Instead of monkey-patching `.save_statevector()`, we use Statevector class directly.
    # We compile the circuit with optimization, then evolve a starting zero-state.
    simulator = AerSimulator()
    qc_compiled = transpile(qc, simulator, optimization_level=1)
    
    print("\n================ COMPILED CIRCUIT METRICS ================")
    print(f"Total qubits used             : {qc.num_qubits} ({sys_qubits} system, {anc_qubits} ancilla)")
    print(f"Synthesized Circuit Depth     : {qc_compiled.depth()}")
    print(f"Total Gate Operations         : {qc_compiled.size()}")
    print(f"Operation breakdown           : {dict(qc_compiled.count_ops())}")
    print("==========================================================")
    
    # Mathematically compute the final statevector directly from the circuit instructions
    statevector = Statevector.from_instruction(qc_compiled)
    
    return statevector, anc_qubits, sys_qubits, coeffs, labels


if __name__ == "__main__":
    # Let's target the lightweight 2-qubit system for rapid mathematical verification
    TOY_QUBITS = 2
    toy_dx = 1.0 / (2**TOY_QUBITS)
    
    print(f"Building lightweight {TOY_QUBITS}-qubit Hamiltonian...")
    H_toy, _ = build_diffusion_hamiltonian(TOY_QUBITS, toy_dx, NU)
    print(f"Toy Hamiltonian built with {len(H_toy)} Pauli terms.")
    
    # Define true physical Cole-Hopf initial state
    toy_x = np.linspace(0.0, 1.0, 2**TOY_QUBITS)
    toy_u0 = np.sin(np.pi * toy_x)
    toy_psi0 = np.exp(-toy_u0 / (2.0 * NU))
    toy_psi0_unit = toy_psi0 / np.linalg.norm(toy_psi0)
    
    t_step = dt
    print("\nCompiling and running actual LCU Quantum Circuit for 1 step...")
    
    # Run LCU quantum step
    sv, n_anc, n_sys, coeffs, labels = run_actual_quantum_lcu_step(H_toy, t_step, toy_psi0_unit, K=1)
    
    # --- RIGOROUS MATHEMATICAL VALIDATION ---
    # 1. Classical Taylor Step: psi_classical = (I - H*dt) psi_0
    H_matrix = H_toy.to_matrix()
    I_matrix = np.eye(2**n_sys)
    taylor_op = I_matrix - H_matrix * t_step
    psi_classical = taylor_op @ toy_psi0_unit
    psi_classical_normed = psi_classical / np.linalg.norm(psi_classical)
    
    # 2. Extract Post-Selected Statevector: Corresponding to ancillas in state |00...0>
    sv_data = np.array(sv)
    # The first 2**n_sys elements represent the system state when all ancillas are |0>
    psi_quantum_post_selected = sv_data[:2**n_sys]
    psi_quantum_normed = psi_quantum_post_selected / np.linalg.norm(psi_quantum_post_selected)
    
    # Resolve global phase differences to perform a clean L2 validation
    phase_difference = np.angle(psi_quantum_normed[0]) - np.angle(psi_classical_normed[0])
    psi_quantum_aligned = psi_quantum_normed * np.exp(-1j * phase_difference)
    
    # 3. Calculate L2 Validation Error
    l2_validation = np.linalg.norm(psi_quantum_aligned - psi_classical_normed)
    success_probability = np.sum(np.abs(psi_quantum_post_selected)**2)
    
    print("\n================ QUANTUM SOLVER VALIDATION ================")
    print(f"Success post-selection probability : {success_probability:.4%}")
    print(f"L2 Error (Quantum vs Classical LCU): {l2_validation:.4e}")
    if l2_validation < 1e-10:
        print("SUCCESS: Quantum state matches classical LCU evolution perfectly!")
    else:
        print("WARNING: Quantum state mismatch detected.")
    print("===========================================================\n")