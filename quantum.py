
# phase4_quantum.py
from params import *
from colehopf import *         
from classical import *       
from hamiltonian import *     

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


from qiskit.quantum_info import state_fidelity

import qiskit_aer
print(qiskit_aer.__version__)

try:
    import qiskit_algorithms
    print(qiskit_algorithms.__version__)
except ImportError:
    print("qiskit_algorithms not installed")


'''
# grid size must match qubit register size

assert N == 2 ** N_QUBITS, (
    f"N ({N}) must equal 2**N_QUBITS ({2**N_QUBITS}) "
    "for StatePreparation to work. Fix this in params.py."
)


# encode psi0 as a normalised quantum state

def encode_psi0(psi0):
    norm = np.linalg.norm(psi0)
    psi0_unit = psi0 / norm
    qc = QuantumCircuit(N_QUBITS)
    prep = StatePreparation(psi0_unit.tolist(), normalize=False)
    qc.append(prep, range(N_QUBITS))
    return qc, norm

qc_init, psi_norm = encode_psi0(psi0)
print(f'State norm (saved for rescaling): {psi_norm:.6f}')

sv_check = Statevector(qc_init).data.real
prep_err = np.max(np.abs(sv_check - psi0 / psi_norm))
print(f'State-prep max error: {prep_err:.2e}')   # expect ~1e-15


# evolve under the heat operator

#comment here
def evolve_heat_exact(psi0_unit, T_matrix, T_final):
    U = expm(T_matrix * T)
    return U @ psi0_unit

# to here
T = N_STEPS * dt
#here
psi0_unit = psi0 / psi_norm
psi_final_unit = evolve_heat_exact(psi0_unit, T_matrix, T)
psi_final = psi_final_unit * psi_norm   # undo normalisation
print('Heat evolution complete (exact exp(T t) applied).')
#to here
#create quant evolution gate
from qiskit.circuit.library import PauliEvolutionGate

evolution_gate = PauliEvolutionGate(
    H,
    time=T
)

qc.append(evolution_gate, range(N_QUBITS))

from qiskit_aer import AerSimulator

backend = AerSimulator(method="statevector")

psi_final = state.data
psi_final *= psi_norm


# Cole–Hopf inverse back to u(x, T)

u_quantum = cole_hopf_inverse(psi_final, nu=NU)

# Compare against classical solution
l2     = np.sqrt(np.mean((u_hist[-1] - u_quantum) ** 2))
l2_int = np.sqrt(np.mean((u_hist[-1][2:-2] - u_quantum[2:-2]) ** 2))
print(f'L2 error (full domain):   {l2:.2e}')
print(f'L2 error (interior only): {l2_int:.2e}')

plt.figure(figsize=(8, 4))
plt.plot(x, u_hist[-1], 'k-',  lw=2.5, label='Classical Burgers')
plt.plot(x, u_quantum,  'b--', lw=2,   label='Quantum (Cole-Hopf + exp(Tt))')
plt.xlabel('x'); plt.ylabel('u(x, T)')
plt.title(f'Classical vs Quantum L2 = {l2:.2e}')
plt.legend(); plt.tight_layout()
#plt.savefig('phase4_sanity.png', dpi=150); plt.show()
'''