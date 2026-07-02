
import numpy as np
from scipy.linalg import expm
from .base import Channel
I2=np.eye(2,dtype=np.complex128); I4=np.eye(4,dtype=np.complex128)
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.array([[1,0],[0,-1]],complex)
PAULIS=[I2,X,Y,Z]
ORDER_SENSITIVE_FAMILIES=["imperfect_cnot","imperfect_swap","correlated_dephasing","two_qubit_depolarizing"]
COHERENT_2Q_FAMILIES=["imperfect_cnot","imperfect_swap"]
STOCHASTIC_2Q_FAMILIES=["correlated_dephasing","two_qubit_depolarizing"]

def two_qubit_depolarizing(p: float) -> Channel:
    p=float(np.clip(p,0,1)); kraus=[np.sqrt(1-p)*I4]
    non_id=[np.kron(a,b) for a in PAULIS for b in PAULIS if not (np.allclose(a,I2) and np.allclose(b,I2))]
    kraus += [np.sqrt(p/15)*u for u in non_id]
    return Channel("two_qubit_depolarizing",4,kraus=kraus,params=np.array([p]))

def correlated_dephasing(p: float, corr: float=1.0) -> Channel:
    p=float(np.clip(p,0,1)); corr=float(np.clip(corr,0,1))
    zi=np.kron(Z,I2); iz=np.kron(I2,Z); zz=np.kron(Z,Z)
    weights=np.array([max(0,1-p), p*(1-corr)/2, p*(1-corr)/2, p*corr],float)
    weights=weights/weights.sum()
    return Channel("correlated_dephasing",4,kraus=[np.sqrt(weights[0])*I4,np.sqrt(weights[1])*zi,np.sqrt(weights[2])*iz,np.sqrt(weights[3])*zz],params=np.array([p,corr]))

def cnot_unitary():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)

def swap_unitary():
    return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)

def imperfect_gate(kind: str="cnot", theta: float=0.02, p: float=0.01) -> Channel:
    base = cnot_unitary() if kind == "cnot" else swap_unitary()
    h = np.kron(Z,Z)
    u = expm(-1j*theta*h) @ base
    dep = two_qubit_depolarizing(p)
    ideal_coherent = Channel(f"imperfect_{kind}_coherent",4,kraus=[u],params=np.array([theta,p]))
    return dep.compose_after(ideal_coherent)

def sample_two_qubit(rng: np.random.Generator, family: str | None=None) -> Channel:
    fams=["correlated_dephasing","two_qubit_depolarizing","imperfect_cnot","imperfect_swap"]
    family=family or rng.choice(fams)
    if family=="correlated_dephasing": return correlated_dephasing(rng.uniform(0,0.2), rng.uniform(0.2,1.0))
    if family=="two_qubit_depolarizing": return two_qubit_depolarizing(rng.uniform(0,0.15))
    if family=="imperfect_cnot": return imperfect_gate("cnot", rng.uniform(-0.08,0.08), rng.uniform(0,0.05))
    if family=="imperfect_swap": return imperfect_gate("swap", rng.uniform(-0.08,0.08), rng.uniform(0,0.05))
    raise ValueError(family)


def _draw_order_sensitive_family(rng: np.random.Generator, active_families: list[str], index: int, required_family: str | None, required_remaining: int) -> str:
    if required_family is not None and required_remaining > 0 and required_remaining >= len(active_families) - 1:
        return required_family
    choices = list(active_families)
    if required_family is not None and required_remaining > 0 and required_family not in choices:
        choices.append(required_family)
    return str(rng.choice(choices))


def _draw_order_sensitive_channel(rng: np.random.Generator, family: str, index: int) -> Channel:
    if family == "imperfect_cnot":
        theta = rng.uniform(-0.70, 0.70) if index % 6 in (0, 4) else rng.uniform(-0.45, 0.45)
        p = rng.uniform(0.0, 0.08) if index % 6 == 0 else rng.uniform(0.0, 0.06)
        return imperfect_gate("cnot", theta, p)
    if family == "imperfect_swap":
        theta = rng.uniform(-0.70, 0.70) if index % 6 in (2, 5) else rng.uniform(-0.45, 0.45)
        p = rng.uniform(0.0, 0.08) if index % 6 == 2 else rng.uniform(0.0, 0.06)
        return imperfect_gate("swap", theta, p)
    if family == "correlated_dephasing":
        return correlated_dephasing(rng.uniform(0.02, 0.25), rng.uniform(0.2, 1.0))
    if family == "two_qubit_depolarizing":
        return two_qubit_depolarizing(rng.uniform(0.02, 0.18))
    raise ValueError(f"unsupported order-sensitive family: {family}")


def sample_order_sensitive_two_qubit_sequence(
    rng: np.random.Generator,
    length: int,
    min_random_gap: float = 0.01,
    max_attempts: int = 128,
    num_random_permutations: int = 16,
    exclude_family: str | None = None,
    required_family: str | None = None,
):
    from physics.composition import exact_sequence_fidelity

    if length < 2:
        raise ValueError("order-sensitive sequence requires length >= 2")

    active_families = [fam for fam in ORDER_SENSITIVE_FAMILIES if fam != exclude_family]
    if not active_families:
        raise ValueError("order-sensitive sampling has no active families")
    if required_family is not None:
        if required_family == exclude_family:
            raise ValueError("required_family cannot equal exclude_family")
        if required_family not in ORDER_SENSITIVE_FAMILIES:
            raise ValueError(f"unknown required_family: {required_family}")
        if required_family not in active_families:
            active_families.append(required_family)

    best = None
    for _ in range(max_attempts):
        seq = []
        required_remaining = 1 if required_family is not None else 0
        for i in range(length):
            fam = _draw_order_sensitive_family(rng, active_families, i, required_family, required_remaining)
            if fam == required_family and required_remaining > 0:
                required_remaining -= 1
            seq.append(_draw_order_sensitive_channel(rng, fam, i))
        if required_family is not None and not any(required_family in ch.name for ch in seq):
            continue
        f_ref = exact_sequence_fidelity(seq)
        rev = list(reversed(seq))
        f_rev = exact_sequence_fidelity(rev)
        best_perm_gap = -1.0
        best_perm_fid = f_ref
        for _perm_try in range(num_random_permutations):
            perm = list(seq)
            rng.shuffle(perm)
            f_perm = exact_sequence_fidelity(perm)
            gap = abs(f_ref - f_perm)
            if gap > best_perm_gap:
                best_perm_gap = gap
                best_perm_fid = f_perm
        reverse_gap = abs(f_ref - f_rev)
        random_gap = best_perm_gap
        score = random_gap + 0.10 * reverse_gap
        record = (score, seq, random_gap, reverse_gap, f_ref, best_perm_fid, f_rev)
        if best is None or score > best[0]:
            best = record
        if random_gap >= min_random_gap:
            _, seq, random_gap, reverse_gap, f_ref, f_perm, f_rev = record
            return seq, {
                "perm_gap_random": float(random_gap),
                "perm_gap_reverse": float(reverse_gap),
                "fidelity_forward": float(f_ref),
                "fidelity_random_perm": float(f_perm),
                "fidelity_reverse": float(f_rev),
            }

    assert best is not None
    _, seq, random_gap, reverse_gap, f_ref, f_perm, f_rev = best
    return seq, {
        "perm_gap_random": float(random_gap),
        "perm_gap_reverse": float(reverse_gap),
        "fidelity_forward": float(f_ref),
        "fidelity_random_perm": float(f_perm),
        "fidelity_reverse": float(f_rev),
    }
