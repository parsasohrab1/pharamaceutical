#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول HQCA: تولید داده سنتتیک، مدار VQC، پیش‌بینی و گزارش
نسخه: 3.0
"""

import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from reporting import generate_pdf_report, generate_text_report
from scoring import binding_energy_to_score, estimate_confidence
from visualization import export_pocket_pdb, generate_pocket_viewer_html

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS, Descriptors, Lipinski

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    raise ImportError("لطفاً RDKit را نصب کنید: pip install rdkit")

try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane unavailable; using classical fallback.")

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pennylane_qiskit import QiskitAer  # type: ignore

    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False

BackendName = Literal[
    "auto",
    "pennylane_default_qubit",
    "qiskit_aer_simulator",
    "classical_fallback",
]
MAX_CIRCUIT_DEPTH = 20
COBYLA_MAXITER = 200
DEFAULT_OUTPUT_DIR = Path(os.getenv("HQCA_OUTPUT_DIR", "output"))


# ---------------------------------------------------------------------------
# F1 — تولید مولکول سنتتیک (ChemBFN-compatible RDKit pipeline)
# ---------------------------------------------------------------------------
class SyntheticMoleculeGenerator:
    """
    تولید مولکول با جهش ساختاری RDKit (BRICS، جایگزینی اتم، اسکافولد)
    به‌عنوان جایگزین ChemBFN برای ثبت اختراع و آموزش QML.
    """

    REPLACEMENTS = [
        ("C", "N"), ("C", "O"), ("N", "C"), ("F", "Cl"), ("O", "S"),
    ]

    def __init__(self, seed_smiles_list: List[str]):
        self.seed_smiles = list(set(seed_smiles_list))
        self.valid_mols = [
            Chem.MolFromSmiles(s) for s in self.seed_smiles if Chem.MolFromSmiles(s)
        ]
        self.valid_smiles = [
            Chem.MolToSmiles(m) for m in self.valid_mols if m is not None
        ]
        if not self.valid_smiles:
            raise ValueError("هیچ SMILES معتبری در بذر اولیه وجود ندارد.")
        self._brics_fragments = self._collect_brics_fragments()

    def _collect_brics_fragments(self) -> List[str]:
        frags: List[str] = []
        for smi in self.valid_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            for frag in BRICS.BRICSDecompose(mol):
                if Chem.MolFromSmiles(frag):
                    frags.append(frag)
        return frags or self.valid_smiles.copy()

    def _rdkit_mutate(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rw = Chem.RWMol(mol)
        atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() > 1 and not a.IsInRing()]
        if not atoms:
            return None
        idx = random.choice(atoms)
        atom = rw.GetAtomWithIdx(idx)
        old_sym = atom.GetSymbol()
        candidates = [b for a, b in self.REPLACEMENTS if a == old_sym]
        if not candidates:
            candidates = ["C", "N", "O"]
        new_sym = random.choice(candidates)
        if new_sym == old_sym:
            return None
        atom.SetAtomicNum(Chem.Atom(new_sym).GetAtomicNum())
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except Exception:
            return None

    def _brics_recombine(self) -> Optional[str]:
        if len(self._brics_fragments) < 2:
            return None
        f1, f2 = random.sample(self._brics_fragments, 2)
        try:
            products = BRICS.BRICSBuild([Chem.MolFromSmiles(f1), Chem.MolFromSmiles(f2)])
            for prod in products:
                if prod is not None:
                    Chem.SanitizeMol(prod)
                    return Chem.MolToSmiles(prod)
        except Exception:
            pass
        return None

    def _lipinski_ok(self, mol: Chem.Mol) -> bool:
        return (
            Descriptors.MolWt(mol) <= 500
            and Lipinski.NumHDonors(mol) <= 5
            and Lipinski.NumHAcceptors(mol) <= 10
            and Descriptors.MolLogP(mol) <= 5
        )

    def _is_valid_smiles(self, sm: str) -> bool:
        mol = Chem.MolFromSmiles(sm)
        return mol is not None and self._lipinski_ok(mol)

    def generate_molecules(self, num_samples: int, valid_only: bool = True) -> List[str]:
        generated: List[str] = []
        attempts = 0
        max_attempts = num_samples * 30
        pool = self.valid_smiles.copy()

        while len(generated) < num_samples and attempts < max_attempts:
            attempts += 1
            strategy = random.choice(["mutate", "brics", "seed"])
            candidate: Optional[str] = None

            if strategy == "brics":
                candidate = self._brics_recombine()
            elif strategy == "mutate":
                parent = Chem.MolFromSmiles(random.choice(pool))
                if parent:
                    mutated = self._rdkit_mutate(parent)
                    if mutated:
                        candidate = Chem.MolToSmiles(mutated)
            else:
                candidate = random.choice(pool)

            if candidate and (not valid_only or self._is_valid_smiles(candidate)):
                generated.append(candidate)
                pool.append(candidate)

        while len(generated) < num_samples:
            generated.append(random.choice(self.valid_smiles))
        return generated[:num_samples]


# ---------------------------------------------------------------------------
# F1 — تولید جیب پروتئینی (PocketGen-compatible، ۵ جیب به‌ازای دارو — FR-04)
# ---------------------------------------------------------------------------
class SyntheticPocketGenerator:
    AMINO_ACIDS = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]
    AA_1LETTER = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }

    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.phi_psi = {
            aa: (random.uniform(-140, 140), random.uniform(-80, 80))
            for aa in self.AMINO_ACIDS
        }

    def _random_sequence(self, length: int) -> str:
        return "".join(random.choices(self.AMINO_ACIDS, k=length))

    def _build_backbone(
        self, sequence: str, offset: Tuple[float, float, float], scale: float
    ) -> Dict:
        atoms = []
        x, y, z = offset
        step = 3.8 * scale
        for i, aa in enumerate(sequence):
            phi, psi = self.phi_psi.get(aa, (0, 0))
            dx = step * np.cos(np.radians(phi))
            dy = step * np.sin(np.radians(phi)) * np.cos(np.radians(psi))
            dz = step * np.sin(np.radians(phi)) * np.sin(np.radians(psi))
            x += dx
            y += dy
            z += dz
            atoms.append({"residue": aa, "index": i + 1, "x": x, "y": y, "z": z})
        n = max(len(sequence), 1)
        center = (x / n, y / n, z / n)
        return {
            "sequence": sequence,
            "atoms": atoms,
            "center": center,
            "length": len(sequence),
        }

    def generate_pocket(
        self, sequence: Optional[str] = None, length: int = 40, variant: int = 0
    ) -> Dict:
        if sequence is None:
            sequence = self._random_sequence(length)
        else:
            seq_1 = "".join(
                self.AA_1LETTER.get(aa.upper(), "A") if len(aa) == 3 else aa.upper()
                for aa in sequence.replace(" ", "").split(",")
            )
            if not seq_1:
                seq_1 = sequence
            padded = (seq_1 + self._random_sequence(max(0, length - len(seq_1))))[:length]
            sequence = "".join(
                self.AMINO_ACIDS[list("ACDEFGHIKLMNPQRSTVWY").index(c)]
                if c in "ACDEFGHIKLMNPQRSTVWY"
                else random.choice(self.AMINO_ACIDS)
                for c in padded
            )

        angle = variant * 0.7
        offset = (variant * 2.1, np.sin(angle) * 3.0, np.cos(angle) * 3.0)
        scale = 0.9 + 0.05 * variant
        return self._build_backbone(sequence[:length], offset, scale)

    def generate_pockets(
        self, sequence: Optional[str] = None, count: int = 5, length: int = 40
    ) -> List[Dict]:
        """FR-04: تولید ۵ جیب متنوع به‌ازای هر دارو."""
        return [
            self.generate_pocket(sequence=sequence, length=length, variant=i)
            for i in range(count)
        ]


# ---------------------------------------------------------------------------
# توصیفگرهای مولکولی
# ---------------------------------------------------------------------------
@dataclass
class MolecularDescriptors:
    MW: float
    LogP: float
    HBD: int
    HBA: int
    RotatableBonds: int
    AromaticRings: int
    TPSA: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.MW, self.LogP, self.HBD, self.HBA,
            self.RotatableBonds, self.AromaticRings, self.TPSA,
        ])

    @staticmethod
    def compute(smiles: str) -> "MolecularDescriptors":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES نامعتبر: {smiles}")
        return MolecularDescriptors(
            MW=Descriptors.MolWt(mol),
            LogP=Descriptors.MolLogP(mol),
            HBD=Lipinski.NumHDonors(mol),
            HBA=Lipinski.NumHAcceptors(mol),
            RotatableBonds=Descriptors.NumRotatableBonds(mol),
            AromaticRings=Descriptors.NumAromaticRings(mol),
            TPSA=Descriptors.TPSA(mol),
        )


def normalize_descriptors(desc_array: np.ndarray) -> np.ndarray:
    default_ranges = {
        0: (0, 800), 1: (-3, 7), 2: (0, 6), 3: (0, 10),
        4: (0, 20), 5: (0, 5), 6: (0, 200),
    }
    normalized = np.zeros_like(desc_array, dtype=float)
    for i, val in enumerate(desc_array):
        min_val, max_val = default_ranges[i]
        if max_val > min_val:
            norm_val = (val - min_val) / (max_val - min_val)
        else:
            norm_val = 0.5
        normalized[i] = np.clip(norm_val, 0.0, 1.0)
    return normalized


# ---------------------------------------------------------------------------
# F4 — مدار VQC منطبق FR-11 با کنترل عمق و انتخاب backend (FR-12, FR-13)
# ---------------------------------------------------------------------------
def count_fr11_gates(n_qubits: int) -> int:
    """RX(n) + CNOT adjacent(n-1) + CNOT skip(n-2) + RY(n)."""
    return n_qubits + max(0, n_qubits - 1) + max(0, n_qubits - 2) + n_qubits


def apply_fr11_layer(params: np.ndarray, features: np.ndarray, n_qubits: int) -> int:
    """سه لایه متوالی FR-11؛ تعداد گیت‌های اعمال‌شده را برمی‌گرداند."""
    gates = 0
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)
        gates += 1
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        gates += 1
    for i in range(n_qubits - 2):
        qml.CNOT(wires=[i, i + 2])
        gates += 1
    for i, f in enumerate(features[:n_qubits]):
        qml.RY(np.arctan(float(f)), wires=i)
        gates += 1
    return gates


class QuantumVQESimulator:
    """VQE با مدار FR-11، COBYLA (FR-16) و انتخاب backend (FR-13)."""

    def __init__(
        self,
        n_qubits: int = 7,
        backend: BackendName = "auto",
        max_depth: int = MAX_CIRCUIT_DEPTH,
    ):
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.backend = self._resolve_backend(backend)
        self.use_quantum = self.backend in (
            "pennylane_default_qubit",
            "qiskit_aer_simulator",
        )
        self.last_gate_depth = 0
        if self.use_quantum:
            self.dev = self._create_device()
            self._circuit = self._build_circuit()

    def _create_device(self):
        if self.backend == "qiskit_aer_simulator" and QISKIT_AER_AVAILABLE:
            return qml.device("qiskit.aer", wires=self.n_qubits, shots=None)
        return qml.device("default.qubit", wires=self.n_qubits)

    @staticmethod
    def available_backends() -> List[str]:
        backends = ["classical_fallback"]
        if PENNYLANE_AVAILABLE:
            backends.append("pennylane_default_qubit")
        if QISKIT_AER_AVAILABLE:
            backends.append("qiskit_aer_simulator")
        backends.append("auto")
        return backends

    @staticmethod
    def _resolve_backend(backend: BackendName) -> str:
        if backend == "auto":
            if PENNYLANE_AVAILABLE:
                return "pennylane_default_qubit"
            return "classical_fallback"
        if backend == "pennylane_default_qubit" and not PENNYLANE_AVAILABLE:
            return "classical_fallback"
        if backend == "qiskit_aer_simulator":
            if QISKIT_AER_AVAILABLE:
                return "qiskit_aer_simulator"
            return "pennylane_default_qubit" if PENNYLANE_AVAILABLE else "classical_fallback"
        return backend

    def _build_circuit(self):
        n = self.n_qubits
        block_gates = count_fr11_gates(n)
        embed_gates = n
        use_full_fr11 = embed_gates + block_gates <= self.max_depth

        @qml.qnode(self.dev)
        def circuit(params, features):
            for i, theta in enumerate(features[:n]):
                qml.RX(np.arctan(float(theta)), wires=i)
            if use_full_fr11:
                apply_fr11_layer(params, features, n)
            else:
                for i in range(n):
                    qml.RX(params[i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.last_gate_depth = (
            embed_gates + block_gates if use_full_fr11 else embed_gates + n + max(0, n - 1)
        )
        return circuit

    def _classical_predict(self, features_normalized: np.ndarray) -> float:
        weights = np.array([0.1, 0.2, -0.15, -0.15, 0.05, 0.05, 0.1])
        raw = -np.dot(features_normalized, weights) * 12
        return float(np.clip(raw, -15.0, -0.1))

    def _optimize_params(self, features_normalized: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            return np.zeros(self.n_qubits)

        def cost(p):
            return float(-self._circuit(p, features_normalized))

        result = minimize(
            cost,
            np.zeros(self.n_qubits),
            method="COBYLA",
            options={"maxiter": COBYLA_MAXITER},
        )
        return result.x

    def predict_affinity(
        self, features_normalized: np.ndarray, optimize: bool = False
    ) -> Tuple[float, int]:
        """خروجی: (انرژی kcal/mol, عمق مدار)."""
        if not self.use_quantum:
            return self._classical_predict(features_normalized), 0

        params = (
            self._optimize_params(features_normalized)
            if optimize
            else np.zeros(self.n_qubits)
        )
        exp_val = float(self._circuit(params, features_normalized))
        energy = -7.5 * (exp_val + 1) / 2 - 0.1
        return float(np.clip(energy, -15.0, -0.1)), self.last_gate_depth


# ---------------------------------------------------------------------------
# F6 — پیش‌بینی کامل با نمره، اطمینان و خروجی ۳D
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    smiles: str
    protein_sequence: str
    binding_energy_kcal_mol: float
    binding_score: float
    confidence_pct: float
    pocket: Dict
    pockets: List[Dict]
    pdb_path: str
    viewer_html_path: str
    gate_depth: int
    backend: str

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "protein_sequence": self.protein_sequence,
            "binding_energy_kcal_mol": self.binding_energy_kcal_mol,
            "binding_score": self.binding_score,
            "confidence_pct": self.confidence_pct,
            "pdb_path": self.pdb_path,
            "viewer_html_path": self.viewer_html_path,
            "gate_depth": self.gate_depth,
            "backend": self.backend,
            "pocket_count": len(self.pockets),
        }


def predict_binding(
    smiles: str,
    fasta: Optional[str] = None,
    output_dir: Optional[str] = None,
    backend: BackendName = "auto",
    num_pockets: int = 5,
) -> PredictionResult:
    """پیش‌بینی یک جفت دارو-پروتئین با خروجی ۳D و نمره ۰–۱۰۰."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"SMILES نامعتبر: {smiles}")

    out = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    desc = MolecularDescriptors.compute(smiles)
    desc_norm = normalize_descriptors(desc.to_array())

    pocket_gen = SyntheticPocketGenerator()
    pockets = pocket_gen.generate_pockets(sequence=fasta, count=num_pockets)
    pocket = pockets[0]

    sim = QuantumVQESimulator(n_qubits=7, backend=backend)
    energy, depth = sim.predict_affinity(desc_norm, optimize=False)
    score = binding_energy_to_score(energy)
    seq = pocket["sequence"]
    confidence = estimate_confidence(desc_norm, len(seq), num_pockets)

    pdb_path = export_pocket_pdb(pocket, str(out / f"pocket_{stamp}.pdb"))
    viewer_path = generate_pocket_viewer_html(
        pocket, score, confidence, str(out / f"viewer_{stamp}.html")
    )

    return PredictionResult(
        smiles=smiles,
        protein_sequence=seq,
        binding_energy_kcal_mol=energy,
        binding_score=score,
        confidence_pct=confidence,
        pocket=pocket,
        pockets=pockets,
        pdb_path=pdb_path,
        viewer_html_path=viewer_path,
        gate_depth=depth,
        backend=sim.backend,
    )


# ---------------------------------------------------------------------------
# خط لوله تولید داده
# ---------------------------------------------------------------------------
class SyntheticDataPipeline:
    def __init__(self, seed_smiles: List[str], random_state: int = 42):
        self.mol_gen = SyntheticMoleculeGenerator(seed_smiles)
        self.pocket_gen = SyntheticPocketGenerator(random_seed=random_state)
        self.quantum_sim = QuantumVQESimulator(
            n_qubits=7,
            backend="auto",
        )
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def generate_pair(self, fasta: Optional[str] = None) -> Dict:
        smiles = self.mol_gen.generate_molecules(1)[0]
        desc = MolecularDescriptors.compute(smiles)
        desc_norm = normalize_descriptors(desc.to_array())
        pockets = self.pocket_gen.generate_pockets(sequence=fasta, count=5)
        pocket = pockets[0]
        energy, depth = self.quantum_sim.predict_affinity(desc_norm, optimize=False)
        score = binding_energy_to_score(energy)

        return {
            "smiles": smiles,
            "protein_sequence": pocket["sequence"],
            "MW": desc.MW,
            "LogP": desc.LogP,
            "HBD": desc.HBD,
            "HBA": desc.HBA,
            "RotatableBonds": desc.RotatableBonds,
            "AromaticRings": desc.AromaticRings,
            "TPSA": desc.TPSA,
            "binding_energy_kcal_mol": energy,
            "binding_score": score,
            "gate_depth": depth,
            "pocket_center_x": pocket["center"][0],
            "pocket_center_y": pocket["center"][1],
            "pocket_center_z": pocket["center"][2],
            "pocket_length": pocket["length"],
            "pocket_count": len(pockets),
        }

    def generate_dataset(
        self,
        num_pairs: int,
        output_csv: str = "synthetic_dataset.csv",
        output_json: Optional[str] = None,
        output_pdf: Optional[str] = None,
        fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        records = []
        print(f"Generating {num_pairs} synthetic samples...")
        for i in range(num_pairs):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Progress: {i + 1}/{num_pairs}")
            try:
                records.append(self.generate_pair(fasta=fasta))
            except Exception as e:
                print(f"Error at sample {i + 1}: {e}")

        df = pd.DataFrame(records)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"Saved JSON: {output_json}")

        txt_path = str(Path(output_csv).with_suffix(".txt"))
        generate_text_report(df, txt_path)

        pdf_path = output_pdf or str(Path(output_csv).with_suffix(".pdf"))
        if len(df) > 0:
            sample = df.iloc[0].to_dict()
            sample["pocket_count"] = int(df["pocket_count"].iloc[0]) if "pocket_count" in df else 5
            sample["protein_length"] = len(str(sample.get("protein_sequence", "")))
            generate_pdf_report(sample, pdf_path, dataset_summary=df)
            print(f"Saved PDF: {pdf_path}")

        print("\n--- Descriptor stats ---")
        print(df[["MW", "LogP", "HBD", "HBA", "binding_score"]].describe())
        return df


def generate_report(df: pd.DataFrame, output_file: str = "data_report.txt") -> str:
    """سازگاری با نسخه قبل — گزارش متنی + PDF."""
    txt = generate_text_report(df, output_file)
    pdf = str(Path(output_file).with_suffix(".pdf"))
    if len(df) > 0:
        row = df.iloc[0].to_dict()
        row["protein_length"] = len(str(row.get("protein_sequence", "")))
        generate_pdf_report(row, pdf, dataset_summary=df)
    return txt


if __name__ == "__main__":
    seed_smiles_list = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)(C)NC(=O)C1=CC=CC=C1C(=O)NC2=CC=C(C=C2)C(F)(F)F",
        "CC1=C(C=C(C=C1)Cl)C2=C(NC(=N2)C3=CC=C(C=C3)S(=O)(=O)N)C4=CC=CC=C4",
        "CN1CCN(CC1)C2=CC=C(C=C2)C3=NC4=CC=CC=C4S3",
    ]

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = predict_binding(
        seed_smiles_list[0],
        fasta="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHRAQLTKL",
        output_dir=str(out_dir),
    )
    print(f"\nPrediction: score={result.binding_score}, confidence={result.confidence_pct}%")
    print(f"3D viewer: {result.viewer_html_path}")
    print(f"PDB: {result.pdb_path}")

    pdf_single = str(out_dir / "prediction_report.pdf")
    generate_pdf_report(result.to_dict(), pdf_single)
    print(f"PDF report: {pdf_single}")

    pipeline = SyntheticDataPipeline(seed_smiles_list, random_state=2025)
    df = pipeline.generate_dataset(
        num_pairs=50,
        output_csv=str(out_dir / "hqca_synthetic_data.csv"),
        output_json=str(out_dir / "hqca_synthetic_data.json"),
        output_pdf=str(out_dir / "hqca_dataset_report.pdf"),
    )
    generate_report(df, str(out_dir / "data_report.txt"))

    print("\nDone.")
