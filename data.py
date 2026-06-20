#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول کامل تولید داده سنتتیک برای سامانه HQCA
عنوان اختراع: سامانه ترکیبی شبیه‌ساز کوآنتوم-کلاسیک برای پیش‌بینی برهم‌کنش مولکولی
نسخه: 2.0 (تولید انبوه داده برای آموزش مدل QML)
تاریخ: ۱۴۰۵/۰۳/۲۰
"""

import os
import json
import logging
import random
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for container and CI friendly structured logs."""

    EXTRA_FIELDS = {
        "event",
        "sample_index",
        "num_pairs",
        "output_csv",
        "output_json",
        "output_metrics",
        "output_report",
        "records_generated",
        "records_failed",
        "duration_seconds",
        "rdkit_available",
        "pennylane_available",
        "use_quantum",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    log_level = os.getenv("HQCA_LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("HQCA_LOG_FORMAT", "json").lower()
    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    logger = logging.getLogger("hqca")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def ensure_parent_directory(path: Optional[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

# ===============================
# بررسی دسترسی به کتابخانه‌های تخصصی
# ===============================
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Lipinski
    if os.getenv("HQCA_RDKIT_LOGS", "disabled").lower() == "disabled":
        RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    raise ImportError("لطفاً RDKit را نصب کنید: pip install rdkit")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    LOGGER.warning(
        "PennyLane در دسترس نیست. از شبیه‌ساز کلاسیک جایگزین استفاده می‌شود.",
        extra={"event": "optional_dependency_missing", "pennylane_available": False},
    )

# ===============================
# 1. کلاس تولید کننده مولکول‌های دارویی سنتتیک
# ===============================
class SyntheticMoleculeGenerator:
    """
    تولید کننده مولکول‌های جدید با استفاده از روش جهش (mutation) و کراس‌اور (crossover)
    بر اساس SMILES اولیه. این روش برای اهداف ثبت اختراع به عنوان جایگزینی برای ChemBFN/RLL عمل می‌کند.
    """
    def __init__(self, seed_smiles_list: List[str]):
        self.seed_smiles = list(set(seed_smiles_list))  # حذف تکراری‌ها
        self.valid_mols = [Chem.MolFromSmiles(s) for s in self.seed_smiles if Chem.MolFromSmiles(s) is not None]
        self.valid_smiles = [s for s,m in zip(self.seed_smiles, self.valid_mols) if m is not None]
        if len(self.valid_smiles) == 0:
            raise ValueError("هیچ SMILES معتبری در بذر اولیه وجود ندارد.")
        
        # کاراکترهای مجاز در SMILES (برای جهش)
        self.allowed_chars = set('ABCDEFGHIKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=#()[]+-.')
    
    def _mutate(self, smiles: str, mutation_rate: float = 0.1) -> str:
        """جهش تصادفی در سطح کاراکتر"""
        if random.random() > mutation_rate:
            return smiles
        chars = list(smiles)
        pos = random.randint(0, len(chars)-1)
        new_char = random.choice(list(self.allowed_chars))
        chars[pos] = new_char
        return ''.join(chars)
    
    def _crossover(self, sm1: str, sm2: str) -> str:
        """ترکیب دو SMILES با برش در نقطه تصادفی"""
        if len(sm1) < 3 or len(sm2) < 3:
            return sm1
        pos1 = random.randint(1, len(sm1)-1)
        pos2 = random.randint(1, len(sm2)-1)
        child = sm1[:pos1] + sm2[pos2:]
        return child
    
    def _is_valid_smiles(self, sm: str) -> bool:
        """بررسی اعتبار SMILES با RDKit"""
        mol = Chem.MolFromSmiles(sm)
        return mol is not None
    
    def generate_molecules(self, num_samples: int, valid_only: bool = True) -> List[str]:
        """
        تولید تعداد مشخصی مولکول سنتتیک (تضمین اعتبار در صورت valid_only=True)
        """
        generated = []
        attempts = 0
        max_attempts = num_samples * 10
        pool = self.valid_smiles.copy()
        
        while len(generated) < num_samples and attempts < max_attempts:
            parent1 = random.choice(pool)
            parent2 = random.choice(pool)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child, mutation_rate=0.15)
            if valid_only and self._is_valid_smiles(child):
                generated.append(child)
                pool.append(child)
            elif not valid_only:
                generated.append(child)
            attempts += 1
        # اگر به تعداد کافی نرسید، از بذرهای اولیه کپی کن
        while len(generated) < num_samples:
            generated.append(random.choice(self.valid_smiles))
        return generated[:num_samples]

# ===============================
# 2. تولید جیب پروتئینی سنتتیک (ساختار سه‌بعدی ساده)
# ===============================
class SyntheticPocketGenerator:
    """
    تولید جیب پروتئین مصنوعی بر اساس توالی اسید آمینه و هندسه تصادفی.
    برای ثبت اختراع، این کلاس قابلیت جایگزینی با PocketGen واقعی را دارد.
    """
    AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    AA_3LETTER = {aa[:1].upper(): aa for aa in AMINO_ACIDS}
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        # زوایای تصادفی برای هر اسید آمینه (فقط برای نمونه)
        self.phi_psi = {aa: (random.uniform(-180,180), random.uniform(-180,180)) for aa in self.AMINO_ACIDS}
    
    def _random_sequence(self, length: int) -> str:
        return ''.join(random.choices(self.AMINO_ACIDS, k=length))
    
    def generate_pocket(self, sequence: Optional[str] = None, length: int = 40) -> Dict:
        """
        تولید یک جیب با توالی داده شده یا تصادفی.
        خروجی شامل: توالی، فهرست اتم‌ها با مختصات (x,y,z) و مرکز جیب.
        """
        if sequence is None:
            sequence = self._random_sequence(length)
        else:
            sequence = sequence[:length] + self._random_sequence(max(0, length - len(sequence)))
            sequence = sequence[:length]
        
        atoms = []
        x, y, z = 0.0, 0.0, 0.0
        step = 3.8  # آنگستروم
        for i, aa in enumerate(sequence):
            phi, psi = self.phi_psi.get(aa, (0,0))
            dx = step * np.cos(np.radians(phi))
            dy = step * np.sin(np.radians(phi)) * np.cos(np.radians(psi))
            dz = step * np.sin(np.radians(phi)) * np.sin(np.radians(psi))
            x += dx
            y += dy
            z += dz
            atoms.append({
                'residue': aa,
                'index': i+1,
                'x': x, 'y': y, 'z': z
            })
        center = (x/len(sequence), y/len(sequence), z/len(sequence))
        return {
            'sequence': sequence,
            'atoms': atoms,
            'center': center,
            'length': len(sequence)
        }

# ===============================
# 3. محاسبه توصیفگرهای مولکولی (شامل ۷ ویژگی اصلی)
# ===============================
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
        return np.array([self.MW, self.LogP, self.HBD, self.HBA, self.RotatableBonds, self.AromaticRings, self.TPSA])
    
    @staticmethod
    def compute(smiles: str) -> 'MolecularDescriptors':
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES نامعتبر: {smiles}")
        return MolecularDescriptors(
            MW = Descriptors.MolWt(mol),
            LogP = Descriptors.MolLogP(mol),
            HBD = Lipinski.NumHDonors(mol),
            HBA = Lipinski.NumHAcceptors(mol),
            RotatableBonds = Descriptors.NumRotatableBonds(mol),
            AromaticRings = Descriptors.NumAromaticRings(mol),
            TPSA = Descriptors.TPSA(mol)
        )

def normalize_descriptors(desc_array: np.ndarray, stats: Optional[Dict] = None) -> np.ndarray:
    """
    نرمالیزه کردن به بازه [0,1] با استفاده از آمار پیش‌فرض یا محاسبه شده.
    """
    default_ranges = {
        0: (0, 800),    # MW
        1: (-3, 7),     # LogP
        2: (0, 6),      # HBD
        3: (0, 10),     # HBA
        4: (0, 20),     # RotatableBonds
        5: (0, 5),      # AromaticRings
        6: (0, 200)     # TPSA
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

# ===============================
# 4. شبیه‌سازی انرژی اتصال با استفاده از VQE (PennyLane)
# ===============================
class QuantumVQESimulator:
    """
    محاسبه انرژی آزاد اتصال با الگوریتم VQE روی ۷ کیوبیت.
    در صورت نبود PennyLane از یک مدل رگرسیون ساده استفاده می‌کند.
    """
    def __init__(self, n_qubits: int = 7, use_quantum: bool = True):
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and PENNYLANE_AVAILABLE
        LOGGER.info(
            "Quantum simulator initialized.",
            extra={
                "event": "quantum_simulator_initialized",
                "pennylane_available": PENNYLANE_AVAILABLE,
                "use_quantum": self.use_quantum,
            },
        )
        if self.use_quantum:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self._circuit = self._build_circuit()
    
    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(params, features):
            # Angle Embedding
            for i, theta in enumerate(features):
                qml.RX(np.arctan(theta), wires=i)
            # لایه وردشی: چرخش RY با پارامترهای قابل یادگیری
            for i, p in enumerate(params):
                qml.RY(p, wires=i % self.n_qubits)
            # درهم‌تنیدگی درخت دودویی
            step = 1
            while step < self.n_qubits:
                for i in range(0, self.n_qubits - step, step*2):
                    if i+step < self.n_qubits:
                        qml.CNOT(wires=[i, i+step])
                step *= 2
            # اندازه‌گیری مقدار انتظاری Z روی کیوبیت اول
            return qml.expval(qml.PauliZ(0))
        return circuit
    
    def predict_affinity(self, features_normalized: np.ndarray, optimize: bool = False) -> float:
        """
        ورودی: بردار نرمالیزه شده توصیفگرها (طول ۷)
        خروجی: انرژی اتصال (کیلوکالری بر مول) در محدوده [-15, 0] (مقدار منفی نشان‌دهنده اتصال پایدار)
        """
        if not self.use_quantum:
            # مدل جایگزین کلاسیک: ترکیب خطی با وزن‌های ثابت
            weights = np.array([0.1, 0.2, -0.15, -0.15, 0.05, 0.05, 0.1])
            raw = -np.dot(features_normalized, weights) * 12  # مقیاس دهی
            return np.clip(raw, -15.0, -0.1)
        
        # حالت کوانتومی
        if optimize:
            # بهینه‌سازی پارامترها با COBYLA (در عمل فقط یکبار روی داده آموزش می‌بینیم)
            params = np.random.randn(self.n_qubits) * 0.1
            opt = qml.optimize.COBYLA(maxiter=50)
            def cost(p):
                return -self._circuit(p, features_normalized)  # ماکزیمم کردن expectation
            best_params = opt.step(cost, params)
            exp_val = self._circuit(best_params, features_normalized)
        else:
            # استفاده از پارامترهای پیش‌فرض (صفر)
            params = np.zeros(self.n_qubits)
            exp_val = self._circuit(params, features_normalized)
        
        # نگاشت expectation در [-1,1] به انرژی در [-15, -0.1]
        energy = -7.5 * (exp_val + 1) / 2 - 0.1
        return np.clip(energy, -15.0, -0.1)

# ===============================
# 5. خط لوله اصلی تولید داده سنتتیک
# ===============================
class SyntheticDataPipeline:
    """
    تولید مجموعه داده کامل شامل:
    - SMILES دارو
    - توالی پروتئین
    - توصیفگرهای مولکولی (خام و نرمال)
    - انرژی اتصال شبیه‌سازی شده با VQE
    - مختصات مرکز جیب (برای نمایش سه‌بعدی)
    """
    def __init__(self, seed_smiles: List[str], random_state: int = 42):
        self.mol_gen = SyntheticMoleculeGenerator(seed_smiles)
        self.pocket_gen = SyntheticPocketGenerator(random_seed=random_state)
        self.quantum_sim = QuantumVQESimulator(n_qubits=7, use_quantum=PENNYLANE_AVAILABLE)
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def generate_pair(self) -> Dict:
        """تولید یک جفت (دارو، پروتئین) با برچسب انرژی"""
        # تولید مولکول جدید (تک نمونه)
        smiles = self.mol_gen.generate_molecules(1)[0]
        # محاسبه توصیفگرها
        desc = MolecularDescriptors.compute(smiles)
        desc_array = desc.to_array()
        desc_norm = normalize_descriptors(desc_array)
        # تولید جیب پروتئین تصادفی
        pocket = self.pocket_gen.generate_pocket(length=random.randint(30, 60))
        # محاسبه انرژی اتصال با VQE
        binding_energy = self.quantum_sim.predict_affinity(desc_norm, optimize=False)
        
        return {
            'smiles': smiles,
            'protein_sequence': pocket['sequence'],
            'MW': desc.MW,
            'LogP': desc.LogP,
            'HBD': desc.HBD,
            'HBA': desc.HBA,
            'RotatableBonds': desc.RotatableBonds,
            'AromaticRings': desc.AromaticRings,
            'TPSA': desc.TPSA,
            'binding_energy_kcal_mol': binding_energy,
            'pocket_center_x': pocket['center'][0],
            'pocket_center_y': pocket['center'][1],
            'pocket_center_z': pocket['center'][2],
            'pocket_length': pocket['length']
        }
    
    def generate_dataset(self, num_pairs: int, output_csv: str = "synthetic_dataset.csv",
                         output_json: Optional[str] = None,
                         output_metrics: Optional[str] = None) -> pd.DataFrame:
        """
        تولید num_pairs جفت و ذخیره در فایل CSV و اختیاری JSON.
        """
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        records = []
        failed_records = 0
        LOGGER.info(
            "Starting synthetic dataset generation.",
            extra={
                "event": "dataset_generation_started",
                "num_pairs": num_pairs,
                "output_csv": output_csv,
                "output_json": output_json,
                "output_metrics": output_metrics,
            },
        )
        for i in range(num_pairs):
            if (i+1) % 100 == 0 or i == 0:
                LOGGER.info(
                    "Synthetic dataset generation progress.",
                    extra={
                        "event": "dataset_generation_progress",
                        "sample_index": i + 1,
                        "num_pairs": num_pairs,
                    },
                )
            try:
                rec = self.generate_pair()
                records.append(rec)
            except Exception as e:
                failed_records += 1
                LOGGER.exception(
                    f"خطا در تولید نمونه {i+1}: {e}",
                    extra={"event": "dataset_generation_sample_failed", "sample_index": i + 1},
                )
                continue
        
        ensure_parent_directory(output_csv)
        ensure_parent_directory(output_json)
        ensure_parent_directory(output_metrics)

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        LOGGER.info(
            "Synthetic dataset CSV written.",
            extra={
                "event": "dataset_csv_written",
                "records_generated": len(df),
                "records_failed": failed_records,
                "output_csv": output_csv,
            },
        )
        
        if output_json:
            # تبدیل به فرمت JSON قابل خواندن
            with open(output_json, 'w') as f:
                json.dump(records, f, indent=2, default=float)
            LOGGER.info(
                "Synthetic dataset JSON written.",
                extra={"event": "dataset_json_written", "output_json": output_json},
            )

        duration_seconds = round(time.perf_counter() - start_time, 3)
        metrics = {
            "started_at": started_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "duration_seconds": duration_seconds,
            "num_pairs_requested": num_pairs,
            "records_generated": int(len(df)),
            "records_failed": int(failed_records),
            "rdkit_available": RDKIT_AVAILABLE,
            "pennylane_available": PENNYLANE_AVAILABLE,
            "use_quantum": bool(self.quantum_sim.use_quantum),
            "output_csv": output_csv,
            "output_json": output_json,
        }
        if output_metrics:
            with open(output_metrics, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            LOGGER.info(
                "Synthetic dataset metrics written.",
                extra={"event": "dataset_metrics_written", "output_metrics": output_metrics},
            )
        
        # نمایش آمار ساده
        if not df.empty:
            LOGGER.info(
                "Synthetic dataset generation completed.",
                extra={
                    "event": "dataset_generation_completed",
                    "records_generated": len(df),
                    "records_failed": failed_records,
                    "duration_seconds": duration_seconds,
                },
            )
            if os.getenv("HQCA_LOG_FORMAT", "json").lower() != "json":
                print("\n--- آمار توصیفگرها ---")
                print(df[['MW', 'LogP', 'HBD', 'HBA', 'binding_energy_kcal_mol']].describe())
        else:
            LOGGER.warning(
                "Synthetic dataset generation completed without records.",
                extra={
                    "event": "dataset_generation_empty",
                    "records_generated": 0,
                    "records_failed": failed_records,
                    "duration_seconds": duration_seconds,
                },
            )
        
        return df

# ===============================
# 6. ابزارهای کمکی برای مستندات اختراع
# ===============================
def generate_report(df: pd.DataFrame, output_file: str = "data_report.txt"):
    """تولید گزارش آماری از داده‌های تولید شده برای پیوست اختراع"""
    ensure_parent_directory(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("گزارش تولید داده سنتتیک برای سامانه HQCA\n")
        f.write("========================================\n")
        f.write(f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"تعداد نمونه‌ها: {len(df)}\n\n")
        f.write("محدوده توصیفگرها:\n")
        for col in ['MW', 'LogP', 'HBD', 'HBA', 'RotatableBonds', 'AromaticRings', 'TPSA', 'binding_energy_kcal_mol']:
            if col in df.columns:
                f.write(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n")
        if 'binding_energy_kcal_mol' in df.columns:
            f.write("\nتوزیع انرژی اتصال:\n")
            f.write(df['binding_energy_kcal_mol'].value_counts(bins=10).to_string())
    LOGGER.info("Report written.", extra={"event": "report_written", "output_report": output_file})


def get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning(
            f"Invalid integer for {name}; using default {default}.",
            extra={"event": "invalid_environment_value"},
        )
        return default

# ===============================
# 7. اجرای نمونه (در صورت اجرای مستقیم)
# ===============================
if __name__ == "__main__":
    # بذر اولیه SMILES (چند داروی شناخته شده)
    seed_smiles_list = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",   # ایبوپروفن
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # کافئین
        "CC(C)(C)NC(=O)C1=CC=CC=C1C(=O)NC2=CC=C(C=C2)C(F)(F)F",  # آتورواستاتین
        "CC1=C(C=C(C=C1)Cl)C2=C(NC(=N2)C3=CC=C(C=C3)S(=O)(=O)N)C4=CC=CC=C4",  # لوزارتان
        "CN1CCN(CC1)C2=CC=C(C=C2)C3=NC4=CC=CC=C4S3"  # کوتیرون
    ]
    
    random_seed = get_int_env("HQCA_RANDOM_SEED", 2025)
    num_pairs = get_int_env("HQCA_NUM_PAIRS", 500)
    output_csv = os.getenv("HQCA_OUTPUT_CSV", "hqca_synthetic_data_500.csv")
    output_json = os.getenv("HQCA_OUTPUT_JSON", "hqca_synthetic_data_500.json")
    output_metrics = os.getenv("HQCA_METRICS_FILE", "hqca_metrics.json")
    output_report = os.getenv("HQCA_REPORT_FILE", "data_report.txt")

    # ایجاد خط لوله
    pipeline = SyntheticDataPipeline(seed_smiles_list, random_state=random_seed)

    # تولید داده‌ها با پارامترهای قابل تنظیم از محیط اجرا
    df = pipeline.generate_dataset(
        num_pairs=num_pairs,
        output_csv=output_csv,
        output_json=output_json,
        output_metrics=output_metrics,
    )

    # تولید گزارش
    generate_report(df, output_report)

    LOGGER.info(
        "Synthetic data generation finished successfully.",
        extra={
            "event": "main_completed",
            "records_generated": len(df),
            "output_csv": output_csv,
            "output_json": output_json,
            "output_metrics": output_metrics,
        },
    )