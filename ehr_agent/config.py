"""Project paths and global settings (environment variables are read here only)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_DIR.parent.parent  # AI-mnesis repository root (.env, certs/)
DATA_DIR = PROJECT_DIR / "data"  # patient data and derived data (not in git)
MODELS_DIR = PROJECT_DIR / "models"  # local model weights (not in git)
RUNS_DIR = PROJECT_DIR / "runs"  # experiment logs and reports
CERTS_DIR = Path(os.environ.get("EHR_CERTS_DIR") or REPO_ROOT / "certs")  # Russian CA chain for the GigaChat API


def ref_policy() -> str:
    """Reference-interval policy for lab values.

    'layered'    — lab blank → guideline decision limit → NHANES by sex and age → indicative catalog;
    'indicative' — catalog intervals only (the policy of the exploratory runs).
    """
    return os.environ.get("EHR_REF_POLICY", "indicative").strip().lower()


def dense_enabled() -> bool:
    """Dense retrieval needs the local encoder weights (models/multilingual-e5-small); EHR_DENSE=0 disables it."""
    return os.environ.get("EHR_DENSE", "1") != "0" and (MODELS_DIR / "multilingual-e5-small" / "model.safetensors").exists()
