"""Point d'entrée : extraction du dataset ISPY2 depuis TCIA."""

from pathlib import Path

from utils.ispy2_extractor import ISPY2Extractor

PROJECT_ROOT = Path(".")

ispy = ISPY2Extractor(project_root=PROJECT_ROOT, sample_size=5, random_state=42)
ispy.pipeline()