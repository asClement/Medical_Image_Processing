###########################################
# POINT D'ENTREE
###########################################

from ispy2_extractor import ISPY2Extractor
from pathlib import Path

PROJECT_ROOT = Path(".")

ispy = ISPY2Extractor(project_root= PROJECT_ROOT, sample_size=5, random_state=42)

ispy.pipeline()