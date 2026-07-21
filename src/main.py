###########################################
# POINT D'ENTREE
###########################################

from ispy2_extractor import ISPY2Extractor
from pathlib import Path

PROJECT_ROOT = Path("C:/Data_Stage")

ispy = ISPY2Extractor(project_root= PROJECT_ROOT, sample_size=20, random_state=42)

ispy.pipeline()