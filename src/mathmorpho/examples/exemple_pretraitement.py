"""
Exemple : prétraitement d'un masque de segmentation tumorale avec mathmorpho.

Ce script simule un volume 3D (pour ne pas dépendre d'un vrai fichier NIfTI),
applique une ouverture morphologique, puis génère un rapport statistique
comparatif complet.
"""

import numpy as np

from mathmorpho import NiftiIO, MathMorphology, MorphologyStats

# --- 1. Simuler un masque de segmentation (à remplacer par un vrai NIfTI) ---
volume = np.zeros((50, 50, 50), dtype=np.uint8)
volume[15:35, 15:35, 15:35] = 1     # region tumorale principale
volume[2, 2, 2] = 1                  # artefact isole (bruit)
volume[25, 25, 25] = 0               # petit trou dans la region

affine = np.eye(4)
NiftiIO.sauvegarder_nifti(volume, affine, "exemple_masque.nii.gz")

# --- 2. Charger et transformer ---
volume_charge, affine, header = NiftiIO.charger_nifti("exemple_masque.nii.gz")

morpho = MathMorphology(volume_charge)
volume_propre = morpho.ouverture(forme="ball", rayon=1)
volume_final = MathMorphology(volume_propre).fermeture(forme="ball", rayon=1)

# --- 3. Statistiques comparatives ---
stats = MorphologyStats(volume_charge, volume_final, save_fig=True, dossier_sortie="figures")
stats.rapport_complet()

# --- 4. Sauvegarde du resultat ---
NiftiIO.sauvegarder_nifti(volume_final, affine, "exemple_masque_clean.nii.gz", header)

print("\nExemple termine. Fichiers generes :")
print("- exemple_masque.nii.gz")
print("- exemple_masque_clean.nii.gz")
print("- figures/ (histogrammes et coupes comparatives)")
