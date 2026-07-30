import sys
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, label
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes

warnings.filterwarnings("ignore")

BIDS_DIR = Path("ISPY2_dataset/derivatives/nifti")
CLINICAL_XLSX = Path("ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx")
MULTIFEAT_XLSX = Path("Multi-feature-MRI-NACT-Data (1).xlsx")
OUTPUT_CSV = Path("results/feature_matrix.csv")
OUTPUT_DIR = Path("results/features")

PATIENT_IDS = [
    "sub-ISPY2-125130",
    "sub-ISPY2-227832",
    "sub-ISPY2-311455",
    "sub-ISPY2-313243",
    "sub-ISPY2-767081",
]
SESSIONS = ["ses-T0", "ses-T1", "ses-T2", "ses-T3"]

BIN_WIDTH = 25


def log(msg):
    print(msg, flush=True)


def load_nifti(path):
    try:
        img = nib.load(str(path))
        return img.get_fdata(dtype=np.float64), img.affine, img.header.get_zooms()[:3]
    except Exception:
        return None, None, None


def get_tumor_mask(mask_data, preferred_label=49):
    binary = (mask_data == preferred_label).astype(np.uint8)
    if binary.sum() == 0:
        binary = (mask_data > 0).astype(np.uint8)
    return binary


def compute_pe_ser(pre, early, late):
    pe = np.zeros_like(pre)
    ser = np.zeros_like(pre)
    valid = pre > 0
    pe[valid] = ((early[valid] - pre[valid]) / pre[valid]) * 100.0
    denom = late - pre
    valid2 = valid & (denom > 0)
    ser[valid2] = (early[valid2] - pre[valid2]) / denom[valid2]
    return pe, ser


def extract_kinetic(patient_id, session):
    base = BIDS_DIR / patient_id / session / "perf"
    ps = f"{patient_id}_{session}"

    pre, _, sp = load_nifti(base / f"{ps}_dce-pre.nii.gz")
    early, _, _ = load_nifti(base / f"{ps}_dce-post-1.nii.gz")
    late, _, _ = load_nifti(base / f"{ps}_dce-post-3.nii.gz")
    if pre is None:
        return {}, None

    mask_path = BIDS_DIR / patient_id / session / "seg" / f"{ps}_mask.nii.gz"
    mask_data, _, sp2 = load_nifti(mask_path)
    if mask_data is None:
        return {}, None

    mask = get_tumor_mask(mask_data)
    if mask.sum() == 0:
        return {}, mask

    pe, ser = compute_pe_ser(pre, early, late)

    ftv = int(((pe >= 70.0) & (ser >= 0.9) & (mask > 0)).sum())

    total = float(mask.sum())
    washout = ((ser > 1.1) & (mask > 0)).sum()
    plateau = ((ser >= 0.9) & (ser <= 1.1) & (mask > 0)).sum()
    persistent = ((ser > 0) & (ser < 0.9) & (mask > 0)).sum()

    pe_in = pe[mask > 0]
    ser_in = ser[mask > 0]

    features = {
        f"FTV_{session}": ftv,
        f"PE_mean_{session}": float(pe_in.mean()) if len(pe_in) > 0 else 0.0,
        f"PE_std_{session}": float(pe_in.std()) if len(pe_in) > 0 else 0.0,
        f"PE_median_{session}": float(np.median(pe_in)) if len(pe_in) > 0 else 0.0,
        f"SER_mean_{session}": float(ser_in.mean()) if len(ser_in) > 0 else 0.0,
        f"SER_std_{session}": float(ser_in.std()) if len(ser_in) > 0 else 0.0,
        f"SER_median_{session}": float(np.median(ser_in)) if len(ser_in) > 0 else 0.0,
        f"washout_frac_{session}": float(washout / total),
        f"plateau_frac_{session}": float(plateau / total),
        f"persistent_frac_{session}": float(persistent / total),
    }
    return features, mask


def extract_shape(patient_id, session):
    mask_path = BIDS_DIR / patient_id / session / "seg" / f"{patient_id}_{session}_mask.nii.gz"
    mask_data, _, spacing = load_nifti(mask_path)
    if mask_data is None:
        return {}

    mask = get_tumor_mask(mask_data)
    if mask.sum() == 0:
        return {}

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    vol = float(mask.sum()) * voxel_vol

    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing, step_size=2)
        sa = 0.0
        for tri in faces:
            a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            sa += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    except Exception:
        sa = 0.0

    spher = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / sa if sa > 0 and vol > 0 else 0.0
    svr = sa / vol if vol > 0 else 0.0
    compact = 1.0 / spher if spher > 0 else 0.0

    coords = np.argwhere(mask > 0)
    cov = np.cov(coords.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1] if cov.shape == (3, 3) else np.ones(3)
    elong = float(np.sqrt(eigvals[1] / eigvals[0])) if eigvals[0] > 0 else 1.0
    flat = float(np.sqrt(eigvals[2] / eigvals[0])) if eigvals[0] > 0 else 1.0
    major = float(4.0 * np.sqrt(eigvals[0])) if eigvals[0] > 0 else 0.0
    minor = float(4.0 * np.sqrt(eigvals[1])) if len(eigvals) > 1 and eigvals[1] > 0 else 0.0

    return {
        f"volume_mm3_{session}": vol,
        f"surface_area_mm2_{session}": float(sa),
        f"sphericity_{session}": spher,
        f"surface_to_volume_ratio_{session}": svr,
        f"compactness_{session}": compact,
        f"elongation_{session}": elong,
        f"flatness_{session}": flat,
        f"major_axis_length_{session}": major,
        f"minor_axis_length_{session}": minor,
    }


def extract_first_order(image, mask_bin):
    voxels = image[mask_bin > 0]
    if len(voxels) == 0:
        return {k: 0.0 for k in ["mean", "variance", "skewness", "kurtosis", "entropy",
                                  "p10", "p50", "p90", "min", "max", "range", "iqr", "mad", "rms"]}

    mean = float(np.mean(voxels))
    std = float(np.std(voxels))
    var = float(np.var(voxels))
    skew = float(np.mean((voxels - mean)**3) / (std**3 + 1e-10))
    kurt = float(np.mean((voxels - mean)**4) / (std**4 + 1e-10)) - 3.0

    hist, _ = np.histogram(voxels, bins=50)
    ent = float(scipy_entropy(hist + 1e-10))

    return {
        "mean": mean, "variance": var, "skewness": skew, "kurtosis": kurt,
        "entropy": ent,
        "p10": float(np.percentile(voxels, 10)),
        "p50": float(np.percentile(voxels, 50)),
        "p90": float(np.percentile(voxels, 90)),
        "min": float(voxels.min()), "max": float(voxels.max()),
        "range": float(voxels.max() - voxels.min()),
        "iqr": float(np.percentile(voxels, 75) - np.percentile(voxels, 25)),
        "mad": float(np.mean(np.abs(voxels - mean))),
        "rms": float(np.sqrt(np.mean(voxels**2))),
    }


def extract_glcm_2d_slices(image, mask_bin):
    contrast_list, corr_list, homo_list, energy_list, diss_list = [], [], [], [], []

    unique_z = np.unique(np.argwhere(mask_bin > 0)[:, 2])
    if len(unique_z) == 0:
        return {"contrast": 0.0, "correlation": 0.0, "homogeneity": 0.0, "energy": 0.0, "dissimilarity": 0.0}

    for z in unique_z:
        sl = image[:, :, z]
        sl_mask = mask_bin[:, :, z]
        if sl_mask.sum() < 5:
            continue

        masked = np.zeros_like(sl)
        masked[sl_mask > 0] = sl[sl_mask > 0]
        min_val = masked[mask_bin[:, :, z] > 0].min()
        disc = np.floor((masked - min_val) / BIN_WIDTH).astype(np.int32)
        disc[disc < 0] = 0
        disc[sl_mask == 0] = 0

        max_g = int(disc.max()) + 1 if disc.max() < 255 else 255
        glcm = graycomatrix(disc.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=min(max_g + 1, 256), symmetric=True, normed=True)

        contrast_list.append(float(graycoprops(glcm, 'contrast').mean()))
        corr_list.append(float(graycoprops(glcm, 'correlation').mean()))
        homo_list.append(float(graycoprops(glcm, 'homogeneity').mean()))
        energy_list.append(float(graycoprops(glcm, 'energy').mean()))
        diss_list.append(float(graycoprops(glcm, 'dissimilarity').mean()))

    return {
        "contrast": float(np.mean(contrast_list)) if contrast_list else 0.0,
        "correlation": float(np.mean(corr_list)) if corr_list else 0.0,
        "homogeneity": float(np.mean(homo_list)) if homo_list else 0.0,
        "energy": float(np.mean(energy_list)) if energy_list else 0.0,
        "dissimilarity": float(np.mean(diss_list)) if diss_list else 0.0,
    }


def extract_glrlm_fast(image, mask_bin):
    voxels = image[mask_bin > 0]
    if len(voxels) < 10:
        return {"SRE": 0.0, "LRHGLE": 0.0}

    disc = np.floor((voxels - voxels.min()) / BIN_WIDTH).astype(np.int32)
    n_g = int(disc.max()) + 1

    runs = []
    sorted_v = np.sort(disc)
    current_val = sorted_v[0]
    run_len = 1
    for v in sorted_v[1:]:
        if v == current_val:
            run_len += 1
        else:
            runs.append((current_val, run_len))
            current_val = v
            run_len = 1
    runs.append((current_val, run_len))

    rlm_mat = np.zeros((max(n_g, 1), 100))
    for g, rl in runs:
        col = min(rl - 1, 99)
        rlm_mat[g, col] += 1

    total = rlm_mat.sum()
    if total == 0:
        return {"SRE": 0.0, "LRHGLE": 0.0}

    rls = np.arange(1, 101, dtype=np.float64)
    gls = np.arange(n_g, dtype=np.float64) + 1

    sre = float(np.sum(rlm_mat / (rls ** 2)))
    lrhgle = float(np.sum(rlm_mat * (gls[:, None] ** 2) * (rls[None, :] ** 2)))

    return {"SRE": sre / total, "LRHGLE": lrhgle / total}


def extract_glszm_fast(image, mask_bin):
    if mask_bin.sum() < 10:
        return {"ZoneEntropy": 0.0, "LargeZoneEmphasis": 0.0}

    struct = np.ones((3, 3, 3), dtype=bool)
    labeled, n_labels = label(mask_bin, structure=struct)
    if n_labels == 0:
        return {"ZoneEntropy": 0.0, "LargeZoneEmphasis": 0.0}

    disc = np.floor((image - image.min()) / BIN_WIDTH).astype(np.int32)

    szm = {}
    for lid in range(1, n_labels + 1):
        zone = labeled == lid
        size = int(zone.sum())
        g = int(disc[zone].mean())
        szm[(g, size)] = szm.get((g, size), 0) + 1

    total = sum(szm.values())
    if total == 0:
        return {"ZoneEntropy": 0.0, "LargeZoneEmphasis": 0.0}

    lrze = sum(cnt * (size ** 2) for (_, size), cnt in szm.items())
    ent = -sum((cnt / total) * np.log2(cnt / total) for cnt in szm.values())

    return {"ZoneEntropy": ent, "LargeZoneEmphasis": lrze / total}


def extract_radiomics(image, mask_bin, session, series_id):
    f = {}
    prefix = f"{series_id}_{session}"

    fo = extract_first_order(image, mask_bin)
    for k, v in fo.items():
        f[f"firstorder_{k}_{prefix}"] = v

    glcm = extract_glcm_2d_slices(image, mask_bin)
    for k, v in glcm.items():
        f[f"glcm_{k}_{prefix}"] = v

    glrlm = extract_glrlm_fast(image, mask_bin)
    for k, v in glrlm.items():
        f[f"glrlm_{k}_{prefix}"] = v

    glszm = extract_glszm_fast(image, mask_bin)
    for k, v in glszm.items():
        f[f"glszm_{k}_{prefix}"] = v

    return f


def load_clinical(patient_ids):
    df = pd.read_excel(CLINICAL_XLSX)
    id_map = {p.replace("sub-ISPY2-", ""): p for p in patient_ids}
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    mask = df["Patient_ID"].isin(id_map.keys())
    df = df[mask].copy()
    df["Patient_ID"] = df["Patient_ID"].map(id_map)
    return df.set_index("Patient_ID")


def load_multifeat(patient_ids):
    df = pd.read_excel(MULTIFEAT_XLSX)
    id_map = {p.replace("sub-ISPY2-", ""): p for p in patient_ids}
    df["CLINICAL-TRIAL-SUBJECT-ID"] = df["CLINICAL-TRIAL-SUBJECT-ID"].astype(str)
    mask = df["CLINICAL-TRIAL-SUBJECT-ID"].isin(id_map.keys())
    df = df[mask].copy()
    df["Patient_ID"] = df["CLINICAL-TRIAL-SUBJECT-ID"].map(id_map)
    return df.set_index("Patient_ID")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Extraction des features ISPY-2")
    log("=" * 60)

    log("\n[1] Chargement des données cliniques...")
    df_clinical = load_clinical(PATIENT_IDS)
    log(f"  {len(df_clinical)} patients - {list(df_clinical.columns)}")

    log("\n[2] Chargement des features multi (excels pré-computés)...")
    df_multi = load_multifeat(PATIENT_IDS)
    log(f"  {len(df_multi)} patients - {list(df_multi.columns)}")

    all_rows = []
    for patient_id in PATIENT_IDS:
        log(f"\n{'=' * 50}")
        log(f"Patient: {patient_id}")
        row = {"Patient_ID": patient_id}

        if patient_id in df_clinical.index:
            for c in df_clinical.columns:
                row[f"clinical_{c}"] = df_clinical.loc[patient_id, c]

        if patient_id in df_multi.index:
            for c in df_multi.columns:
                if c not in ("Patient_ID", "CLINICAL-TRIAL-SUBJECT-ID"):
                    row[f"multifeat_{c}"] = df_multi.loc[patient_id, c]
            log(f"  Multi-features chargées")

        for session in SESSIONS:
            log(f"  [{session}]")
            kin_feat, mask = extract_kinetic(patient_id, session)
            row.update(kin_feat)
            n_kin = len(kin_feat)
            log(f"    Cinétique: {n_kin} features")

            shape_feat = extract_shape(patient_id, session)
            row.update(shape_feat)
            n_shape = len(shape_feat)
            if n_shape > 0:
                vol_key = f"volume_mm3_{session}"
                log(f"    Forme: {n_shape} features (volume={shape_feat.get(vol_key, 0):.1f} mm³)")

            base = BIDS_DIR / patient_id / session / "perf"
            ps = f"{patient_id}_{session}"
            img_data, _, _ = load_nifti(base / f"{ps}_dce-post-1.nii.gz")
            if img_data is not None and mask is not None and mask.sum() > 0:
                radio_feat = extract_radiomics(img_data, mask, session, "dce-post-1")
                row.update(radio_feat)
                log(f"    Radiomique: {len(radio_feat)} features")
            else:
                log(f"    Radiomique: ignoré (pas de masque/imagerie)")

        all_rows.append(row)

    df = pd.DataFrame(all_rows).set_index("Patient_ID")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV)

    log(f"\n{'=' * 60}")
    log(f"Sauvegardé: {OUTPUT_CSV}")
    log(f"Matrice: {df.shape[0]} patients × {df.shape[1]} features")
    log(f"{'=' * 60}")

    info = {
        "n_patients": len(PATIENT_IDS),
        "n_features": int(df.shape[1]),
        "feature_names": list(df.columns),
        "patients": PATIENT_IDS,
        "sessions": SESSIONS,
        "clinical_source": str(CLINICAL_XLSX),
        "multifeature_source": str(MULTIFEAT_XLSX),
        "features_computed": ["PE/SER maps", "FTV", "kinetic sub-volumes",
                              "shape (volume, surface, sphericity, elongation, compactness)",
                              "first-order radiomics", "GLCM", "GLRLM", "GLSZM"],
    }
    (OUTPUT_DIR / "extraction_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("\nColonnes disponibles:")
    for c in df.columns:
        log(f"  - {c}")

    return df


if __name__ == "__main__":
    main()
