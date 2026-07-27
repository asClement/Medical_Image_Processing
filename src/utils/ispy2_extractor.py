"""
Code de définition de la classe ISPY2Extractor

La classe ISPY2Extractor va permettre à tout utilisateur de :
    - Choisir un sous-ensemble reproductible (selon une graine aléatoire donné) de la collection;
    - Télécharger les données DICOM de ce sous-ensemble de patients;
    - Les convertir en NIfTI (format .nii.gz) dans un dossier dédié en suivant une architecture adéquate (standard BIDS);
    - Faire une première exploration des données téléchargées.
"""

from pathlib import Path
import json, logging, datetime, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread, Dataset
import dicom2nifti
import nibabel as nib
from tcia_utils import nbia
import regex as re
from operator import itemgetter
import os, shutil
import gzip


#Collection TCIA concernée : ISPY2 (Investigation of Serial Studies to Predict Your Anticancer Response with Imaging and Molecular Analysis 2)
ISPY2_COLLECTION = 'ISPY2'

#Classe
class ISPY2Extractor:
    """
    """

    def __init__(self, project_root : Path, sample_size : int, random_state : int):
        """
        Classe permettant de choisir un sous-ensemble reproductible (selon une graine aléatoire donné) de la collection ISPY2 de l'API TCIA,
        télécharger les données DICOM de ce sous-ensemble de patientes, les convertir en NIfTI (format .nii.gz) dans un dossier dédié en suivant 
        une architecture adéquate (standard BIDS), faire une première exploration des données téléchargées.

        Args:
        ---------
            ``project_root`` : chemin local absolu du projet (créez un dossier sur votre PC et renseignez son chemin absolu)
            ``sample_size`` : taille de l'échantillon  
            ``random_state`` : graine aléatoire à utiliser pour votre sélection
        """

        self.project_root = project_root
        self.sample_size = sample_size
        self.random_state = random_state

        #Création du dossier de stockage des DICOM
        self.raw_dicom = self.project_root / "raw_dicom"
        self.raw_dicom.mkdir(parents=True, exist_ok=True)

        #Création du dossier de stockage des NIfTI
        self.bids_dir = self.project_root / "ISPY2_dataset"
        self.bids_dir.mkdir(parents=True, exist_ok=True)

        self.sample_patients= None  #Liste des patientes sélectionnées après échantillonnage


    def info(self):
        """
        Affiche un ensemble d'informations utiles sur la collection TCIA ISPY2
        """

        #Recuperation des collections TCIA 
        print("Récupération des collections TCIA…")
        collections = nbia.getCollections(format='df')
        print("Récupération terminée !")

        #Recherche de la collection ISPY2
        print("\nRecherche de la collection ISPY2…")

        #Proctection contre les erreurs de collection non trouvée
        try:
            ispy2_collection = collections[collections['Collection']== ISPY2_COLLECTION]
        except :
            raise NameError(f"Erreur ! La collection {ISPY2_COLLECTION} n'existe peut-être pas. Vérifiez l'orthographe du nom de la collection")
        else: 
            print("Collection ISPY2 trouvée !")

            #Nombre de patientes
            counts = nbia.getCollectionPatientCounts(format='df')
            print(f"Nombre de patients de la collection : {counts[counts['Collection']== ISPY2_COLLECTION]['Count'].values[0]} patientes")

            #Recherche de la description de la collection et de l’URL officiel de la documentation
            infos = nbia.getCollectionDescriptions(format="df")
            info, uri = infos[infos['collectionName']==ISPY2_COLLECTION]['description'].values[0], infos[infos['collectionName']==ISPY2_COLLECTION]['descriptionURI'].values[0]

            #Suppression des balises html et et espaces multiples
            info= re.sub(r"<[^>]+>", " ", info)
            info= re.sub(r"\s+", " ", info)
            info.strip()

            #Resumé
            print("Description de la collection :")
            print(f"\t-Description (en anglais) : {info}")
            print(f"\t-URL officielle de la documentation : {uri}")

        finally:
            print("Fin de cette partie")
            print("="*70)
            print()


    def select_sample_patients(self, sample_size : int = 20) -> None:
        """
        Sélectionne un enchantillon aléatoire de patientes de taille donnée sur l'ensemble des patientes de la collection

        Args
        --------------
        sample_size : int
            Taille de l'échantillon

        Returns
        --------------
        None
        """

        #Graine aléatoire pour la sélection
        np.random.seed(self.random_state)

        #Récupération de la liste des PatientIDs de la collection
        #Sélection avec np.randon.choice avec graine et tri pour reproductibilité
        print("Récupération de la liste des IDs patient…")
        ids = []
        for patient in nbia.getPatient(collection= ISPY2_COLLECTION):
            ids.append(patient['PatientId'])
        print("Récupération terminée !")
        print()

        print("Selection en cours...")
        print(f"Taille de l'échantillon : {sample_size} patientes; Graine aléatoire : {self.random_state}")
        print()

        #Protection contre erreur d'exécution
        try:
            sample = sorted(np.random.choice(np.array(ids), size=sample_size, replace=False).tolist())
        except:
            raise RuntimeError("Erreur ! L'extraction a échouée. Revoir peut-être la signature…")
        else:
            self.sample_patients = sample
            print("Extraction de l'échantillon terminée !")
            print("Patientes sélectionnées :")
            for i, pid in enumerate(sample, 1):
                print(f"  {i:2d}. {pid}")
            print()
        finally:
            print("Fin de cette partie !")
            print("="*70)
            print()

    


    def create_bids_structure(self, patient_id : str, sessions : list = ['T0', 'T1', 'T2', 'T3']) -> dict:
        """
        Crée l'arborescence BIDS pour un patient. Retourne un dict session → {datatype: Path}

        Args
        --------
        patient_id : str
            ``PatientID`` du patient
        
        sessions : list
            Liste des différentes sessions du patient. Pour ``ISPY2``, correspond aux ``Studies`` du patient \: 4 sessions chronologiques (T0, T1, T2, T3)

        Returns
        ---------
        res : dict
            Dictionnaire des chemins créés pour chaque session
        """

        #Destination patient selon standard BIDS
        patient_dir = self.bids_dir / 'derivatives' / 'nifti' / f'sub-{patient_id}'
        patient_dir.mkdir(parents=True, exist_ok=True)
        res = {}

        for session in sessions:
            #Destination de chaque session
            session_dir = patient_dir / f"ses-{session}"
            session_dir.mkdir(parents=True, exist_ok=True)

            #Destination pour les volumes DCE-MRI (perf) et le masque de segmentation (seg) selon standard BIDS
            (session_dir / 'perf').mkdir(parents=True, exist_ok=True)
            (session_dir / 'seg').mkdir(parents=True, exist_ok=True)
            if (session_dir / 'perf').exists() and (session_dir / 'seg').exists():
                res[f"ses-{session}"] = {'perf' : session_dir / 'perf', 
                                'seg' : session_dir / 'seg'}
            
            #Protection contre erreur de création de destination
            else:
                raise FileExistsError("Erreur ! Destination non trouvée")
            
        sort = dict(sorted(res.items()))
        res= sort

        return res
    

    def download_dicoms(self, pid : int, filters : list = ['original DCE', 'Analysis Mask']) -> Path:
        """
        Télecharge les séries DICOMS spécifiques d'un patient donné

        Args
        ---------
        pid : int
            ``PaitientID`` dans la collection
        filters : list
            Liste de mots-clés permettant de filtrer les séries DICOM à télécharger

        Returns
        ------------
        pth : Path
            Destination d'entrée (Studies -> Series -> .dcm) vers les DICOM du dossier du patient
        """

        print('='*70)
        #Recuperation des Study du patient avec getStudy
        print(f"Récupération des données Study du patient {pid}…")
        pstudies= nbia.getStudy(collection= ISPY2_COLLECTION, patientId= pid)
        
        
        for study in pstudies:
            date = study['StudyDate']
            study['StudyDate']= datetime.datetime.strptime(date, '%m-%d-%Y').date()
        
        #Tri temporel selon la date de visite (T0, T1, T2, T3)
        pstudies.sort(key=itemgetter('StudyDate'))
        print(f"\nDonnées Study du patient {pid} trouvées et triées !\n")
        print('='*70)

        print()

        if len(pstudies) < 4:
            print('='*70)
            print(f"\nRécupération stoppée : moins de 4 Studies trouvées ({len(pstudies)}/4).\n")
            print('='*70)
            return None
            
        #Recuperation des StudyInstanceUIDs 
        sids = [pstudy['StudyInstanceUID'] for pstudy in pstudies]

        print('='*70)
        print("Début de la récupération des DICOM…")

        #Destination du dossier patient
        pth = self.raw_dicom / f"sub-{pid}"

        #Pour chaque study :
        #Filtrage sur les series souhaitées avec filters et téléchargement des DICOM
        for sid in sids:
            series = []
            for serie in nbia.getSeries(studyUid=sid):
                for kw in filters :
                    if kw.upper() in serie['SeriesDescription'].upper():
                        series.append(serie)

            print(f"\t-Fin du filtrage sur les séries")
            print("\tSéries retenues : ")
            for idx, serie in enumerate(series, 1):
                print(f"\t\t{idx:2d}. {serie['SeriesDescription']}")

            for serie in series:
                output_path = pth / f"ses-T{sids.index(sid)}"

                fichiers_avant = len(list(output_path.rglob('*.dcm'))) if output_path.exists() else 0
                print('-'*100)
                print(f"\t-Dèbut du téléchargement de la série {series.index(serie) + 1} de la session T{sids.index(sid)} du patient {pid}…")
                nbia.downloadSeries(series_data=[serie['SeriesInstanceUID']], 
                                    input_type="list", 
                                    path= output_path
                                    )
                fichiers_apres = len(list(output_path.rglob('*.dcm')))
                dicoms_telecharges = fichiers_apres - fichiers_avant

                print(f"    -Fin du téléchargement de la série {series.index(serie) + 1} du patient {pid}")
                print(f"    {dicoms_telecharges} fichiers DICOM téléchargés !")
                print('-'*100)
                print()

        print("Fin de la récupération des DICOM")
        print("="*70)
        print()

        print(f"Destination d'entrée vers les Studies du patient {pid} : {pth}")
        print()

        return pth
    

    def extract_volumes(self, patient_dir: Path) -> dict[Path]:
        """
        Extrait les fichiers des phases PRE, POST_1, POST_2 et POST_3 à partir de la série originale DCE

        Args
        -------
        patient_dir : Path
            Destination d'entrée vers les données du patient (Studies)

        Returns
        --------
        res : dict[Path]
            Destination des dossiers contenant les phases PRE, POST_1, POST_2 et POST_3
        """

        # Sécurisation du parcours des sessions
        if not patient_dir.is_dir():
            print(f"Erreur : Le dossier patient {patient_dir} n'existe pas.")
            return

        studies = [d for d in os.listdir(patient_dir) if (patient_dir / d).is_dir() and not d.startswith('.')]

        res = {}

        print("="*50)
        print(f"Patient {patient_dir.name}")
        print("="*50)

        for study in studies:
            study_path = patient_dir / study
            print(f"\n=== Récupération des phases de la session {study} ===")

            # Filtrage des dossiers pour éviter les fichiers cachés (ex: .DS_Store)
            series_dirs = [d for d in os.listdir(study_path) if (study_path / d).is_dir()][:2]
            
            # Configuration des dossiers cibles
            phases_dirs = {
                "dce-pre": study_path / "PRE",
                "dce-post-1": study_path / "POST_1",
                "dce-post-2": study_path / "POST_2",
                "dce-post-3": study_path / "POST_3"
            }
            mask_dir = study_path / "MASK"


            # Création des dossiers
            for p_dir in list(phases_dirs.values()) + [mask_dir]:
                p_dir.mkdir(parents=True, exist_ok=True)

            for s_id in series_dirs:
                dir_path = study_path / s_id
                paths = list(dir_path.rglob("*.dcm"))
                
                if not paths:
                    continue

                # On couple le chemin et le DICOM dès le départ
                # stop_before_pixels=True pour aller bcp plus vite
                dicoms_data = []
                tags_to_read = [
                    'PatientID', 'StudyDate', 'StudyInstanceUID', 'StudyDescription', 
                    'SeriesInstanceUID', 'SeriesDescription', 'Rows', 'Columns', 
                    'SliceLocation', 'AcquisitionTime', 'InstanceNumber'
                ]
                
                for pth in paths:
                    try:
                        dcm = dcmread(pth, stop_before_pixels=True, specific_tags=tags_to_read)
                        dicoms_data.append({'path': pth, 'metadata': dcm})
                    except Exception as e:
                        print(f"Impossible de lire le fichier {pth.name} : {e}")

                # Détection Masque vs Série Originale
                if len(paths) <= 1:
                    print(f"Récupération du masque de segmentation {s_id}...")
                    shutil.copy2(paths[-1], mask_dir)
                    print(f"Extraction vers {mask_dir.name} terminée.")
                    continue

                # --- Traitement de la série originale ---
                print(f"Extraction des phases de la série originale {s_id}...")

                # Extraction des valeurs uniques de géométrie et de temps
                slice_locations = sorted(list(set(round(float(d['metadata'].SliceLocation), 4) for d in dicoms_data if hasattr(d['metadata'], 'SliceLocation'))))
                acquisition_times = sorted(list(set(float(d['metadata'].AcquisitionTime) for d in dicoms_data if hasattr(d['metadata'], 'AcquisitionTime'))))
                
                n_slices = len(slice_locations)
                n_phases = len(acquisition_times)
                
                if n_slices == 0:
                    print(f"Erreur : Aucune SliceLocation valide trouvée dans la série {s_id}")
                    continue

                rows = dicoms_data[0]['metadata'].Rows
                cols = dicoms_data[0]['metadata'].Columns
                
                print(f"Structure détectée : {n_phases} phases temporelles | {n_slices} coupes par phase.")
                print(f"Dimensions d'une coupe : {rows}x{cols}")

                # Cas 1 : Multiples phases encodées avec des AcquisitionTime différents
                if n_phases > 1:
                    phases_to_process = acquisition_times[:4]
                    phase_keys = ["dce-pre", "dce-post-1", "dce-post-2", "dce-post-3"][:len(phases_to_process)]
                    
                    for phase_time, phase_key in zip(phases_to_process, phase_keys):
                        # Filtrage des coupes appartenant à cette phase temporelle
                        phase_slices = [d for d in dicoms_data if float(d['metadata'].AcquisitionTime) == phase_time]
                        
                        if len(phase_slices) != n_slices:
                            print(f"Erreur ! Nombre de coupes incohérent pour {phase_key}. {len(phase_slices)} trouvées contre {n_slices} attendues.")
                            return

                        target_dir = phases_dirs[phase_key]
                        for item in phase_slices:
                            shutil.copy2(item['path'], target_dir)
                        
                        print(f"Extraction vers {target_dir.name} terminée")

                # Cas 2 : Une seule ou aucune AcquisitionTime distincte (les phases sont empilées par InstanceNumber)
                else:
                    # Tri global par InstanceNumber
                    dicoms_data.sort(key=lambda x: int(x['metadata'].InstanceNumber or 0))
                    phase_keys = ["dce-pre", "dce-post-1", "dce-post-2", "dce-post-3"]
                    
                    for i, phase_key in enumerate(phase_keys):
                        start_idx = i * n_slices
                        end_idx = start_idx + n_slices
                        phase_slices = dicoms_data[start_idx:end_idx]

                        if len(phase_slices) != n_slices:
                            print(f"Erreur ! Pas assez de coupes pour extraire la phase {phase_key}.")
                            return

                        target_dir = phases_dirs[phase_key]
                        for item in phase_slices:
                            shutil.copy2(item['path'], target_dir)
                        
                        print(f"Extraction vers {target_dir.name} terminée")

            print(f"===== Extraction de la session {study} terminée ! =====")

            phases_dirs.update({"mask" : mask_dir})
            res[study] = phases_dirs
        
        sort = dict(sorted(res.items()))
        res= sort

        return res
    


    def convert_series_to_nifti(self, series_dir : Path, output_nii : Path) -> Path:
        """
        Convertit une série DICOM en NIfTI

        Args
        --------
        series_dir : Path
            Destination de la série DICOM
        
        output_nii : Path
            Destination à laquelle stocker le fichier NIfTI créé

        Returns
        ----------
        Destination du fichier ``.nii.gz``
        """
        
        #Liste des fichiers DICOM de la destination de la serie
        dicom_paths = list(series_dir.rglob("*.dcm"))

        if len(dicom_paths) < 1:
            print(f"La série {series_dir} ne contient aucun fichier .dcm")
            return None
        
        #Recuperation des differents SeriesInstanceUIDs des DICOM de la serie
        #Doit etre unique sinon serie corrompue
        dsts = [dcmread(pth, stop_before_pixels=True, specific_tags=['SeriesInstanceUID']) for pth in dicom_paths]
        serie_ids = list(set(dst.SeriesInstanceUID for dst in dsts))

        if len(serie_ids) > 1:
            raise ValueError(f"{len(serie_ids)} détectées. Série corrompue")
        

        #Creation de la destination de stockage
        output_nii.mkdir(parents=True, exist_ok=True)

        #Verification de la creation
        if os.path.exists(output_nii):
            print(f"{output_nii} créé avec succès !")
        else:
            raise FileNotFoundError(f"{output_nii} n'existe pas")

        #Conversion fichiers DICOM -> volume NIfTI
        try:
            dicom2nifti.convert_directory(str(series_dir), str(output_nii), compression=True, reorient=True)
            
        except Exception as e:
            print(f"Exception dicom2nifti au fichier {series_dir} : {e}")


        #Verification de la presence du fichier .nii.gz
        nii_paths = list(output_nii.rglob("*.nii.gz"))

        if len(nii_paths) == 0:
            raise RuntimeError(f"Aucun zip trouvé dans {output_nii}")
        
        if len(nii_paths) > 1:
            raise RuntimeError(f"Plus d'un zip trouvé dans {output_nii}")
        
        nii_path = nii_paths[0]

        #Verification du contenu (taille, premieres donnees)
        if os.path.getsize(nii_path) == 0:
            raise RuntimeError(f"Le fichier {nii_path} est vide (0 octet).")

        with gzip.open(nii_path, 'rb') as f_in:
            header = f_in.read(10)
            if not header :
                raise RuntimeError(f"Aucune donnée lisible dans le fichier {nii_path}")
                
        #Reorientation LAS -> RAS
        try:
            img_las= nib.load(str(nii_path))

            ornt_init = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
            ornt_fin = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
            transform = nib.orientations.ornt_transform(ornt_init, ornt_fin)

            img = img_las.as_reoriented(transform)

            shutil.rmtree(nii_path.parent)

        except Exception as e:
            print(f"Exception lors de la réorientation LAS -> RAS du fichier {nii_path} : {e}")
            return None
            
        nib.save(img, output_nii)

        return output_nii


    def inspect_nifti(self, nii_path: Path) -> dict:
        """
        Charge un NIfTI et retourne un dict avec ses caractéristiques.
        
        Args
        --------
        nii_path : Path
            Destination / emplacement du fichier NIfTI

        Returns
        ---------
        meta : dict
            Dictionnaire de quelques metadonnées du fichier NIfTI
        """

        #Chargement et extraction du volume et de la matrice affine
        img= nib.load(str(nii_path))
        data = img.get_fdata()
        affine = img.affine

        #Metadonnees
        meta = {
            'Path' : str(nii_path),
            'shape' : img.shape,
            'dtype' : data.dtype,
            'affine' : affine,
            'axcodes' : nib.aff2axcodes(affine),
            'voxel_sizes' : img.header.get_zooms()[:3],
            'min' : np.min(data),
            'max' : np.max(data),
            'mean' : np.mean(data), 
            'std' : np.std(data)
        }

        return meta
    


    def pipeline(self) -> None:
        """
        Exécute le pipeline complet :
        info -> sélection de l'échantillon de patientes -> téléchargement des DICOMs -> extraction des phases pre, post 1 à 3 et du masque de segmentation -> conversion en NIfTI et stockage suivant standard BIDS

        Affiche les métadonnées principales pour chaque NIfTI créé
        """

        #Infos sur la collection ISPY2
        self.info()

        #Selection pour echantillon de patientes
        self.select_sample_patients(sample_size= self.sample_size)

        print("="*70)
        print("Début du pipeline")
        print("="*70)
        print()

        for pid in self.sample_patients :
            
            print()
            print('-'*70)
            print(f"Patient {pid}")
            print('-'*70)
            print()

            #Lancement du telechargement des DICOMs
            print()
            print("Début du téléchargement des dossiers DICOMs")
            print()
            raw_dicom_dir = self.download_dicoms(pid= pid)
            print()
            print("Téléchargements terminés !")
            print()

            if raw_dicom_dir is None:
                continue

            #Creation des destinations BIDS
            bids_dirs = self.create_bids_structure(patient_id= pid)
            print()
            print("Structure BIDS créé !")
            print()

            #Extraction des phases
            print()
            print("Début de l'extraction des phases temporelles des DICOMs")
            print()
            volumes = self.extract_volumes(raw_dicom_dir)
            print()
            print("Extractions terminées !")
            print()

            print()
            print("Début de la conversion DICOM -> NIfTI")
            print()

            for dir_label, vol_label in zip(bids_dirs, volumes):
                
                #Si sessions differentes, ne pas lancer la conversion
                if dir_label != vol_label:
                    raise ValueError(f"Sessions incompatibles : {dir_label} pour la destination bids contre {vol_label} pour le volume")
                
                #Recuperation de la session
                ses = min([dir_label, vol_label])
                print()
                print(f"Session {ses}")
                print()
                
                series_dirs= volumes[ses]#Destinations des phases temporelles DICOMs
                nii_dirs= bids_dirs[ses]#Destinations perf et seg de la structure BIDS

                #Pour chaque serie :
                #   -Trouver la bonne destination BIDS
                #   -Convertir en .nii.gz
                #   -Afficher les metadonnees
                for suffix, serie_dir in series_dirs.items():
                    if "dce" in suffix:
                        output_nii = nii_dirs["perf"] / f"sub-{pid}_{ses}_{suffix}.nii.gz"

                        if output_nii.exists():
                            size = os.path.getsize(output_nii)
                            if size == 0 :
                                print()
                                print(f"Le NIfTI (.nii.gz) de la phase {suffix} de la session {ses} existe déjà mais vide !")
                                print()
                                continue

                            else:
                                print()
                                print(f"Le NIfTI (.nii.gz) de la phase {suffix} de la session {ses} existe déjà ! Taille : {size}")
                                print()
                                continue

                        nii_path = self.convert_series_to_nifti(serie_dir, output_nii)
                        print()
                        print(f"NIfTI de la phase {suffix} de la session {ses} créé !")
                        print()

                        metas = self.inspect_nifti(nii_path)
                        print(f"Metadonnées du fichier {nii_path}")
                        print("{")
                        for key, val in metas.items():
                            print(f"\t{key} : {val}")
                        print("}")

                    else : 
                        output_nii = nii_dirs["seg"] / f"sub-{pid}_{ses}_{suffix}.nii.gz"

                        if output_nii.exists():
                            size = os.path.getsize(output_nii)
                            if size == 0 :
                                print()
                                print(f"Le NIfTI (.nii.gz) de la phase {suffix} de la session {ses} existe déjà mais vide !")
                                print()
                                continue

                            else:
                                print()
                                print(f"Le NIfTI (.nii.gz) de la phase {suffix} de la session {ses} existe déjà ! Taille : {size}")
                                print()
                                continue

                        nii_path = self.convert_series_to_nifti(serie_dir, output_nii)
                        print()
                        print(f"NIfTI de la phase {suffix} de la session {ses} créé !")
                        print()

                        metas = self.inspect_nifti(nii_path)
                        print(f"Metadonnées du fichier {nii_path}")
                        print("{")
                        for key, val in metas.items():
                            print(f"\t{key} : {val}")
                        print("}")
                print()
                print("-"*70)
                print()

        print()
        print("="*70)
        print("Fin du pipeline")
        print("="*70)
        print()

        return None