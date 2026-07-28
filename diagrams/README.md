# Diagrammes UML — Medical Image Processing

Diagrammes UML au format PlantUML pour le rapport.

## Fichiers

| Fichier | Contenu |
|---|---|
| `package_diagram.puml` | Diagramme de packages global (dépendances entre modules) |
| `class_diagram_core.puml` | Vue globale des responsabilités et dépendances |
| `class_diagram_ingestion.puml` | Ingestion TCIA/DICOM et modèle NIfTI |
| `class_diagram_preparation.puml` | Préparation, débruitage et contours |
| `class_diagram_segmentation_evaluation.puml` | Segmentation, nettoyage, métriques et visualisation |
| `class_diagram_morphology_analysis.puml` | Morphologie, topologie et analyse de forme |
| `pipeline_diagram.puml` | Pipeline de traitement d’une IRM DCE |
| `sequence_ispy2.puml` | Séquence d’extraction TCIA → DICOM → NIfTI |

## Rendu

### En ligne
Coller le contenu d'un fichier `.puml` sur : https://www.plantuml.com/plantuml/uml/

### En local (VS Code)
1. Installer l'extension **PlantUML** (jebbs.plantuml)
2. Ouvrir un fichier `.puml`
3. `Alt+D` pour afficher l'aperçu

### En local (CLI)
```bash
# Avec Java + PlantUML installé
plantuml diagrams/*.puml
# ou
java -jar plantuml.jar diagrams/*.puml
```

Les images PNG/SVG seront générées dans `diagrams/`.

Les diagrammes activent le moteur de layout intégré `smetana`, donc Graphviz
(`dot`) n'est pas nécessaire. Si PlantUML est configuré pour forcer Graphviz
et renvoie une erreur sur un chemin comme `/opt/local/bin/dot`, supprimer ce
paramétrage ou utiliser une version récente de PlantUML.

## Aperçu des diagrammes

### Package diagram
```
src/
├── medio          → modèle d'image et I/O NIfTI
├── preprocessing  → intensités et correction de biais
├── denoise        → réduction du bruit
├── edges          → contours
├── mathmorpho     → morphologie et mesures
├── segmed3d       → segmentation + évaluation
└── utils          → extraction TCIA / ISPY2
```

### Découpage fonctionnel

- **Ingestion** : TCIA/DICOM, conversion et modèle NIfTI.
- **Préparation** : normalisation, débruitage et contours.
- **Segmentation** : algorithmes classiques, atlas et clustering.
- **Évaluation** : nettoyage de masque, métriques et visualisation.
- **Analyse** : morphologie, topologie, distances et forme.
