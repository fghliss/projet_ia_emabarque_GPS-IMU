# projet_ia_emabarque_GPS-IMU


# GPS–IMU Fusion for Vehicle Localization (LSTM-based)

Ce projet présente une approche d’**estimation de pose et de vitesse pour un véhicule autonome** à partir de données **GPS + IMU**, basée sur un **réseau LSTM avec mécanisme de fusion**.  
L’objectif est de **limiter la dérive de l’IMU**, de rester robuste aux **coupures GPS**, et de préparer un **déploiement embarqué sur NVIDIA Jetson** via ONNX / TensorRT.

---

## Objectifs

- Estimer la **pose du véhicule** (déplacements et orientation)
- Estimer optionnellement la **vitesse du véhicule**
- Gérer les **blackouts GPS** de manière réaliste
- Produire des modèles **exportables et optimisables** pour l’embarqué

---

## Méthodologie

- **LSTM** pour modéliser la dynamique temporelle des mesures IMU
- **MLP GPS** pour traiter l’information de correction GPS
- **Fusion IMU–GPS via FiLM** (Feature-wise Linear Modulation)
- Entraînement supervisé sur le dataset **KITTI / OXTS**
- Prédiction **incrémentale** (dx, dy, dyaw) afin de reconstruire la trajectoire

---

## 📂 Contenu du dépôt

Le dépôt contient **deux implémentations distinctes**, chacune accompagnée de son modèle exporté en ONNX.

---

### 1️⃣ LSTM Fusion – Estimation de la Pose

- **Sorties du réseau** :
  - dx, dy, dz
  - dyaw
- **Usage** :
  - Estimation de trajectoire
  - Analyse de la dérive en cas de coupure GPS
- **Fichiers** :
  - `lstm-fusion_pose.py`
  - `lstm-fusion_pose.onnx`

---

### 2️⃣ LSTM Fusion – Estimation de la Pose + Vitesse

- **Sorties du réseau** :
  - dx, dy, dz
  - dyaw
  - vn, ve, vu (vitesses)
- **Usage** :
  - Estimation conjointe de la pose et de la vitesse
  - Amélioration de la stabilité en auto-régression
- **Fichiers** :
  - `lstm-fusion_pose_vitesse.py`
  - `lstm-fusion_pose_vitesse.onnx`

---


## Dataset

- **KITTI Odometry – OXTS**
- Données GPS, IMU et orientation synchronisées
- Découpage strict train / validation par séquences

---



---
