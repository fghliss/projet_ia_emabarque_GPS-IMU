# -*- coding: utf-8 -*-
"""
Kaggle-ready — Fusion IA GPS + IMU + COMPASS + SPEED PREDICTION (EXPLICITE)
VERSION FINALE : SORTIES = [dx, dy, dz, dyaw, vn, ve, vu]
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# =========================
# 1) CONFIGURATION
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {DEVICE}")

# Paramètres
MAX_DRIVES = 30
CHUNK_LEN = 300
SEQ_LEN = 20
BATCH = 64
EPOCHS = 25       
LR = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
CLIP_NORM = 1.0

# Robustesse
GPS_DROP_PROB = 0.2     
ACCURACY_THRESHOLD = 0.20 # 20 cm

# =========================
# 2) DATA LOADING
# =========================
oxts_fields = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu", "ax", "ay", "az",
    "af", "al", "au", "wx", "wy", "wz", "wf", "wl", "wu",
    "pos_accuracy", "vel_accuracy", "navstat", "numsats",
    "posmode", "velmode", "orimode",
]
EARTH_R = 6378137.0

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def find_kitti_root():
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return "./data/2011_09_26" if os.path.isdir("./data/2011_09_26") else base
    for slug in os.listdir(base):
        cand = os.path.join(base, slug, "data", "2011_09_26")
        if os.path.isdir(cand): return cand
        cand2 = os.path.join(base, slug, "2011_09_26")
        if os.path.isdir(cand2): return cand2
    return base

ROOT_DIR = find_kitti_root()
base_dir = ROOT_DIR

def load_dt_sequence(seq_name):
    ts_path = os.path.join(base_dir, seq_name, "oxts/timestamps.txt")
    ts = []
    with open(ts_path, "r") as f:
        for line in f:
            if line.strip(): ts.append(datetime.strptime(line.strip()[:26], "%Y-%m-%d %H:%M:%S.%f"))
    dt = np.array([(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)], dtype=np.float32)
    return np.concatenate([dt, dt[-1:]], axis=0)

def load_oxts_sequence(seq_name):
    path = os.path.join(base_dir, seq_name, "oxts/data")
    files = sorted(os.listdir(path))
    records = []
    for fname in files:
        with open(os.path.join(path, fname), "r") as f:
            val = list(map(float, f.read().strip().split()))
            if len(val) == len(oxts_fields): records.append(val)
    df = pd.DataFrame(records, columns=oxts_fields)
    df["seq"] = seq_name
    df["frame"] = np.arange(len(df), dtype=np.int32)
    df["dt"] = load_dt_sequence(seq_name)[:len(df)]
    return df

def latlon_to_enu(df):
    lat0, lon0, alt0 = df.iloc[0][["lat", "lon", "alt"]]
    lat, lon = np.deg2rad(df["lat"].values), np.deg2rad(df["lon"].values)
    l0, ln0 = np.deg2rad(lat0), np.deg2rad(lon0)
    df["x_e"] = EARTH_R * (lon - ln0) * np.cos(l0)
    df["y_n"] = EARTH_R * (lat - l0)
    df["z_u"] = df["alt"].values - alt0
    return df

def add_features(df):
    out = df.copy()
    out["dx"], out["dy"], out["dz"], out["dyaw"] = 0.0, 0.0, 0.0, 0.0
    out["sin_yaw"] = np.sin(out["yaw"])
    out["cos_yaw"] = np.cos(out["yaw"])
    
    for _, g in out.groupby("seq"):
        idx = g.index.to_numpy()
        if len(idx) < 2: continue
        out.loc[idx[:-1], "dx"] = g["x_e"].values[1:] - g["x_e"].values[:-1]
        out.loc[idx[:-1], "dy"] = g["y_n"].values[1:] - g["y_n"].values[:-1]
        out.loc[idx[:-1], "dz"] = g["z_u"].values[1:] - g["z_u"].values[:-1]
        out.loc[idx[:-1], "dyaw"] = wrap_pi(g["yaw"].values[1:] - g["yaw"].values[:-1])
    return out

def add_seq_parts(df, chunk_len=300):
    df["seq_part"] = df["seq"] + "_p" + (df["frame"] // chunk_len).astype(str)
    return df

# --- SPLIT RIGOUREUX ---
all_drives = sorted([d for d in os.listdir(base_dir) if "drive" in d and "sync" in d])
val_seqs = [d for d in all_drives if "0018" in d]
if not val_seqs: val_seqs = all_drives[-2:]
train_seqs = [d for d in all_drives if d not in val_seqs]

print(f"✅ TRAIN : {len(train_seqs)} séquences | VAL : {len(val_seqs)} séquences")

df_all = pd.concat([latlon_to_enu(load_oxts_sequence(s)) for s in all_drives], ignore_index=True)
df_all = add_features(df_all)

df_train = df_all[df_all["seq"].isin(train_seqs)].copy().reset_index(drop=True)
df_val   = df_all[df_all["seq"].isin(val_seqs)].copy().reset_index(drop=True)

df_train = add_seq_parts(df_train, CHUNK_LEN)
df_val   = add_seq_parts(df_val, CHUNK_LEN)

# =========================
# 3) BUILD WINDOWS (AVEC 7 SORTIES)
# =========================
IMU_COLS = ["ax", "ay", "az", "wx", "wy", "wz", "dt", "sin_yaw", "cos_yaw"]

# ON AJOUTE LES VITESSES EXPLICITES DANS LA CIBLE
# vn = Velocity North, ve = Velocity East, vu = Velocity Up
Y_COLS   = ["dx", "dy", "dz", "dyaw", "vn", "ve", "vu"]

imu_mean = df_train[IMU_COLS].values.mean(axis=0).astype(np.float32)
imu_std  = (df_train[IMU_COLS].values.std(axis=0) + 1e-6).astype(np.float32)
y_mean = df_train[Y_COLS].values.mean(axis=0).astype(np.float32)
y_std  = (df_train[Y_COLS].values.std(axis=0) + 1e-6).astype(np.float32)

y_mean_t = torch.tensor(y_mean, device=DEVICE)
y_std_t  = torch.tensor(y_std, device=DEVICE)

def build_windows(df, seq_len=20):
    Ximu_list, Xgps_prev_list, y_list = [], [], []
    meta = [] 
    
    imu_all = ((df[IMU_COLS].values.astype(np.float32) - imu_mean) / imu_std)
    y_all   = ((df[Y_COLS].values.astype(np.float32) - y_mean) / y_std)

    for part_id, g in df.groupby("seq_part"):
        idx = g.index.to_numpy()
        if len(idx) < (seq_len + 1): continue

        imu = imu_all[idx]
        y   = y_all[idx]
        
        # Le GPS précédent contient maintenant AUSSI la vitesse précédente (7 valeurs)
        # Cela aide l'IA en auto-régression : elle sait à quelle vitesse elle allait à t-1
        gps_prev = np.zeros_like(y, dtype=np.float32)
        gps_prev[1:] = y[:-1]

        for i in range(0, len(idx) - seq_len):
            t = i + seq_len - 1
            Ximu_list.append(imu[i:i+seq_len])
            y_list.append(y[t])
            Xgps_prev_list.append(gps_prev[t])
            meta.append((part_id, int(idx[t]), 0)) # dt plus nécessaire dans meta, on l'a dans IMU

    return np.asarray(Ximu_list, dtype=np.float32), np.asarray(Xgps_prev_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), np.asarray(meta, dtype=object)

Ximu_train, Xgps_train, y_train, meta_train = build_windows(df_train, SEQ_LEN)
Ximu_val, Xgps_val, y_val, meta_val = build_windows(df_val, SEQ_LEN)

# =========================
# 4) MODÈLE FUSION (7 sortis)
# =========================
class FusionDataset(Dataset):
    def __init__(self, Ximu, Xgps, y):
        self.Ximu = torch.from_numpy(Ximu)
        self.Xgps = torch.from_numpy(Xgps)
        self.y    = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Ximu[i], self.Xgps[i], self.y[i]

train_loader = DataLoader(FusionDataset(Ximu_train, Xgps_train, y_train), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader   = DataLoader(FusionDataset(Ximu_val, Xgps_val, y_val), batch_size=BATCH, shuffle=False)

class FusionLSTM(nn.Module):
    def __init__(self, imu_dim, gps_dim, hidden=96, layers=2, dropout=0.2, out_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(imu_dim, hidden, layers, batch_first=True, dropout=dropout)
        
        # GPS MLP prend maintenant 7 entrées (dx,dy,dz,dyaw, vn,ve,vu) + 1 mask = 8
        self.gps_mlp = nn.Sequential(nn.Linear(gps_dim + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        
        self.film = nn.Linear(64, 2 * hidden)
        self.head = nn.Sequential(nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, x_imu, gps_prev_norm, gps_mask):
        out, _ = self.lstm(x_imu)
        h_last = out[:, -1, :]
        g = self.gps_mlp(torch.cat([gps_prev_norm, gps_mask], dim=1))
        gamma, beta = torch.chunk(self.film(g), 2, dim=1)
        h_mod = h_last * (1.0 + gamma) + beta
        return self.head(torch.cat([h_mod, g], dim=1))

# gps_dim=7 et out_dim=7
model = FusionLSTM(imu_dim=len(IMU_COLS), gps_dim=len(Y_COLS), hidden=HIDDEN_DIM, layers=NUM_LAYERS, dropout=DROPOUT, out_dim=len(Y_COLS)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.SmoothL1Loss()

# =========================
# 5) VALIDATION & ACCURACY
# =========================
@torch.no_grad()
def eval_metrics(loader):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    for ximu, xgps, y in loader:
        ximu, xgps, y = ximu.to(DEVICE), xgps.to(DEVICE), y.to(DEVICE)
        gps_mask = torch.ones((xgps.size(0), 1), device=DEVICE)
        
        pred = model(ximu, xgps, gps_mask)
        loss = criterion(pred, y)
        total_loss += loss.item()
        
        # Accuracy sur la Position (Indices 0 et 1)
        pred_m = pred * y_std_t + y_mean_t
        y_m    = y * y_std_t + y_mean_t
        dist_err = torch.sqrt((pred_m[:,0] - y_m[:,0])**2 + (pred_m[:,1] - y_m[:,1])**2)
        correct_preds += (dist_err < ACCURACY_THRESHOLD).sum().item()
        total_samples += y.size(0)
        
    return total_loss / len(loader), (correct_preds / total_samples) * 100

# =========================
# 7) TRAINING
# =========================
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

print(f"\n🚀 Start Training (Multi-Task: Pos + Speed Prediction)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_sum = 0
    
    for ximu, xgps, y in train_loader:
        ximu, xgps, y = ximu.to(DEVICE), xgps.to(DEVICE), y.to(DEVICE)
        gps_mask = torch.ones((xgps.size(0), 1), device=DEVICE)
        drop = (torch.rand((xgps.size(0), 1), device=DEVICE) < GPS_DROP_PROB).float()
        gps_mask = gps_mask * (1.0 - drop)
        
        pred = model(ximu, xgps * gps_mask, gps_mask)
        loss = criterion(pred, y)
        
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM); optimizer.step()
        tr_sum += loss.item()
    
    val_loss, val_acc = eval_metrics(val_loader)
    history['train_loss'].append(tr_sum/len(train_loader))
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Ep {epoch:02d} | Loss: {tr_sum/len(train_loader):.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.1f}%")

# =========================
# 10) TEST AUTO-RÉGRESSIF (7 DIMENSIONS)
# =========================
TARGET_PART = "2011_09_26_drive_0018_sync_p0" 
mask_part = (meta_val[:, 0] == TARGET_PART)
if np.sum(mask_part) == 0:
    potential = [m for m in np.unique(meta_val[:,0]) if "0018" in str(m)]
    TARGET_PART = potential[0] if potential else meta_val[0,0]
    mask_part = (meta_val[:, 0] == TARGET_PART)

Ximu_test = Ximu_val[mask_part]
Xgps_test = Xgps_val[mask_part]
y_test    = y_val[mask_part]

n = len(Ximu_test)
b0, b1 = int(n*0.3), int(n*0.7)
gps_mask_arr = np.ones((n,1), dtype=np.float32)
gps_mask_arr[b0:b1] = 0.0

print(f"\n🔄 Inférence Auto-régressive (avec mémoire de vitesse) sur {TARGET_PART}...")

model.eval()
predictions = []
# Initialisation avec la première vérité terrain (Pos + Vitesse)
prev_pred = torch.from_numpy(Xgps_test[0:1]).to(DEVICE)

with torch.no_grad():
    for i in range(n):
        ximu_in = torch.from_numpy(Ximu_test[i:i+1]).to(DEVICE)
        
        current_mask_val = gps_mask_arr[i, 0]
        mask_in = torch.tensor([[current_mask_val]], device=DEVICE)
        
        if current_mask_val == 1.0:
            gps_in = torch.from_numpy(Xgps_test[i:i+1]).to(DEVICE)
        else:
            # MAGIE : On réinjecte non seulement la position prédite, 
            # mais aussi la VITESSE prédite à l'instant d'avant !
            gps_in = prev_pred 
            
        pred_out = model(ximu_in, gps_in * mask_in, mask_in)
        predictions.append(pred_out.cpu().numpy())
        prev_pred = pred_out

pred = np.concatenate(predictions, axis=0)

def denorm_y(y_norm): return y_norm * y_std + y_mean
pred_den, gt_den = denorm_y(pred), denorm_y(y_test)

# Reconstruction Trajectoire (Indices 0 et 1)
x_gt = np.cumsum(gt_den[:,0])
y_gt = np.cumsum(gt_den[:,1])
x_pred = np.cumsum(pred_den[:,0])
y_pred = np.cumsum(pred_den[:,1])

# --- CALCUL METRIQUES ---
pos_err = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
ate = np.mean(pos_err)

dist_totale = np.sum(np.sqrt(gt_den[:,0]**2 + gt_den[:,1]**2))
drift_pct = (pos_err[-1] / dist_totale) * 100

# Calcul de la Vitesse (Magnitude des vecteurs Vn, Ve prédits)
# Indices : 4=Vn, 5=Ve, 6=Vu
speed_gt_explicit = np.sqrt(gt_den[:,4]**2 + gt_den[:,5]**2)
speed_pred_explicit = np.sqrt(pred_den[:,4]**2 + pred_den[:,5]**2)
rmse_speed = np.sqrt(np.mean((speed_gt_explicit - speed_pred_explicit)**2))

print("\n" + "="*50)
print(f"📊 RAPPORT FINAL (Prédiction Vitesse Explicite)")
print("="*50)
print(f"📍 ATE (Position) : {ate:.2f} m")
print(f"🚀 RMSE (Vitesse) : {rmse_speed:.2f} m/s")
print(f"📉 Drift Final    : {drift_pct:.2f} %")
print(f"🎯 Accuracy Finale      : {history['val_acc'][-1]:.2f} % (Tolérance {ACCURACY_THRESHOLD}m)")
print("="*50)

# --- GRAPHIQUE 1 : TRAJECTOIRE ---
plt.figure(figsize=(10, 8))
plt.plot(x_gt, y_gt, 'g-', linewidth=3, label="Vraie Route")
plt.plot(x_pred[:b0], y_pred[:b0], 'b--', linewidth=2, label="IA (GPS ON)")
plt.plot(x_pred[b0-1:b1+1], y_pred[b0-1:b1+1], 'r-', linewidth=4, label="IA (IMU Seul)")
plt.plot(x_pred[b1:], y_pred[b1:], 'b--', linewidth=2)
plt.title(f"Fusion Trajectory (Speed-Aware) : {TARGET_PART}")
plt.axis('equal'); plt.grid(True); plt.legend()
plt.savefig("final_traj_explicit.png")
plt.show()

# --- GRAPHIQUE 2 : VITESSE PRÉDITE PAR L'IA ---
plt.figure(figsize=(10, 4))
plt.plot(speed_gt_explicit * 3.6, 'g-', label="Vitesse Réelle (Capteur)")
plt.plot(speed_pred_explicit * 3.6, 'b--', label="Vitesse PRÉDITE par l'IA")
plt.axvspan(b0, b1, color='red', alpha=0.1, label="GPS OFF")
plt.title("Prédiction Directe de la Vitesse (Sortie Réseau)")
plt.xlabel("Temps"); plt.ylabel("km/h"); plt.grid(True, alpha=0.3); plt.legend()
plt.savefig("final_speed_explicit.png")
plt.show()

# --- GRAPHIQUE 3 : STATS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history['val_acc'], 'g'); plt.title("Accuracy (<20cm)"); plt.grid(True)
plt.savefig("final_stats_explicit.png")
plt.show()