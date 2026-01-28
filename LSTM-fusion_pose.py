# -*- coding: utf-8 -*-
"""
Kaggle-ready — Fusion IA GPS + IMU + COMPASS (KITTI / OXTS)
VERSION RIGOUREUSE : TRAIN/VAL SÉPARÉS + ACCURACY (TOLÉRANCE 20cm)
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
# 1) CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {DEVICE}")

# Data
MAX_DRIVES = 30
CHUNK_LEN = 300

# Model
SEQ_LEN = 20
BATCH = 64
EPOCHS = 25       
LR = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

# Training tricks
CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 10
GPS_DROP_PROB = 0.2     
GPS_NOISE_STD_NORM = 0.02
BLACKOUT_FRAC = (0.30, 0.70) 

# SEUIL D'ACCURACY (en mètres)
ACCURACY_THRESHOLD = 0.20 # 20 cm de tolérance

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

# --- SPLIT RIGOUREUX (TRAIN / VAL SÉPARÉS) ---
all_drives = sorted([d for d in os.listdir(base_dir) if "drive" in d and "sync" in d])

# 1. On isole la séquence 0018 pour la VALIDATION
val_seqs = [d for d in all_drives if "0018" in d]
if not val_seqs:
    print("⚠️ Séquence 0018 introuvable. Validation sur les 2 dernières.")
    val_seqs = all_drives[-2:]

# 2. TOUT LE RESTE va dans l'ENTRAÎNEMENT
train_seqs = [d for d in all_drives if d not in val_seqs]

print(f"✅ TRAIN : {len(train_seqs)} séquences")
print(f"✅ VAL (Test caché) : {val_seqs}")

# Chargement
df_all = pd.concat([latlon_to_enu(load_oxts_sequence(s)) for s in all_drives], ignore_index=True)
df_all = add_features(df_all)

df_train = df_all[df_all["seq"].isin(train_seqs)].copy().reset_index(drop=True)
df_val   = df_all[df_all["seq"].isin(val_seqs)].copy().reset_index(drop=True)

df_train = add_seq_parts(df_train, CHUNK_LEN)
df_val   = add_seq_parts(df_val, CHUNK_LEN)

# =========================
# 3) BUILD WINDOWS
# =========================
IMU_COLS = ["ax", "ay", "az", "wx", "wy", "wz", "dt", "sin_yaw", "cos_yaw"]
Y_COLS   = ["dx", "dy", "dz", "dyaw"]

imu_mean = df_train[IMU_COLS].values.mean(axis=0).astype(np.float32)
imu_std  = (df_train[IMU_COLS].values.std(axis=0) + 1e-6).astype(np.float32)
y_mean = df_train[Y_COLS].values.mean(axis=0).astype(np.float32)
y_std  = (df_train[Y_COLS].values.std(axis=0) + 1e-6).astype(np.float32)

# Sauvegarde des stats pour dénormalisation dans la boucle d'accuracy
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
        gps_prev = np.zeros_like(y, dtype=np.float32)
        gps_prev[1:] = y[:-1]

        for i in range(0, len(idx) - seq_len):
            t = i + seq_len - 1
            Ximu_list.append(imu[i:i+seq_len])
            y_list.append(y[t])
            Xgps_prev_list.append(gps_prev[t])
            meta.append((part_id, int(idx[t]), int(t)))

    return np.asarray(Ximu_list, dtype=np.float32), np.asarray(Xgps_prev_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), np.asarray(meta, dtype=object)

Ximu_train, Xgps_train, y_train, meta_train = build_windows(df_train, SEQ_LEN)
Ximu_val, Xgps_val, y_val, meta_val = build_windows(df_val, SEQ_LEN)

# =========================
# 4) MODEL
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
    def __init__(self, imu_dim, gps_dim, hidden=96, layers=2, dropout=0.2, out_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(imu_dim, hidden, layers, batch_first=True, dropout=dropout)
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

model = FusionLSTM(imu_dim=len(IMU_COLS), gps_dim=len(Y_COLS), hidden=HIDDEN_DIM, layers=NUM_LAYERS, dropout=DROPOUT, out_dim=len(Y_COLS)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.SmoothL1Loss()

# =========================
# 5) VALIDATION & ACCURACY FUNCTION
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
        
        # --- CALCUL ACCURACY ---
        # 1. Dénormaliser pour revenir aux mètres
        pred_m = pred * y_std_t + y_mean_t
        y_m    = y * y_std_t + y_mean_t
        
        # 2. Calculer l'erreur Euclidienne sur dx, dy (distance 2D)
        dist_err = torch.sqrt((pred_m[:,0] - y_m[:,0])**2 + (pred_m[:,1] - y_m[:,1])**2)
        
        # 3. Compter les "bons" résultats (erreur < seuil)
        correct_preds += (dist_err < ACCURACY_THRESHOLD).sum().item()
        total_samples += y.size(0)
        
    avg_loss = total_loss / len(loader)
    accuracy = (correct_preds / total_samples) * 100
    return avg_loss, accuracy

# =========================
# 7) TRAINING LOOP (AVEC ACCURACY)
# =========================
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'grads': []}

print(f"\n🚀 Démarrage (Seuil Accuracy: {ACCURACY_THRESHOLD*100:.0f}cm)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_sum = 0
    grad_norm_epoch = 0
    
    for ximu, xgps, y in train_loader:
        ximu, xgps, y = ximu.to(DEVICE), xgps.to(DEVICE), y.to(DEVICE)
        gps_mask = torch.ones((xgps.size(0), 1), device=DEVICE)
        drop = (torch.rand((xgps.size(0), 1), device=DEVICE) < GPS_DROP_PROB).float()
        gps_mask = gps_mask * (1.0 - drop)
        
        pred = model(ximu, xgps * gps_mask, gps_mask)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        grad_norm_epoch += total_norm.item()
        
        optimizer.step()
        tr_sum += loss.item()
    
    avg_train_loss = tr_sum / len(train_loader)
    val_loss, val_acc = eval_metrics(val_loader)
    avg_grad       = grad_norm_epoch / len(train_loader)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['grads'].append(avg_grad)
    
    print(f"Ep {epoch:02d} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# =========================
# 10) PLOT SUR LA SÉQUENCE 0018 (Celle du VAL)
# =========================
TARGET_PART = "2011_09_26_drive_0018_sync_p0" 

print(f"\n🔍 Affichage de la séquence : {TARGET_PART}")

mask_part = (meta_val[:, 0] == TARGET_PART)
if np.sum(mask_part) == 0:
    potential = [m for m in np.unique(meta_val[:,0]) if "0018" in str(m)]
    if potential:
        TARGET_PART = potential[0]
        mask_part = (meta_val[:, 0] == TARGET_PART)
    else:
        mask_part = (meta_val[:, 0] == meta_val[0,0])

Ximu_test = Ximu_val[mask_part]
Xgps_test = Xgps_val[mask_part]
y_test    = y_val[mask_part]
n = len(Ximu_test)

b0, b1 = int(n * 0.3), int(n * 0.7)
gps_mask = np.ones((n,1), dtype=np.float32)
gps_mask[b0:b1] = 0.0

model.eval()
with torch.no_grad():
    ximu_t = torch.from_numpy(Ximu_test).to(DEVICE)
    xgps_t = torch.from_numpy(Xgps_test).to(DEVICE)
    m_t    = torch.from_numpy(gps_mask).to(DEVICE)
    pred = model(ximu_t, xgps_t*m_t, m_t).cpu().numpy()

def denorm_y(y_norm): return y_norm * y_std + y_mean
pred_den, gt_den = denorm_y(pred), denorm_y(y_test)

# Reconstruction
x_gt, y_gt = np.cumsum(gt_den[:,0]), np.cumsum(gt_den[:,1])
x_pred, y_pred = np.cumsum(pred_den[:,0]), np.cumsum(pred_den[:,1])

# --- CALCUL DES MÉTRIQUES ---
pos_errors = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
ate = np.mean(pos_errors)

dist_totale = np.sum(np.sqrt(gt_den[:,0]**2 + gt_den[:,1]**2))
erreur_finale = pos_errors[-1]
drift_pct = (erreur_finale / dist_totale) * 100

yaw_gt = np.cumsum(gt_den[:, 3])
yaw_pred = np.cumsum(pred_den[:, 3])
angle_errors = np.abs(yaw_gt - yaw_pred)
mae_ang = np.degrees(np.mean(angle_errors))

print("\n" + "="*40)
print(f"📊 RÉSULTATS DÉTAILLÉS (Séquence {TARGET_PART})")
print(f"="*40)
print(f"📍 ATE (Erreur Moyenne) : {ate:.3f} m")
print(f"🧭 Erreur Moyenne Angle : {mae_ang:.3f} deg")
print(f"📉 Drift Final          : {drift_pct:.2f} %")
print(f"🎯 Accuracy Finale      : {history['val_acc'][-1]:.2f} % (Tolérance {ACCURACY_THRESHOLD}m)")
print(f"="*40)

# --- GRAPHIQUE 1 : TRAJECTOIRE AVEC MÉTRIQUES ---
plt.figure(figsize=(10, 8))
plt.plot(x_gt, y_gt, 'g-', linewidth=3, label="Vraie Route (GPS)")
plt.plot(x_pred[:b0], y_pred[:b0], 'b--', linewidth=2, label="IA (GPS ON)")
plt.plot(x_pred[b0-1:b1+1], y_pred[b0-1:b1+1], 'r-', linewidth=4, label="IA (IMU Seul - Blackout)")
plt.plot(x_pred[b1:], y_pred[b1:], 'b--', linewidth=2)
plt.title(f"Fusion Result : {TARGET_PART}\nATE: {ate:.2f}m | Drift: {drift_pct:.2f}% | Val Acc: {history['val_acc'][-1]:.1f}%")
plt.xlabel("Est (m)"); plt.ylabel("Nord (m)")
plt.axis('equal'); plt.grid(True); plt.legend()
plt.savefig("fusion_metrics_trajectory.png")
plt.show()

# --- GRAPHIQUE 2 : LOSS & ACCURACY ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Courbes de Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True, alpha=0.3); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], 'green', marker='.')
plt.title(f"Accuracy Validation (Tolérance {ACCURACY_THRESHOLD}m)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_stats.png")
plt.show()

# Export ONNX
print("\n📦 EXPORT ONNX...")
model.to("cpu").eval()
dummy_ximu = torch.randn(1, SEQ_LEN, len(IMU_COLS))
dummy_xgps = torch.randn(1, len(Y_COLS))
dummy_mask = torch.ones(1, 1)
torch.onnx.export(model, (dummy_ximu, dummy_xgps, dummy_mask), "fusion_model.onnx", opset_version=12, input_names=["imu", "gps", "mask"], output_names=["out"], dynamic_axes={'imu':{0:'b'}, 'gps':{0:'b'}, 'mask':{0:'b'}, 'out':{0:'b'}})
print("✅ ONNX Ready.")