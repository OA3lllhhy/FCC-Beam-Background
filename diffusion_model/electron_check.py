import pickle
import numpy as np
from collections import Counter

pkl_path = "/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl"

with open(pkl_path, "rb") as f:
    events = pickle.load(f)

print("Loaded events:", len(events))

# 每个 event 的 e-/e+ 数量
n_tot = []
n_em = []
n_ep = []

for ev in events:
    pdg = np.asarray(ev["pdg"], dtype=np.int32)
    n_em.append(int(np.sum(pdg == 11)))
    n_ep.append(int(np.sum(pdg == -11)))
    n_tot.append(int(len(pdg)))

print("\nMultiplicity (e± total) summary:")
print("  min:", np.min(n_tot))
print("  max:", np.max(n_tot))
print("  mean:", np.mean(n_tot))
print("  median:", np.median(n_tot))

print("\nElectron counts summary:")
print("  e-  mean/median/min/max:", np.mean(n_em), np.median(n_em), np.min(n_em), np.max(n_em))
print("  e+  mean/median/min/max:", np.mean(n_ep), np.median(n_ep), np.min(n_ep), np.max(n_ep))

print("\nTop multiplicities (total e±):")
print(Counter(n_tot).most_common(20))

# e-/e+ 差值分布（看是否强烈 charge asymmetry）
diff = np.array(n_em) - np.array(n_ep)
print("\n(e- - e+) diff summary:")
print("  min/max/mean/median:", diff.min(), diff.max(), diff.mean(), np.median(diff))
print("Top diffs:", Counter(diff.tolist()).most_common(10))