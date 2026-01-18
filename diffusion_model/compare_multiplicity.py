import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

# Import from your training script
from dm_train import MomentumDenoiser, DiffusionSchedule, EPairsDataset, sample_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt")
    ap.add_argument("--pkl", type=str, required=True, help="Path to real events pkl")
    ap.add_argument("--n_gen", type=int, default=200, help="Number of generated events to sample")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--outdir", type=str, default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/multiplicity_check")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins (auto range)")
    ap.add_argument("--use_steps", type=int, default=-1, help="Override sampling steps; -1 uses ckpt T")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"device={device}")

    # ----- load real data -----
    with open(args.pkl, "rb") as f:
        real_events = pickle.load(f)

    # real multiplicity
    real_N = np.array([len(np.asarray(ev["pdg"])) for ev in real_events], dtype=np.int32)
    print(f"Real events loaded: {len(real_events)}")
    print(f"Real N stats: min={real_N.min()} max={real_N.max()} mean={real_N.mean():.3f} median={np.median(real_N):.1f}")

    # ----- load checkpoint -----
    ckpt = torch.load(args.ckpt, map_location=device)
    T = int(ckpt["sched_T"])
    kmax = int(ckpt["kmax"])
    steps = T if args.use_steps < 0 else int(args.use_steps)

    model = MomentumDenoiser(
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sched = DiffusionSchedule(T=T).to(device)

    # Dataset is only used for Option-A PDG/multiplicity sampling
    dataset = EPairsDataset(real_events, Kmax=kmax)

    # ----- generate -----
    gen_N = []
    for i in range(args.n_gen):
        pdg_id, p_out = sample_event(model, dataset, sched, device=device, steps=steps)
        gen_N.append(len(p_out))
        if (i + 1) % max(1, args.n_gen // 10) == 0:
            print(f"Generated {i+1}/{args.n_gen} events...")

    gen_N = np.array(gen_N, dtype=np.int32)
    print(f"\nGenerated events: {len(gen_N)}")  # <-- 你要的“生成的数据数量”
    print(f"Gen N stats:  min={gen_N.min()} max={gen_N.max()} mean={gen_N.mean():.3f} median={np.median(gen_N):.1f}")

    # ----- quick table: top multiplicities -----
    print("\nTop multiplicities (REAL):", Counter(real_N.tolist()).most_common(10))
    print("Top multiplicities (GEN): ", Counter(gen_N.tolist()).most_common(10))

    # ----- histogram -----
    # Use shared range so plot is comparable
    nmin = int(min(real_N.min(), gen_N.min()))
    nmax = int(max(real_N.max(), gen_N.max()))
    bins = np.linspace(nmin, nmax + 1, args.bins + 1)

    plt.figure()
    plt.hist(real_N, bins=bins, alpha=0.6, density=True, label="Real")
    plt.hist(gen_N, bins=bins, alpha=0.6, density=True, label="Generated")
    plt.xlabel("Multiplicity N(e±) per event")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Multiplicity comparison | real={len(real_N)}, gen={len(gen_N)} | Kmax={kmax} | steps={steps}")
    out_png = os.path.join(args.outdir, "multiplicity_hist.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved histogram: {out_png}")

    # Save raw arrays too (方便你后续画别的图)
    np.save(os.path.join(args.outdir, "real_N.npy"), real_N)
    np.save(os.path.join(args.outdir, "gen_N.npy"), gen_N)
    print(f"Saved arrays: {args.outdir}/real_N.npy and gen_N.npy")


if __name__ == "__main__":
    main()