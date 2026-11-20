from pathlib import Path

import numpy as np
import yaml

from raman_detection.generate import generate_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    X, y, snr, nu = generate_dataset(cfg, rng)

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "dataset.npz", X=X, y=y, snr=snr, nu=nu)

    print(f"Saved {X.shape[0]} spectra ({y.sum()} with signal) to {out_dir / 'dataset.npz'}")


if __name__ == "__main__":
    main()