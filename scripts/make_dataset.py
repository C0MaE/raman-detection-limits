import numpy as np
import yaml
from raman_detection.generate import generate_dataset

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["seed"])
    X, y, snr = generate_dataset(cfg, rng)
    np.savez("data/dataset.npz", X=X, y=y, snr=snr)

if __name__ == "__main__":
    main()