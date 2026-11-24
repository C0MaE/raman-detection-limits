from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from raman_detection import baseline, evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. Config und Datensatz laden (X, y, snr, nu)
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    split = cfg["split"]
    dataset = cfg["dataset"]

    data = np.load("../data/dataset.npz")
    X, y, snr, nu = data["X"], data["y"], data["snr"], data["nu"]
    # 2. Aufteilen mit evaluate.split_dataset, Werte aus cfg["split"]
    X_tr, X_te, y_tr, y_te, snr_tr, snr_te = evaluate.split_dataset(X, y, snr, test_size=split["test_size"], seed=split["seed"])

    # 3. Teststatistik s für Train und Test
    s_tr = baseline.detection_statistic(X=X_tr, nu=nu, cfg=cfg)
    s_te = baseline.detection_statistic(X=X_te, nu=nu, cfg=cfg)

    # 4. Schwelle nur aus den Train-Spektren ohne Signal
    thr = baseline.calibrate_threshold(s_tr[y_tr == 0], fpr=cfg["baseline"]["fpr"])

    # 5. Kontrolle: Fehlalarmrate auf den Test-Spektren ohne Signal
    fpr_test = np.mean(s_te[y_te == 0] > thr)
    print(f"Threshold: {thr:.3f}, false positive rate on test set: {fpr_test:.2%}")

    # 6. Effizienz auf den Test-Spektren mit Signal
    #    (sig, snr_sig, detected wie besprochen; bins aus np.geomspace)
    lo, hi = dataset["snr_range"]  # 0.5 und 20
    sig = y_te == 1
    snr_sig = snr_te[sig]
    detected = s_te[sig] > thr
    bins = np.geomspace(lo, hi, cfg["evaluation"]["n_bins"] + 1)
    eps, sigma_eps = evaluate.efficiency_vs_snr(snr_sig, detected, bins)
    centers = np.sqrt(bins[:-1] * bins[1:])

    # 7. Zahlen speichern, damit der Gesamtplot später alle Modelle zusammen zeigen kann
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "efficiency_baseline.npz",
             centers=centers, eps=eps, sigma_eps=sigma_eps, threshold=thr, fpr_test=fpr_test)

    # 8. Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(centers, eps, yerr=sigma_eps, fmt="o-", capsize=3, label="Baseline (no ML)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Detection efficiency")
    ax.set_title("Detection efficiency at 1 % false positive rate")
    ax.legend()
    fig.savefig(out_dir / "efficiency_baseline.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()