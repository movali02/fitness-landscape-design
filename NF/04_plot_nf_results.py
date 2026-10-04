#!/usr/bin/env python3
"""
04_plot_nf_results.py

Plot outputs from the NPE + Neural Spline Flow pipeline.

Expected:
  nf_results/posterior/posterior_samples.csv
  nf_results/posterior/posterior_summary.csv
  nf_results/posterior/conditioning_target.csv

Optional validation outputs:
  nf_results/validation/posterior_predictive_landscapes.csv
  nf_results/validation/posterior_predictive_summary.csv

Usage:
  python 04_plot_nf_results.py
"""

from pathlib import Path
from itertools import combinations
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_phi(label):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(label))
    if m is None:
        raise ValueError(f"Could not extract phenotype from {label}")
    return float(m.group())


def plot_target(conditioning_file, outdir):
    df = pd.read_csv(conditioning_file)

    phi = df["phi"].to_numpy(dtype=float)
    f = df["F_target"].to_numpy(dtype=float)

    order = np.argsort(phi)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(phi[order], f[order], marker="o")
    ax.set_xlabel("Focal phenotype lambda")
    ax.set_ylabel("Target fitness")
    ax.set_title("Target fitness landscape")

    fig.tight_layout()
    fig.savefig(
        outdir / "01_target_fitness_landscape.png",
        dpi=300
    )
    plt.close(fig)


def plot_marginals(samples_file, outdir):
    df = pd.read_csv(samples_file)
    theta_cols = [c for c in df.columns if c.startswith("theta_")]

    for col in theta_cols:
        name = col.replace("theta_", "")
        values = df[col].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins=50)
        ax.set_xlabel(name)
        ax.set_ylabel("Posterior sample count")
        ax.set_title(f"Posterior distribution: {name}")
        fig.tight_layout()
        fig.savefig(outdir / f"02_posterior_hist_{name}.png", dpi=300)
        plt.close(fig)

        if np.all(values > 0):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(np.log10(values), bins=50)
            ax.set_xlabel(f"log10({name})")
            ax.set_ylabel("Posterior sample count")
            ax.set_title(f"Posterior distribution in log space: {name}")
            fig.tight_layout()
            fig.savefig(outdir / f"03_posterior_hist_log10_{name}.png", dpi=300)
            plt.close(fig)


def plot_pairwise(samples_file, outdir):
    df = pd.read_csv(samples_file)
    theta_cols = [c for c in df.columns if c.startswith("theta_")]

    if len(df) > 5000:
        df = df.sample(5000, random_state=123)

    pairdir = outdir / "pairwise"
    pairdir.mkdir(exist_ok=True)

    for xcol, ycol in combinations(theta_cols, 2):
        xname = xcol.replace("theta_", "")
        yname = ycol.replace("theta_", "")
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=8, alpha=0.25)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"Posterior relationship: {xname} vs {yname}")
        fig.tight_layout()
        fig.savefig(pairdir / f"{xname}_vs_{yname}.png", dpi=300)
        plt.close(fig)

        if np.all(x > 0) and np.all(y > 0):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(np.log10(x), np.log10(y), s=8, alpha=0.25)
            ax.set_xlabel(f"log10({xname})")
            ax.set_ylabel(f"log10({yname})")
            ax.set_title(f"Posterior relationship in log space: {xname} vs {yname}")
            fig.tight_layout()
            fig.savefig(pairdir / f"log10_{xname}_vs_log10_{yname}.png", dpi=300)
            plt.close(fig)


def plot_intervals(summary_file, outdir):
    df = pd.read_csv(summary_file)
    median = df["median"].to_numpy(dtype=float)
    lo95 = df["q2.5"].to_numpy(dtype=float) / median
    hi95 = df["q97.5"].to_numpy(dtype=float) / median
    lo50 = df["q25"].to_numpy(dtype=float) / median
    hi50 = df["q75"].to_numpy(dtype=float) / median
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(df)):
        ax.plot([lo95[i], hi95[i]], [y[i], y[i]], linewidth=2)
        ax.plot([lo50[i], hi50[i]], [y[i], y[i]], linewidth=6)
        ax.plot(1.0, y[i], marker="o")

    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Parameter value / posterior median")
    ax.set_title("Posterior uncertainty by environmental parameter")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outdir / "04_posterior_credible_intervals.png", dpi=300)
    plt.close(fig)


def plot_validation(validation_dir, conditioning_file, outdir):
    pred_file = validation_dir / "posterior_predictive_landscapes.csv"
    sum_file = validation_dir / "posterior_predictive_summary.csv"

    if not pred_file.exists():
        print("Validation files not found; skipping validation plots.")
        return
    target = pd.read_csv(conditioning_file)

    phi = target["phi"].to_numpy(dtype=float)
    ftarget = target["F_target"].to_numpy(dtype=float)

    order = np.argsort(phi)

    pred = pd.read_csv(pred_file)
    fcols = [c for c in pred.columns if c.startswith("F_")]

    fig, ax = plt.subplots(figsize=(8, 5))
    shown = 0
    for _, row in pred.iterrows():
        if "valid" in pred.columns and str(row["valid"]).lower() not in ("true", "1"):
            continue
        vals = row[fcols].to_numpy(dtype=float)
        ax.plot(phi[order], vals[order], alpha=0.10)
        shown += 1
        if shown >= 100:
            break

    ax.plot(phi[order], ftarget[order], marker="o", linewidth=3, label="Target")
    ax.set_xlabel("Focal phenotype lambda")
    ax.set_ylabel("Fitness")
    ax.set_title("Posterior predictive landscapes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "05_posterior_predictive_landscapes.png", dpi=300)
    plt.close(fig)

    if sum_file.exists():
        summary = pd.read_csv(sum_file)
        if "rmse" in summary.columns:
            rmse = summary["rmse"].dropna().to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(rmse, bins=40)
            ax.set_xlabel("Landscape RMSE")
            ax.set_ylabel("Validated posterior sample count")
            ax.set_title("Posterior predictive error")
            fig.tight_layout()
            fig.savefig(outdir / "06_posterior_predictive_rmse.png", dpi=300)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posterior-dir", default="nf_results/posterior")
    p.add_argument("--validation-dir", default="nf_results/validation")
    p.add_argument("--outdir", default="nf_results/plots")
    args = p.parse_args()

    posterior_dir = Path(args.posterior_dir)
    validation_dir = Path(args.validation_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples_file = posterior_dir / "posterior_samples.csv"
    summary_file = posterior_dir / "posterior_summary.csv"
    conditioning_file = posterior_dir.parent / "target_landscape.csv"

    for f in (samples_file, summary_file, conditioning_file):
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    plot_target(conditioning_file, outdir)
    plot_marginals(samples_file, outdir)
    plot_pairwise(samples_file, outdir)
    plot_intervals(summary_file, outdir)
    plot_validation(validation_dir, conditioning_file, outdir)

    print(f"Plots written to: {outdir}")


if __name__ == "__main__":
    main()
