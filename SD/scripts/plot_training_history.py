#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_float_column(rows, key):
    values = []
    for row in rows:
        raw = row.get(key, "")
        values.append(float(raw) if raw != "" else float("nan"))
    return values


def main():
    parser = argparse.ArgumentParser(description="Plot SD unlearning training losses.")
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    with args.history_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {args.history_csv}")

    steps = _read_float_column(rows, "step")
    total_loss = _read_float_column(rows, "total_loss")
    forget_loss = _read_float_column(rows, "forget_loss")
    remain_loss = _read_float_column(rows, "remain_loss")

    out = args.out or args.history_csv.with_name("training_losses.png")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_loss, label="total_loss", linewidth=1.8)
    plt.plot(steps, forget_loss, label="forget_loss", linewidth=1.4)
    plt.plot(steps, remain_loss, label="remain_loss", linewidth=1.4)
    plt.xlabel("step")
    plt.ylabel("loss")
    if args.title:
        plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
