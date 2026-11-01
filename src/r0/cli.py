"""Command-line interface: ``r0 simulate configs/seir.yaml``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import load, simulate_deterministic, simulate_stochastic
from .model import doubling_time, growth_rate


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _band(values: np.ndarray, fmt=_fmt) -> str:
    if values.size == 1:
        return fmt(float(values[0]))
    lo, mid, hi = np.percentile(values, [5, 50, 95])
    return f"{fmt(mid)} [{fmt(lo)} ; {fmt(hi)}]"


def simulate(args: argparse.Namespace) -> int:
    model = load(args.config)
    if args.engine == "deterministic":
        res = simulate_deterministic(model, args.days, dt=args.dt)
    else:
        res = simulate_stochastic(
            model, args.days, replicates=args.replicates, seed=args.seed, dt=args.dt
        )

    cfg = model.config
    r = growth_rate(model, res.beta)
    engine = {"deterministic": "déterministe", "stochastic": "stochastique"}[res.engine]
    print(f"{cfg.name} — moteur {engine}, {args.days} jours")
    print(f"  R0 = {cfg.R0:g}   beta = {res.beta:.4g} par contact et par jour")
    print(f"  croissance initiale r = {r:.4f}/jour, doublement en {doubling_time(model, res.beta):.1f} jours")
    print(f"  population : {_fmt(model.N.sum())}")
    print(f"  part de la population infectée : {_band(res.attack_rate(), lambda v: f'{100 * v:.1f} %')}")
    if res.n_replicates > 1:
        print("  (médiane [intervalle 90 %] sur les réplications)")
    print()
    print(f"  {'compartiment':<14}{'pic (jour)':>20}{'pic (effectif)':>38}{'entrées totales':>40}")
    for name in model.compartments:
        if name == cfg.infection.susceptible:
            continue
        total_in = _band(res.inflow(name).sum(axis=1))
        if cfg.compartments[name].duration is None:
            # absorbing compartment: occupancy only grows, a peak is meaningless
            day, height = "-", "-"
        else:
            d, h = res.peak(name)
            day, height = _band(d.astype(float)), _band(h)
        print(f"  {name:<14}{day:>20}{height:>38}{total_in:>40}")
    if args.out:
        res.to_csv(args.out)
        print(f"\n  trajectoires écrites dans {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r0", description="Simulateur de propagation d'épidémie")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="simuler un modèle décrit par un fichier YAML")
    p.add_argument("config", help="fichier de configuration YAML")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--engine", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--replicates", type=int, default=20, help="réplications (moteur stochastique)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.1, help="pas de temps en jours")
    p.add_argument("--out", help="écrire les trajectoires dans ce fichier CSV")
    p.set_defaults(func=simulate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
