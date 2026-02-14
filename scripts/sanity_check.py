"""scripts/sanity_check.py

Sanity checks rápidos (target ~30s) para validar:

- Runout completo em ALL-IN/showdown (board com 5 cartas quando >1 player chega ao showdown)
- Coerência de custo de raise (diff + raise_add) vs legal_actions e stacks
- Invariantes de stacks/pot (nunca negativos, conservação razoável)
- Distribuição de ações condicionada (diff == 0 vs diff > 0) usando uma policy simples

Uso (Windows):
    venv/Scripts/python.exe -u scripts/sanity_check.py --episodes 500
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ActionType enum (C++): mantenha consistente com poker_env.h
ACTION_NAME = {
    0: "FOLD",
    1: "CHECK/CALL",
    2: "BET_33",
    3: "BET_50",
    4: "BET_75",
    5: "BET_POT",
    6: "ALL_IN",
}

# Pot-fraction raise_add EXACT (determinístico, sem float):
# - BET_33 = floor(pot * 33 / 100)  (mesmo critério do C++)
# - BET_50 = floor(pot / 2)
# - BET_75 = floor(pot * 75 / 100)
# - BET_POT = pot
POT_RAISE_ADD = {
    2: ("33/100", lambda pot: (pot * 33) // 100),
    3: ("1/2", lambda pot: pot // 2),
    4: ("75/100", lambda pot: (pot * 75) // 100),
    5: ("1/1", lambda pot: pot),
}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pick_choice_rng(rng):
    """Retorna rng.choice para numpy, ou um wrapper compatível."""
    if hasattr(rng, "choice"):
        return rng.choice
    # random.Random fallback
    import random  # noqa: F401

    def _choice(seq, p=None):
        if p is None:
            # type: ignore[attr-defined]
            return rng.choice(seq)
        # type: ignore[attr-defined]
        return rng.choices(seq, weights=p, k=1)[0]

    return _choice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    # Import local project
    from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode, default_env_config

    try:
        import cpoker  # type: ignore
    except Exception:
        # Alguns setups colocam cpoker dentro de env/
        import importlib

        cpoker = importlib.import_module("env.cpoker")  # type: ignore

    scen_file = Path(__import__("deepcfr.scenario").scenario.__file__).resolve()  # type: ignore[attr-defined]
    print(f"[INFO] deepcfr.scenario file: {scen_file}")
    print(f"[INFO] deepcfr.scenario sha256: {_sha256_file(scen_file)}")
    try:
        print(f"[INFO] cpoker module: {Path(cpoker.__file__).resolve()}")
    except Exception:
        print("[INFO] cpoker module: <unknown>")

    # ScenarioSampler
    sampler = ScenarioSampler(default_env_config)
    rng = np.random.default_rng(int(args.seed))
    rng_choice = _pick_choice_rng(rng)

    steps_total = 0
    showdown_total = 0
    showdown_incomplete = 0
    violations_afford = 0
    violations_negative = 0

    act_counts: Dict[str, Counter] = {"diff0": Counter(), "diff1": Counter()}

    t0 = time.time()

    for ep in range(int(args.episodes)):
        scen = sampler.sample()

        # ScenarioSampler.sample() atual retorna dict.
        # Mantemos fallback para versões antigas que retornavam tupla.
        if isinstance(scen, dict):
            is_hu = bool(scen.get("is_heads_up", False))
            stacks = list(map(int, scen.get("stacks", [])))
            sb = int(scen.get("sb", 10))
            bb = int(scen.get("bb", 20))
        else:
            is_hu, stacks, sb, bb = scen  # compat antiga
            stacks = list(map(int, stacks))

        dealer_id = int(choose_dealer_id_for_episode(sampler.rng, stacks, sb, bb, is_hu))

        # seed do jogo por episódio (determinístico)
        g = cpoker.PokerGame(3, int(args.seed) + ep)
        g.reset(stacks, dealer_id, int(sb), int(bb))

        while not g.is_over():
            pid = int(g.get_player_id())
            st: Dict[str, Any] = g.get_state(pid)

            obs = np.asarray(st["obs"], dtype=np.float32)

            # Layout fixo conforme poker_env.cpp
            # [0:52] hero hand one-hot
            # [52:104] board one-hot
            # [104:107] stacks (bb)
            # [107:110] current_bets (round_.raised) (bb)
            # [110] pot (bb)
            bets_bb = obs[107:110]

            # converter para chips inteiros (robusto a floats)
            bets_chips = np.rint(bets_bb * float(bb)).astype(np.int64)
            diff = int(bets_chips.max() - bets_chips[pid])

            raw = st["raw_obs"]
            remained = int(raw["remained_chips"][pid])
            pot = int(raw["pot"])

            legal = list(map(int, st["raw_legal_actions"]))

            # --- valida affordability de raises (diff + raise_add <= remained)
            for a in legal:
                if a in POT_RAISE_ADD:
                    raise_add = int(POT_RAISE_ADD[a][1](pot))
                    cost = diff + raise_add
                    if cost > remained:
                        violations_afford += 1

            # --- escolhe ação (policy simples)
            choose_from = list(legal)

            # Evita fold gratuito para não poluir stats (ainda registramos a disponibilidade pela lista legal).
            if diff == 0 and 0 in choose_from and len(choose_from) > 1:
                choose_from = [a for a in choose_from if a != 0]

            a = int(rng_choice(choose_from))
            act_counts["diff0" if diff == 0 else "diff1"][ACTION_NAME.get(a, str(a))] += 1

            g.step(a)
            steps_total += 1

            # invariantes básicos durante a mão
            if pot < 0:
                violations_negative += 1
            for x in raw["remained_chips"]:
                if int(x) < 0:
                    violations_negative += 1
                    break

        # terminal checks
        st0: Dict[str, Any] = g.get_state(0)
        raw0 = st0["raw_obs"]
        pub = raw0["public_cards"]
        num_not_folded = int(raw0.get("num_not_folded", 0))

        # showdown = mais de 1 player não foldou
        if num_not_folded > 1:
            showdown_total += 1
            if len(pub) < 5:
                showdown_incomplete += 1

    dt = time.time() - t0

    print("\n===== SANITY CHECK REPORT =====")
    print(f"Episodes: {args.episodes}")
    print(f"Steps total: {steps_total}  |  steps/episode: {steps_total / max(1,args.episodes):.2f}")
    print(f"Runtime: {dt:.2f}s")

    if showdown_total:
        p = 100.0 * showdown_incomplete / showdown_total
    else:
        p = 0.0
    print(f"Showdown: {showdown_total}  |  showdown com board<5: {showdown_incomplete} ({p:.3f}%)")

    print(f"Violations (raise affordability): {violations_afford}")
    print(f"Violations (negative pot/stack): {violations_negative}")

    print("\nAções escolhidas (policy simples) por diff:")
    for k in ("diff0", "diff1"):
        c = act_counts[k]
        tot = sum(c.values())
        print(f"  {k}: total={tot}")
        for name, cnt in c.most_common():
            print(f"    {name:<14} {cnt:6d}  ({(100.0*cnt/max(1,tot)):.2f}%)")

    if violations_afford > 0:
        print("\n[FAIL] Existe ação de raise listada como legal quando (diff + raise_add) > remained.")
        return 2

    print("\n[OK] Nenhuma violação de affordability detectada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
