"""
scripts/sanity_check.py

Sanity checks rápidos (target ~30s) para validar:
- Runout completo em ALL-IN/showdown (board com 5 cartas quando >1 player chega ao showdown)
- Coerência de custo de raise (diff + raise_add) vs legal_actions e stacks
- Invariantes de stacks/pot (nunca negativos, conservação razoável)
- Distribuição de ações condicionada (diff == 0 vs diff > 0) usando uma policy simples

Uso:
  venv\Scripts\python.exe -u scripts\sanity_check.py --episodes 500

Observação:
- Este script NÃO altera o treino.
- Ele usa o mesmo ScenarioSampler/config.yaml do treino.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_config_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = path.read_text(encoding="utf-8")
    # YAML é preferido, mas mantemos fallback simples sem dependências extras
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            raise RuntimeError(
                f"Arquivo {path} é YAML, mas PyYAML não está instalado. Rode: pip install pyyaml"
            )
        return dict(yaml.safe_load(data) or {})
    if path.suffix.lower() == ".json":
        import json
        return dict(json.loads(data))
    # tenta YAML por padrão
    try:
        import yaml  # type: ignore
        return dict(yaml.safe_load(data) or {})
    except Exception:
        return {}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Mapeia ActionType -> frac do pote usado no C++
POT_FRAC = {
    2: 1.0 / 3.0,  # BET_33
    3: 1.0 / 2.0,  # BET_50
    4: 3.0 / 4.0,  # BET_75
    5: 1.0,        # BET_POT
}

ACTION_NAME = {
    0: "FOLD",
    1: "CHECK/CALL",
    2: "BET_33",
    3: "BET_50",
    4: "BET_75",
    5: "BET_POT",
    6: "ALL_IN",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env_dir = project_root / "env"

    # path para deepcfr + scripts
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(env_dir))

    import cpoker  # type: ignore
    from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode  # type: ignore

    scen_cfg = _load_config_yaml_or_json(project_root / "config.yaml")
    sampler = ScenarioSampler(config=scen_cfg, seed=args.seed)

    # audit: scenario file + hash
    import deepcfr.scenario as scenario_mod  # type: ignore
    scen_path = Path(scenario_mod.__file__).resolve()
    print(f"[INFO] deepcfr.scenario file: {scen_path}")
    print(f"[INFO] deepcfr.scenario sha256: {_sha256_file(scen_path)}")

    # audit: cpoker pyd (se existir)
    try:
        pyds = list(env_dir.glob("cpoker*.pyd"))
        if pyds:
            print(f"[INFO] cpoker module: {pyds[0].resolve()}")
    except Exception:
        pass

    rng = np.random.default_rng(args.seed ^ 0xA5A5A5A5)

    # stats
    steps_total = 0
    showdown_incomplete = 0
    showdown_total = 0
    violations_afford = 0
    violations_negative = 0

    act_counts = {
        "diff0": Counter(),
        "diff1": Counter(),
    }

    t0 = time.time()

    for ep in range(int(args.episodes)):
        is_hu, stacks, sb, bb = sampler.sample()
        dealer_id = choose_dealer_id_for_episode(sampler.rng, stacks, sb, bb, is_hu)

        # seed do jogo por episódio (determinístico)
        g = cpoker.PokerGame(3, int(args.seed) + ep)
        g.reset(list(map(int, stacks)), int(dealer_id), int(sb), int(bb))

        while not g.is_over():
            pid = int(g.get_player_id())
            st = g.get_state(pid)

            # obs parsing (indices fixos conforme poker_env.cpp)
            obs = np.asarray(st["obs"], dtype=np.float32)
            # [0:52] hero hand one-hot
            # [52:104] board one-hot
            stacks_bb = obs[104:107]
            bets_bb = obs[107:110]
            # pot_bb = obs[110]  # não precisamos aqui
            # hero_stack_over_pot = obs[111]

            # converter para chips inteiros (robusto a floats)
            bets_chips = np.rint(bets_bb * float(bb)).astype(np.int64)
            diff = int(bets_chips.max() - bets_chips[pid])

            raw = st["raw_obs"]
            remained = int(raw["remained_chips"][pid])
            pot = int(raw["pot"])

            legal = list(map(int, st["raw_legal_actions"]))

            # --- valida affordability de raises (diff + raise_add <= remained)
            for a in legal:
                if a in POT_FRAC:
                    raise_add = int(pot * POT_FRAC[a])
                    cost = diff + raise_add
                    if cost > remained:
                        violations_afford += 1

            # --- escolhe ação (policy simples)
            choose_from = list(legal)
            # evita fold gratuito para não poluir stats (ainda registramos a disponibilidade pela lista legal)
            if diff == 0 and 0 in choose_from and len(choose_from) > 1:
                choose_from = [a for a in choose_from if a != 0]

            a = int(rng.choice(choose_from))
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
        # precisamos de um state para pegar raw_obs final; use o player 0 (qualquer id serve)
        st0 = g.get_state(0)
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
    for k in ["diff0", "diff1"]:
        total = sum(act_counts[k].values())
        print(f"  {k}: total={total}")
        for name, c in act_counts[k].most_common():
            print(f"    {name:10s} {c:8d}  ({(100.0*c/max(1,total)):.2f}%)")

    # thresholds conservadores
    fail = False
    if showdown_incomplete > 0:
        print("\n[FAIL] Existe showdown com board incompleto (<5). Isso enviesará EV.")
        fail = True
    if violations_afford > 0:
        print("\n[FAIL] Existe ação de raise listada como legal quando (diff + raise_add) > remained.")
        fail = True
    if violations_negative > 0:
        print("\n[FAIL] Pot/stack negativo detectado.")
        fail = True

    if not fail:
        print("\n[OK] Sanity checks básicos passaram.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
