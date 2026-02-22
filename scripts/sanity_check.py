"""scripts/sanity_check.py

Sanity checks rápidos (target ~30s) para validar:

- Runout completo em ALL-IN/showdown (board com 5 cartas quando >1 player chega ao showdown)
- Coerência de custo de raise (diff + raise_add) vs legal_actions e stacks
- Invariantes de stacks/pot (nunca negativos)
- Distribuição de ações condicionada (diff == 0 vs diff > 0) usando uma policy simples

IMPORTANTE:
- Seu deepcfr/scenario.py (na branch atual) NÃO exporta default_env_config.
  O treino (scripts/train_deepcfr.py) instancia ScenarioSampler(config=..., seed=...).
  Portanto, este sanity_check segue a mesma API.

Uso (Windows):
    venv/Scripts/python.exe -u scripts/sanity_check.py --episodes 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

# -----------------------------------------------------------------------------
# sys.path bootstrap: ao rodar "python scripts/xxx.py", sys.path[0] vira "scripts/".
# Precisamos inserir a raiz do projeto para importar "deepcfr", e "env/" para cpoker.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ENV_DIR = ROOT / "env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))


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
# - BET_33 = floor(pot * 33 / 100)
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

    def _choice(seq, p=None):
        if p is None:
            return rng.choice(seq)  # type: ignore[attr-defined]
        return rng.choices(seq, weights=p, k=1)[0]  # type: ignore[attr-defined]

    return _choice


def _load_config_yaml_or_json(path: Path) -> dict:
    """Mantém compat com seu train_deepcfr.py: tenta YAML e depois JSON."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    # YAML (opcional)
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return dict(data or {})
    except Exception:
        pass
    # JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f) or {})
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    # Import local project (após sys.path bootstrap acima)
    from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode

    try:
        import cpoker  # type: ignore
    except Exception:
        import importlib

        cpoker = importlib.import_module("env.cpoker")  # type: ignore

    # Auditoria: arquivo e hash de scenario importado
    try:
        import deepcfr.scenario as scenario_mod  # type: ignore

        scen_file = Path(scenario_mod.__file__).resolve()
        print(f"[INFO] deepcfr.scenario file: {scen_file}")
        print(f"[INFO] deepcfr.scenario sha256: {_sha256_file(scen_file)}")
    except Exception as e:
        print(f"[WARN] Não consegui resolver __file__/sha256 do scenario: {e}")

    try:
        print(f"[INFO] cpoker module: {Path(cpoker.__file__).resolve()}")
    except Exception:
        print("[INFO] cpoker module: <unknown>")

    scen_cfg = _load_config_yaml_or_json(ROOT / "config.yaml")

    # ScenarioSampler (API atual do repo)
    sampler = ScenarioSampler(config=scen_cfg, seed=int(args.seed))

    rng = np.random.default_rng(int(args.seed))
    rng_choice = _pick_choice_rng(rng)

    steps_total = 0
    showdown_total = 0
    showdown_incomplete = 0
    violations_afford = 0
    violations_preflop_no_raise = 0
    violations_negative = 0

    act_counts: Dict[str, Counter] = {"diff0": Counter(), "diff1": Counter()}

    t0 = time.time()

    for ep in range(int(args.episodes)):
        scen = sampler.sample()

        if not isinstance(scen, dict):
            raise RuntimeError("ScenarioSampler.sample() deveria retornar dict nesta versão do repo.")

        is_hu = bool(scen.get("game_is_hu", False))
        stacks = list(map(int, scen.get("stacks", [])))
        sb = int(scen.get("sb", 10))
        bb = int(scen.get("bb", 20))

        dealer_id = int(choose_dealer_id_for_episode(sampler.rng, stacks, sb, bb, is_hu))

        # seed do jogo por episódio (determinístico)
        g = cpoker.PokerGame(3, int(args.seed) + ep)
        if hasattr(g, "set_seed"):
            g.set_seed(int(args.seed) + ep)
        g.reset(stacks, dealer_id, int(sb), int(bb))

        # Necessário: raw_obs só é incluído se debug_raw_obs estiver ativo no C++ env.
        if hasattr(g, "set_debug_raw_obs"):
            g.set_debug_raw_obs(True)

        while not g.is_over():
            pid = int(g.get_player_id())
            st: Dict[str, Any] = g.get_state(pid)

            # raw_obs é a fonte correta para diff/pot/stacks
            raw = st["raw_obs"]

            in_chips = list(map(int, raw["in_chips"]))
            diff = int(max(in_chips) - in_chips[pid])

            remained = int(raw["remained_chips"][pid])
            pot = int(raw["pot"])
            stage = int(raw.get("stage", -1))

            legal = list(map(int, st["raw_legal_actions"]))

            # Pré-flop: se não consegue cobrir o call (diff >= remained), NÃO pode existir raise nem ALL_IN.
            # CHECK/CALL cobre o call-all-in via clamp.
            if stage == 0 and diff > 0 and diff >= remained:
                for a in legal:
                    if a in (2, 3, 4, 5, 6):  # raise labels + all-in
                        violations_preflop_no_raise += 1
                        break

            # valida affordability de raises (diff + raise_add <= remained)
            # OBS: no pré-flop (stage==0) os rótulos RAISE_* são BB-based (open/iso/3bet abstraídos),
            # não pot-fraction. Portanto, este check de pot-fraction só é válido pós-flop (stage>0).
            if stage > 0:
                for a in legal:
                    if a in POT_RAISE_ADD:
                        raise_add = int(POT_RAISE_ADD[a][1](pot))
                        cost = diff + raise_add
                        if cost > remained:
                            violations_afford += 1

            # policy simples para stats
            choose_from = list(legal)
            if diff == 0 and 0 in choose_from and len(choose_from) > 1:
                # evita fold gratuito para não poluir distribuição
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

        if num_not_folded > 1:
            showdown_total += 1
            if len(pub) < 5:
                showdown_incomplete += 1

    dt = time.time() - t0

    print("\n===== SANITY CHECK REPORT =====")
    print(f"Episodes: {args.episodes}")
    print(f"Steps total: {steps_total}  |  steps/episode: {steps_total / max(1,args.episodes):.2f}")
    print(f"Runtime: {dt:.2f}s")

    p = (100.0 * showdown_incomplete / showdown_total) if showdown_total else 0.0
    print(f"Showdown: {showdown_total}  |  showdown com board<5: {showdown_incomplete} ({p:.3f}%)")

    print(f"Violations (raise affordability, postflop pot-fraction): {violations_afford}")
    print(f"Violations (preflop raise when cannot call): {violations_preflop_no_raise}")
    print(f"Violations (negative pot/stack): {violations_negative}")

    print("\nAções escolhidas (policy simples) por diff:")
    for k in ("diff0", "diff1"):
        c = act_counts[k]
        tot = sum(c.values())
        print(f"  {k}: total={tot}")
        for name, cnt in c.most_common():
            print(f"    {name:<14} {cnt:6d}  ({(100.0*cnt/max(1,tot)):.2f}%)")

    if violations_afford > 0:
        print("\n[FAIL] Pós-flop: existe raise pot-fraction legal quando (diff + raise_add) > remained.")
        return 2

    if violations_preflop_no_raise > 0:
        print("\n[FAIL] Pré-flop: existe raise/all-in legal quando o jogador não cobre o call (diff >= remained).")
        return 3

    print("\n[OK] Nenhuma violação detectada (postflop affordability + preflop no-raise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())