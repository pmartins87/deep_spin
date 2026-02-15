# scripts/hand_history_debug.py
# Debug de mãos com log estilo hand history (PT4-like)
# - Determinístico por padrão (seed fixo)
# - Opcional: seed automático por execução (--seed auto)

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# -----------------------------------------------------------------------------
# sys.path bootstrap: ao rodar via scripts/, garantir imports do projeto e env/
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ENV_DIR = ROOT / "env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades, hearts, diamonds, clubs


def card_str(card_idx: int) -> str:
    if card_idx is None or int(card_idx) < 0:
        return "??"
    c = int(card_idx)
    suit = c // 13
    rank = c % 13
    return f"{RANKS[rank]}{SUITS[suit]}"


def action_name(a: int) -> str:
    return {
        0: "FOLD",
        1: "CHECK/CALL",
        2: "BET_33",
        3: "BET_50",
        4: "BET_75",
        5: "BET_POT",
        6: "ALL_IN",
    }.get(int(a), f"ACT_{a}")


def street_name(round_counter: int) -> str:
    rc = int(round_counter)
    return ["PREFLOP", "FLOP", "TURN", "RIVER", "DONE"][min(max(rc, 0), 4)]


def _load_config_yaml_or_json(path: Path) -> dict:
    """Tenta YAML e depois JSON (compat com o fluxo do treino)."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return dict(data or {})
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f) or {})
    except Exception:
        return {}


def _choose_dealer_id_fallback(choose_fn, rng, stacks, sb, bb, is_hu) -> int:
    """Assinatura de choose_dealer_id_for_episode variou no projeto.
    Tentamos chamadas comuns em ordem.
    """
    tries = [
        lambda: choose_fn(rng, stacks, sb, bb, is_hu),
        lambda: choose_fn(rng, stacks, sb, bb),
        lambda: choose_fn(rng, stacks),
        lambda: choose_fn(rng, is_hu),
        lambda: choose_fn(rng),
    ]
    last_err = None
    for t in tries:
        try:
            return int(t())
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Falha ao chamar choose_dealer_id_for_episode com fallbacks. Último erro: {last_err}")


def _get_legal_actions(game, st: Dict[str, Any]) -> List[int]:
    if "raw_legal_actions" in st:
        return list(map(int, st["raw_legal_actions"]))
    if "legal_actions" in st:
        return list(map(int, st["legal_actions"]))
    if hasattr(game, "get_legal_actions"):
        return list(map(int, game.get_legal_actions()))
    return []


def _select_action(policy: str, legal: List[int], rng: np.random.Generator, diff_chips: int) -> int:
    if not legal:
        return 1
    legal = list(map(int, legal))

    if policy == "random":
        return int(rng.choice(legal))

    if policy == "no_free_fold":
        # se diff==0 removemos fold, mas só se existir outra ação
        if diff_chips == 0 and 0 in legal and len(legal) > 1:
            legal = [a for a in legal if a != 0]
        return int(rng.choice(legal))

    if policy == "passive":
        pref = [1, 2, 3, 4, 5, 6, 0]
        for a in pref:
            if a in legal:
                return a
        return legal[0]

    if policy == "aggressive":
        pref = [6, 5, 4, 3, 2, 1, 0]
        for a in pref:
            if a in legal:
                return a
        return legal[0]

    return int(rng.choice(legal))


def _parse_seed(seed_arg: str) -> int:
    """seed pode ser int ou 'auto'."""
    if seed_arg.strip().lower() == "auto":
        # entropia suficiente p/ variar entre execuções, mas ainda reprodutível se você logar o valor
        return int(time.time_ns() ^ (os.getpid() << 16)) & 0x7FFFFFFF
    return int(seed_arg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=str, default="123", help="int ou 'auto' para variar a cada execução")
    ap.add_argument("--policy", type=str, default="random",
                    choices=["random", "passive", "aggressive", "no_free_fold"])
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    seed0 = _parse_seed(args.seed)

    # Imports do projeto (após sys.path bootstrap)
    from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode

    try:
        import cpoker  # type: ignore
    except Exception:
        import importlib
        cpoker = importlib.import_module("env.cpoker")  # type: ignore

    cfg = _load_config_yaml_or_json(ROOT / "config.yaml")

    sampler = ScenarioSampler(config=cfg, seed=int(seed0))
    rng = np.random.default_rng(int(seed0))

    for ep in range(int(args.episodes)):
        # usar seeds diferentes por mão, mantendo reprodutibilidade (seed base + ep)
        hand_seed = int(seed0) + ep

        scen = sampler.sample()
        if not isinstance(scen, dict):
            raise RuntimeError("ScenarioSampler.sample() deveria retornar dict nesta versão.")

        is_hu = bool(scen.get("is_heads_up", False))
        stacks = list(map(int, scen.get("stacks", [])))
        sb = int(scen.get("sb", 10))
        bb = int(scen.get("bb", 20))

        dealer_id = _choose_dealer_id_fallback(
            choose_dealer_id_for_episode,
            getattr(sampler, "rng", rng),
            stacks,
            sb,
            bb,
            is_hu,
        )

        if len(stacks) != 3:
            while len(stacks) < 3:
                stacks.append(0)
            stacks = stacks[:3]

        g = cpoker.PokerGame(3, hand_seed)
        g.reset(stacks, int(dealer_id), int(sb), int(bb))

        print("\n" + "=" * 110)
        print(f"HAND #{ep + 1}/{args.episodes}  | seed={hand_seed} (base={seed0})  | HU={is_hu}  | dealer={dealer_id}  | blinds={sb}/{bb}")
        print(f"Stacks (chips): {stacks}")

        hole_all: List[List[int]] = []
        for pid in range(3):
            stp = g.get_state(pid)
            rawp = stp.get("raw_obs", {})
            hole = list(map(int, rawp.get("hand", [])))
            hole_all.append(hole)

        print("Hole cards (debug):")
        for pid in range(3):
            h = hole_all[pid]
            hs = " ".join(card_str(c) for c in h) if h else "(unknown)"
            print(f"  P{pid}: {hs}")

        step_id = 0
        last_board: List[int] = []

        while not g.is_over():
            pid = int(g.get_player_id())
            st: Dict[str, Any] = g.get_state(pid)

            raw = st.get("raw_obs", {})
            obs = np.asarray(st.get("obs", []), dtype=np.float32)

            pot = int(raw.get("pot", 0))
            remained = list(map(int, raw.get("remained_chips", [0, 0, 0])))
            round_counter = int(raw.get("round_counter", 0))

            board = list(map(int, raw.get("public_cards", [])))
            if board != last_board:
                last_board = board
                bs = " ".join(card_str(c) for c in board) if board else "(empty)"
                print(f"\n[{street_name(round_counter)}] BOARD: {bs}")

            diff_chips = 0
            to_call_bb = 0.0
            if obs.size >= 110:
                bets_bb = obs[107:110]
                bets_chips = np.rint(bets_bb * float(bb)).astype(np.int64)
                diff_chips = int(bets_chips.max() - bets_chips[pid])
                to_call_bb = float(diff_chips) / float(bb) if bb > 0 else 0.0

            legal = _get_legal_actions(g, st)
            a = _select_action(args.policy, legal, rng, diff_chips)

            pot_bb = float(pot) / float(bb) if bb > 0 else 0.0
            stack_bb = float(remained[pid]) / float(bb) if bb > 0 else 0.0

            print(
                f"step={step_id:03d}  street={street_name(round_counter):7s}  pid={pid}  "
                f"action={action_name(a):10s}  "
                f"to_call_bb={to_call_bb:6.3f}  pot_bb={pot_bb:7.3f}  stack_bb={stack_bb:7.3f}  "
                f"remained={remained}  legal={[action_name(x) for x in legal]}"
            )

            g.step(int(a))
            step_id += 1
            if step_id > 500:
                print("[ABORT] Muitos steps (possível loop).")
                break

        pay = list(map(float, g.get_payoffs()))
        st0 = g.get_state(0)
        raw0 = st0.get("raw_obs", {})
        board_final = list(map(int, raw0.get("public_cards", [])))
        bs_final = " ".join(card_str(c) for c in board_final) if board_final else "(empty?)"

        print("\n--- RESULT ---")
        print(f"BOARD FINAL: {bs_final}")
        for pid in range(3):
            hs = " ".join(card_str(c) for c in hole_all[pid]) if hole_all[pid] else "(unknown)"
            print(f"P{pid}  hole={hs:<10s}  payoff={pay[pid]:.1f}")
        print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
