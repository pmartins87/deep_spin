# -*- coding: utf-8 -*-
"""sanity_check.py

Sanity smoke-test for the Spin&Go NoLimitHoldem C++ env (cpoker) + scenario sampler.

Goals (fast, <~30s):
- Detect "all-in but board < 5" (showdown runout bug)
- Action distribution split by to_call==0 vs to_call>0
- Raise/bet frequency over time
- Validate pot/chip accounting never goes negative or absurd

Usage (Windows CMD):
  venv\Scripts\python.exe -u scripts\sanity_check.py --seconds 25 --hands 5000

Obs:
- This does NOT train anything. It just plays many random-ish hands quickly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np


A_FOLD = 0
A_CHECK_CALL = 1
A_RAISE_33 = 2
A_RAISE_HALF = 3
A_RAISE_75 = 4
A_RAISE_POT = 5
A_ALL_IN = 6

ACTION_NAMES = {
    A_FOLD: "FOLD",
    A_CHECK_CALL: "CHECK_CALL",
    A_RAISE_33: "RAISE_33_POT",
    A_RAISE_HALF: "RAISE_HALF_POT",
    A_RAISE_75: "RAISE_75_POT",
    A_RAISE_POT: "RAISE_POT",
    A_ALL_IN: "ALL_IN",
}


def _setup_paths() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_dir = os.path.join(root, "env")
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _choose_action(rng: np.random.Generator, legal: list[int], to_call: int) -> int:
    """Heuristic random policy to stress various actions."""
    legal_set = set(legal)

    raises = [a for a in (A_RAISE_33, A_RAISE_HALF, A_RAISE_75, A_RAISE_POT) if a in legal_set]
    can_allin = A_ALL_IN in legal_set

    # Never fold for free in sanity runs (we still track if it's legal).
    if to_call == 0:
        candidates = [a for a in legal if a != A_FOLD]
        if raises and rng.random() < 0.60:
            return int(rng.choice(raises))
        if can_allin and rng.random() < 0.05:
            return A_ALL_IN
        return A_CHECK_CALL if A_CHECK_CALL in candidates else int(rng.choice(candidates))

    # Facing a bet/raise
    r = float(rng.random())
    if A_CHECK_CALL in legal_set and r < 0.55:
        return A_CHECK_CALL
    if A_FOLD in legal_set and r < 0.70:
        return A_FOLD
    if raises and r < 0.95:
        return int(rng.choice(raises))
    if can_allin:
        return A_ALL_IN
    # fallback
    return int(rng.choice(legal))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    _setup_paths()

    import cpoker  # noqa: E402
    from deepcfr.scenario import ScenarioSampler, default_env_config, choose_dealer_id_for_episode  # noqa: E402

    rng = np.random.default_rng(args.seed)
    scen = ScenarioSampler(default_env_config, seed=args.seed)

    g = cpoker.PokerGame(3, int(args.seed))

    t0 = time.time()
    hand_i = 0

    # Metrics
    actions_call0 = Counter()
    actions_callpos = Counter()
    legal_call0 = Counter()
    legal_callpos = Counter()

    raise_bins = []  # list of (hands_end, raise_count, action_count)
    cur_bin_raise = 0
    cur_bin_actions = 0

    allin_board_lt5 = 0
    allin_hands = 0

    terminal_board_hist = Counter()

    accounting_errors = 0
    accounting_first_error = None

    while hand_i < args.hands and (time.time() - t0) < args.seconds:
        # Sample a scenario
        is_hu, stacks, sb, bb = scen.sample()
        dealer_id = choose_dealer_id_for_episode(rng, is_hu)

        # Normalize stacks to 3 seats already ensured by sampler
        g.reset(stacks, int(dealer_id), int(sb), int(bb))

        # Track initial total chips
        s0 = g.get_state(g.get_player_id())
        raw0 = s0["raw_obs"]
        init_total = int(sum(raw0["in_chips"]) + sum(raw0["remained_chips"]))

        # Play hand
        while not g.is_over():
            pid = int(g.get_player_id())
            st = g.get_state(pid)
            raw = st["raw_obs"]
            legal = list(st["raw_legal_actions"])  # ints

            in_chips = list(raw["in_chips"])
            rem = list(raw["remained_chips"])
            pot = int(raw["pot"])

            # to_call in chips (cumulative difference)
            mx = max(in_chips)
            to_call = int(mx - in_chips[pid])

            # Accounting checks (fast):
            try:
                if pot < 0:
                    raise ValueError(f"pot<0: {pot}")
                if any(x < 0 for x in in_chips):
                    raise ValueError(f"in_chips<0: {in_chips}")
                if any(x < 0 for x in rem):
                    raise ValueError(f"remained<0: {rem}")
                if pot != sum(in_chips):
                    raise ValueError(f"pot!=sum(in_chips): pot={pot} sum={sum(in_chips)}")
                if pot > init_total:
                    raise ValueError(f"pot>init_total: pot={pot} init_total={init_total}")
                # Each seat should conserve chips during the hand: in_chips+remained <= initial_total
                if sum(in_chips) + sum(rem) != init_total:
                    raise ValueError(f"sum(in)+sum(rem) != init_total: {sum(in_chips)}+{sum(rem)} vs {init_total}")
            except Exception as e:
                accounting_errors += 1
                if accounting_first_error is None:
                    accounting_first_error = (hand_i, pid, str(e), raw)

            # Record legal/action distribution split by to_call
            if to_call == 0:
                for a in legal:
                    legal_call0[a] += 1
            else:
                for a in legal:
                    legal_callpos[a] += 1

            a = _choose_action(rng, legal, to_call)

            if to_call == 0:
                actions_call0[a] += 1
            else:
                actions_callpos[a] += 1

            if a in (A_RAISE_33, A_RAISE_HALF, A_RAISE_75, A_RAISE_POT):
                cur_bin_raise += 1
            cur_bin_actions += 1

            g.step(int(a))

        # Terminal checks
        pid0 = int(g.get_player_id())
        stT = g.get_state(pid0)
        rawT = stT["raw_obs"]
        board_len = int(len(rawT["public_cards"]))
        terminal_board_hist[board_len] += 1

        remT = list(rawT["remained_chips"])
        allin_ct = sum(1 for x in remT if int(x) == 0)
        if allin_ct >= 2:
            allin_hands += 1
            if board_len < 5:
                allin_board_lt5 += 1

        # bins each 200 hands
        hand_i += 1
        if hand_i % 200 == 0:
            raise_bins.append((hand_i, cur_bin_raise, cur_bin_actions))
            cur_bin_raise = 0
            cur_bin_actions = 0

    # Final bin
    if cur_bin_actions > 0:
        raise_bins.append((hand_i, cur_bin_raise, cur_bin_actions))

    dur = time.time() - t0

    def fmt_action_counter(c: Counter) -> str:
        total = sum(c.values())
        parts = []
        for k, v in sorted(c.items()):
            name = ACTION_NAMES.get(k, str(k))
            pct = (100.0 * v / total) if total else 0.0
            parts.append(f"{name}:{v} ({pct:.1f}%)")
        return ", ".join(parts) if parts else "(vazio)"

    def fmt_legal_counter(c: Counter) -> str:
        total = sum(c.values())
        parts = []
        for k, v in sorted(c.items()):
            name = ACTION_NAMES.get(k, str(k))
            pct = (100.0 * v / total) if total else 0.0
            parts.append(f"{name}:{pct:.1f}%")
        return ", ".join(parts) if parts else "(vazio)"

    print("\n=== SANITY CHECK (cpoker + scenario) ===")
    print(f"Hands played: {hand_i} in {dur:.2f}s  (target seconds={args.seconds})")

    print("\n[Terminal board length histogram]")
    for k in sorted(terminal_board_hist.keys()):
        v = terminal_board_hist[k]
        pct = 100.0 * v / max(1, hand_i)
        print(f"  len(public_cards)={k}: {v} ({pct:.2f}%)")

    print("\n[All-in runout correctness]")
    if allin_hands == 0:
        print("  No hands detected with >=2 all-in players (increase seconds/hands to stress this).")
    else:
        pct = 100.0 * allin_board_lt5 / allin_hands
        print(f"  all-in hands (>=2 all-in): {allin_hands}")
        print(f"  all-in with board<5: {allin_board_lt5}  ({pct:.4f}%)")

    print("\n[Legal actions availability] (percent share of legal-action occurrences)")
    print(f"  to_call==0: {fmt_legal_counter(legal_call0)}")
    print(f"  to_call>0 : {fmt_legal_counter(legal_callpos)}")

    print("\n[Chosen actions distribution] (our heuristic policy, split by to_call)")
    print(f"  to_call==0: {fmt_action_counter(actions_call0)}")
    print(f"  to_call>0 : {fmt_action_counter(actions_callpos)}")

    print("\n[Raise frequency over time] (per ~200 hands bin)")
    for hands_end, r_ct, a_ct in raise_bins:
        pct = 100.0 * r_ct / max(1, a_ct)
        print(f"  up to hand {hands_end}: raises={r_ct} / actions={a_ct} ({pct:.2f}%)")

    print("\n[Accounting checks]")
    if accounting_errors == 0:
        print("  OK: no errors detected")
    else:
        print(f"  ERRORS: {accounting_errors}")
        if accounting_first_error is not None:
            hi, pid, msg, raw = accounting_first_error
            print(f"  First error at hand={hi}, pid={pid}: {msg}")
            print(f"  raw_obs snapshot: {raw}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
