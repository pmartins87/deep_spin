import os, sys, time, math, random
import numpy as np

# Force cpoker import from ./env
ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(ROOT, "env")
sys.path.insert(0, ENV_DIR)

import cpoker


ACTIONS = {
    0: "FOLD",
    1: "CHECK/CALL",
    2: "BET_33",
    3: "BET_50",
    4: "BET_75",
    5: "BET_POT",
    6: "ALL_IN",
}

def detect_obs_dim():
    g = cpoker.PokerGame(3, 123)
    g.reset([500,500,500], 0, 10, 20)
    p = int(g.get_game_pointer())
    s = g.get_state(p)
    obs = np.array(s["obs"], dtype=np.float32)
    return int(obs.shape[0])

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)

def get_legal_mask_from_state(s):
    la = s.get("raw_legal_actions", [])
    mask = np.zeros((7,), dtype=np.float32)
    for a in la:
        if 0 <= int(a) <= 6:
            mask[int(a)] = 1.0
    return mask

def extract_legal_mask_from_obs(obs):
    # last 7 dims are legal mask by design
    return obs[-7:].copy()

def step_random_legal(g):
    p = int(g.get_game_pointer())
    s = g.get_state(p)
    la = s.get("raw_legal_actions", [])
    _assert(isinstance(la, list) and len(la) > 0, "No legal actions returned")
    a = int(random.choice(la))
    g.step(a)
    return a, s

def check_seed_determinism():
    # Same seed, same reset => identical first K transitions (obs + legal actions)
    K = 50
    seed = 777
    stacks = [500, 500, 500]
    dealer = 0

    def rollout_signature():
        g = cpoker.PokerGame(3, seed)
        g.reset(stacks, dealer, 10, 20)
        sig = []
        for _ in range(K):
            p = int(g.get_game_pointer())
            s = g.get_state(p)
            obs = np.array(s["obs"], dtype=np.float32)
            la = tuple(int(x) for x in s.get("raw_legal_actions", []))
            # Use a stable random policy with fixed rng:
            # choose smallest legal action for determinism
            a = int(min(la)) if len(la) else 1
            sig.append((p, obs.tobytes(), la, a))
            g.step(a)
            if bool(g.is_over()):
                break
        return sig

    sig1 = rollout_signature()
    sig2 = rollout_signature()
    _assert(len(sig1) == len(sig2), f"Seed determinism: length mismatch {len(sig1)} vs {len(sig2)}")
    for i in range(len(sig1)):
        _assert(sig1[i][0] == sig2[i][0], f"Seed determinism: player mismatch at {i}")
        _assert(sig1[i][2] == sig2[i][2], f"Seed determinism: legal mismatch at {i}")
        _assert(sig1[i][3] == sig2[i][3], f"Seed determinism: action mismatch at {i}")
        _assert(sig1[i][1] == sig2[i][1], f"Seed determinism: obs mismatch at {i}")
    print("[OK] seed determinism")

def check_hu_3h_integrity():
    # HU should be represented as a dead seat, not "3-handed with a fold that changes later"
    # We'll create a HU game by starting with one stack at 0, then ensure that player never acts.
    seed = 1234
    g = cpoker.PokerGame(3, seed)
    g.reset([500, 500, 0], 0, 10, 20)  # seat2 dead
    seen_to_act = set()
    for _ in range(200):
        p = int(g.get_game_pointer())
        seen_to_act.add(p)
        _assert(p != 2, "HU integrity failed: dead seat (2) got to act")
        s = g.get_state(p)
        la = s.get("raw_legal_actions", [])
        _assert(isinstance(la, list) and len(la) > 0, "HU integrity: no legal actions")
        # play deterministically
        g.step(int(min(la)))
        if bool(g.is_over()):
            break
    _assert(2 not in seen_to_act, "HU integrity failed: dead seat appeared in to_act")
    print("[OK] HU/3H integrity (dead seat never acts)")

def check_preflop_sizing_rules():
    # This doesn't inspect internal C++ bb-sizes, but checks that legal action labels are consistent
    # with your design: preflop allows only subsets depending on ctx10-like state.
    # We approximate ctx by pot and by presence of BET_33/50/75 in legal set.
    seed = 2026
    g = cpoker.PokerGame(3, seed)
    g.reset([500,500,500], 0, 10, 20)

    # Explore many preflop nodes with random legal actions
    for _ in range(400):
        p = int(g.get_game_pointer())
        s = g.get_state(p)
        raw = s.get("raw_legal_actions", [])
        obs = np.array(s["obs"], dtype=np.float32)

        # infer street from street one-hot (positions 112..115 by your layout)
        street = int(np.argmax(obs[112:116]))
        if street == 0:  # PREFLOP
            legal = set(int(x) for x in raw)
            # In preflop, you should never see BET_POT (5) as a legal raise
            _assert(5 not in legal, f"Preflop sizing rule violated: BET_POT legal in preflop, raw={raw}")
            # Also, preflop always should include CHECK/CALL (1) or FOLD (0), except weird all-in-only spot
            _assert(0 in legal or 1 in legal or 6 in legal, f"Preflop legal set suspicious: {raw}")

        # take a random legal action to move around
        if not raw:
            break
        g.step(int(random.choice(raw)))
        if bool(g.is_over()):
            break

    print("[OK] preflop sizing/legal label sanity")

def check_payoff_conservation():
    # Ensure sum of payoffs = 0 at terminal, and chips conserved (within tolerance)
    seed = 4321
    tol = 1e-6
    for hand in range(200):
        g = cpoker.PokerGame(3, seed + hand)
        init = [500, 500, 500]
        g.reset(init, 0, 10, 20)

        steps = 0
        while not bool(g.is_over()) and steps < 500:
            p = int(g.get_game_pointer())
            s = g.get_state(p)
            raw = s.get("raw_legal_actions", [])
            _assert(raw, "No legal actions in non-terminal state")
            g.step(int(random.choice(raw)))
            steps += 1

        _assert(bool(g.is_over()), "Payoff test: hand did not finish")
        pay = g.get_payoffs()
        pay = [float(x) for x in pay]
        _assert(abs(sum(pay)) <= tol, f"Payoff conservation failed: sum(payoffs)={sum(pay)} payoffs={pay}")
    print("[OK] payoff conservation (sum=0)")

def check_obs_dim_and_legal_mask():
    seed = 999
    g = cpoker.PokerGame(3, seed)
    g.reset([500,500,500], 0, 10, 20)
    obs_dim = detect_obs_dim()

    for _ in range(200):
        p = int(g.get_game_pointer())
        s = g.get_state(p)
        obs = np.array(s["obs"], dtype=np.float32)
        _assert(obs.shape == (obs_dim,), f"obs shape mismatch: got {obs.shape}, expected {(obs_dim,)}")
        _assert(np.all(np.isfinite(obs)), "obs has NaN/inf")

        m1 = get_legal_mask_from_state(s)
        m2 = extract_legal_mask_from_obs(obs)
        _assert(np.array_equal(m1.astype(np.int32), m2.astype(np.int32)),
                f"legal_mask mismatch. from_state={m1.tolist()} from_obs={m2.tolist()} raw={s.get('raw_legal_actions', [])}")

        raw = s.get("raw_legal_actions", [])
        g.step(int(random.choice(raw)))
        if bool(g.is_over()):
            break

    print("[OK] obs_dim + legal_mask consistency")

def main():
    print("cpoker module:", cpoker.__file__)
    print("OBS_DIM detected from env:", detect_obs_dim())
    random.seed(0)
    np.random.seed(0)

    check_obs_dim_and_legal_mask()
    check_seed_determinism()
    check_hu_3h_integrity()
    check_preflop_sizing_rules()
    check_payoff_conservation()

    print("\nALL ULTRA CHECKS PASSED ✅")

if __name__ == "__main__":
    main()
