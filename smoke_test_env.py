import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
import cpoker

NUM_ACTIONS = 7

def legal_mask_from_state(state):
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in state.get("raw_legal_actions", []):
        ai = int(a)
        if 0 <= ai < NUM_ACTIONS:
            mask[ai] = 1.0
    return mask

def pick_random_legal(state, rng):
    acts = state.get("raw_legal_actions", [])
    return int(rng.choice(acts))

def main():
    rng = np.random.default_rng(123)
    g = cpoker.PokerGame(3, 123)

    hands = 200  # aumenta depois
    decisions = 0

    for h in range(hands):
        g.reset([500, 500, 500], int(rng.integers(0, 3)), 10, 20)

        while not g.is_over():
            p = int(g.get_game_pointer())
            state = g.get_state(p)

            obs = np.asarray(state["obs"], dtype=np.float32)
            if obs.shape != (260,):
                raise RuntimeError(f"obs shape errado: {obs.shape}")

            mask = legal_mask_from_state(state)
            if mask.sum() < 1:
                raise RuntimeError("sem ações legais, bug")

            a = pick_random_legal(state, rng)
            if mask[a] != 1.0:
                raise RuntimeError("ação escolhida não é legal, bug")

            g.step(a)
            decisions += 1

        pay = g.get_payoffs()
        if len(pay) != 3:
            raise RuntimeError("payoffs != 3, bug")

    print(f"OK. hands={hands} decisions={decisions}")

if __name__ == "__main__":
    main()
