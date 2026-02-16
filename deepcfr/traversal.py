from __future__ import annotations

import os
import numpy as np

NUM_PLAYERS = 3
NUM_ACTIONS = 7

# OBS_DIM must match the C++ environment's obs vector length.
# `scripts/train_deepcfr.py` should set SPIN_OBS_DIM before importing deepcfr.*.
OBS_DIM = int(os.environ.get("SPIN_OBS_DIM", "338"))

# One-time log to confirm whether we are using fast clone-based traversal.
_CLONE_LOGGED = False


def legal_mask_from_state(state) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in state.get("raw_legal_actions", []):
        ai = int(a)
        if 0 <= ai < NUM_ACTIONS:
            mask[ai] = 1.0
    return mask


def regret_matching(adv: np.ndarray, legal_mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Strategy = normalize(ReLU(adv)) over legal actions.
    If all <=0, uniform over legal.
    """
    adv = np.asarray(adv, dtype=np.float32)
    lm = np.asarray(legal_mask, dtype=np.float32)

    r = np.maximum(adv, 0.0) * lm
    s = float(r.sum())
    if s > eps:
        return r / s

    cnt = float(lm.sum())
    if cnt <= 0:
        return np.ones((NUM_ACTIONS,), dtype=np.float32) / float(NUM_ACTIONS)
    return lm / cnt


def sample_action(rng: np.random.Generator, sigma: np.ndarray, raw_legal_actions: list[int], greedy: bool = False) -> int:
    """
    Sample (or greedy-pick) an action restricted to raw_legal_actions.
    Returns an integer in [0..6].
    """
    legal = [int(a) for a in raw_legal_actions]
    if not legal:
        return 1  # CHECK/CALL as a safe fallback

    if greedy:
        best_a = legal[0]
        best_p = float(sigma[best_a])
        for a in legal[1:]:
            p = float(sigma[a])
            if p > best_p:
                best_p = p
                best_a = a
        return int(best_a)

    probs = np.array([float(max(0.0, sigma[a])) for a in legal], dtype=np.float64)
    s = float(probs.sum())
    if s <= 0.0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs /= s
    return int(rng.choice(np.array(legal, dtype=np.int64), p=probs))


class EpisodeSpec:
    __slots__ = ("ep_seed", "stacks", "dealer_id", "sb", "bb", "game_is_hu")

    def __init__(self, ep_seed: int, stacks: List[int], dealer_id: int, sb: int, bb: int, game_is_hu: bool = False):
        self.ep_seed = int(ep_seed)
        self.stacks = list(map(int, stacks))
        self.dealer_id = int(dealer_id)
        self.sb = int(sb)
        self.bb = int(bb)
        self.game_is_hu = bool(game_is_hu)



class Replayer:
    """
    Deterministic replayer for history-based tree traversal.
    Fallback path when cpoker does NOT provide clone/snapshot.
    """
    def __init__(self, cpoker_module, base_game_seed: int = 0):
        self.cpoker = cpoker_module
        self.base_game_seed = int(base_game_seed)

    def make_game(self, spec: EpisodeSpec):
        g = self.cpoker.PokerGame(NUM_PLAYERS, self.base_game_seed)
        if hasattr(g, "set_seed"):
            g.set_seed(int(spec.ep_seed))
        g.reset(spec.stacks, spec.dealer_id, spec.sb, spec.bb)
        return g

    def replay_to_node(self, spec: EpisodeSpec, history_actions: list[int]):
        g = self.make_game(spec)
        for a in history_actions:
            g.step(int(a))
            if g.is_over():
                break
        if g.is_over():
            return g, -1, None

        p = int(g.get_game_pointer())
        s = g.get_state(p)

        # Defensive copies: avoids rare native crashes if the underlying memory is mutated.
        obs = np.array(s["obs"], dtype=np.float32, copy=True)
        s["obs"] = np.ascontiguousarray(obs, dtype=np.float32)
        if isinstance(s.get("legal_actions", None), dict):
            s["legal_actions"] = dict(s["legal_actions"])
        if isinstance(s.get("raw_legal_actions", None), list):
            s["raw_legal_actions"] = list(s["raw_legal_actions"])
        return g, p, s


def build_sigma_from_advnet(advnet, obs: np.ndarray, legal_mask_or_raw, device=None, torch=None) -> np.ndarray:
    """
    Flexible helper used by different trainer versions.

    Accepts either:
      - legal_mask_or_raw = np.ndarray shape (7,) with 0/1 legal mask
      - legal_mask_or_raw = list[int] raw_legal_actions

    Returns sigma (7,) using regret matching on advnet(obs).
    """
    if isinstance(legal_mask_or_raw, np.ndarray):
        legal_mask = legal_mask_or_raw.astype(np.float32, copy=False)
    else:
        legal_mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        for a in legal_mask_or_raw or []:
            ai = int(a)
            if 0 <= ai < NUM_ACTIONS:
                legal_mask[ai] = 1.0

    if torch is None:
        import torch as _torch
        torch = _torch

    if device is None:
        device = "cpu"

    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        adv = advnet(obs_t).view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
    return regret_matching(adv, legal_mask)


class ExternalSamplingTraverser:
    """
    External Sampling MCCFR-style traversal.

    Performance note:
      - If cpoker provides PokerGame.clone(), we use a clone-based traversal that avoids
        O(N^2) replay_to_node overhead. This is the recommended path.
      - Otherwise, we fall back to replay_to_node (slower).
    """
    def __init__(
        self,
        replayer: Replayer | None = None,
        *,
        cpoker_module=None,
        base_game_seed: int = 0,
        adv_nets: list | None = None,
        adv_buffers: list | None = None,
        traverser_id: int | None = None,
        rng: np.random.Generator | None = None,
        device=None,
        torch=None,
    ):
        if replayer is None:
            if cpoker_module is None:
                raise TypeError("ExternalSamplingTraverser: provide replayer=... OR cpoker_module=...")
            replayer = Replayer(cpoker_module, base_game_seed=base_game_seed)

        if torch is None:
            import torch as _torch
            torch = _torch

        self.replayer = replayer
        self.adv_nets = adv_nets
        self.adv_buffers = adv_buffers
        self.traverser_id = None if traverser_id is None else int(traverser_id)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.device = device if device is not None else "cpu"
        self.torch = torch

        # feature-detect clone path
        try:
            g0 = self.replayer.cpoker.PokerGame(NUM_PLAYERS, 0)
            self._has_clone = hasattr(g0, "clone")
            global _CLONE_LOGGED
            if not _CLONE_LOGGED:
                # Em workers (ProcessPool), suprime logs para evitar spam.
                # O processo principal imprime o status logo no início.
                if os.environ.get("SPIN_WORKER") != "1":
                    if self._has_clone:
                        print("[INFO] CLONE ENABLED (PokerGame.clone detected)")
                    else:
                        print("[WARN] CLONE DISABLED (fallback to replay, slow)")
                _CLONE_LOGGED = True
        except Exception:
            self._has_clone = False

    def _terminal_value(self, g, traverser: int, spec: EpisodeSpec) -> float:
        pay = g.get_payoffs()
        return float(pay[traverser]) / float(max(1, spec.bb))

    def run_episode(self, stacks, dealer_id, sb, bb, *, is_hu: bool = False, ep_seed: int | None = None) -> float:
        if self.traverser_id is None:
            raise RuntimeError("run_episode requires traverser_id to be set in the constructor.")
        if ep_seed is None:
            ep_seed = int(self.rng.integers(0, 2**31 - 1))
        spec = EpisodeSpec(ep_seed=int(ep_seed), stacks=list(stacks), dealer_id=int(dealer_id), sb=int(sb), bb=int(bb), is_hu=bool(is_hu))
        return self.traverse(spec, traverser=self.traverser_id, history_actions=[])

    # ---------------------------
    # Fast path: clone traversal
    # ---------------------------
    def _traverse_game(self, g, spec: EpisodeSpec, traverser: int) -> float:
        if g.is_over():
            return self._terminal_value(g, traverser, spec)

        p = int(g.get_game_pointer())
        state = g.get_state(p)

        # Defensive copy of obs (stability with native buffers)
        obs = np.array(state["obs"], dtype=np.float32, copy=True)
        raw_legal = list(state.get("raw_legal_actions", []))
        legal_mask = legal_mask_from_state({"raw_legal_actions": raw_legal})

        if legal_mask.sum() <= 0:
            return self._terminal_value(g, traverser, spec)

        sigma_p = build_sigma_from_advnet(self.adv_nets[p], obs, legal_mask, device=self.device, torch=self.torch)

        if p != traverser:
            a = sample_action(self.rng, sigma_p, raw_legal_actions=raw_legal, greedy=False)
            g2 = g.clone()
            g2.step(int(a))
            return float(self._traverse_game(g2, spec, traverser))

        # traverser node: branch all legal actions
        action_values = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        v = 0.0
        for a in raw_legal:
            ai = int(a)
            g2 = g.clone()
            g2.step(ai)
            va = self._traverse_game(g2, spec, traverser)
            action_values[ai] = float(va)
            v += float(sigma_p[ai]) * float(va)

        regret_target = (action_values - float(v)) * legal_mask

        self.adv_buffers[traverser].add(
            obs=obs,
            legal_mask=legal_mask,
            target=regret_target,
            player_id=traverser
        )
        return float(v)

    # ---------------------------
    # Fallback path: replay_to_node
    # ---------------------------
    def _traverse_replay(self, spec: EpisodeSpec, traverser: int, history_actions: list[int]) -> float:
        g, p, state = self.replayer.replay_to_node(spec, history_actions)
        if p == -1:
            return self._terminal_value(g, traverser, spec)

        obs = state["obs"]
        raw_legal = state.get("raw_legal_actions", [])
        legal_mask = legal_mask_from_state(state)

        if legal_mask.sum() <= 0:
            return self._terminal_value(g, traverser, spec)

        sigma_p = build_sigma_from_advnet(self.adv_nets[p], obs, legal_mask, device=self.device, torch=self.torch)

        if p != traverser:
            a = sample_action(self.rng, sigma_p, raw_legal_actions=raw_legal, greedy=False)
            history_actions.append(a)
            v = self._traverse_replay(spec, traverser, history_actions)
            history_actions.pop()
            return float(v)

        action_values = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        v = 0.0
        for a in raw_legal:
            ai = int(a)
            history_actions.append(ai)
            va = self._traverse_replay(spec, traverser, history_actions)
            history_actions.pop()
            action_values[ai] = float(va)
            v += float(sigma_p[ai]) * float(va)

        regret_target = (action_values - float(v)) * legal_mask
        self.adv_buffers[traverser].add(obs=obs, legal_mask=legal_mask, target=regret_target, player_id=traverser)
        return float(v)

    def traverse(self, spec: EpisodeSpec, traverser: int, history_actions: list[int]) -> float:
        if self._has_clone and not history_actions:
            g = self.replayer.make_game(spec)
            return self._traverse_game(g, spec, traverser)
        return self._traverse_replay(spec, traverser, history_actions)