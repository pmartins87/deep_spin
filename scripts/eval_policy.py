from __future__ import annotations

import os
import sys
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Paths: project root + env (cpoker .pyd)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ENV_DIR = ROOT / "env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

# Hint threads (CPU-only)
CPU_COUNT = os.cpu_count() or 16
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))

import cpoker  # type: ignore


def detect_obs_dim() -> int:
    g = cpoker.PokerGame(3, 0)
    g.reset([500, 500, 500], 0, 10, 20)
    p = int(g.get_game_pointer())
    s = g.get_state(p)
    obs = np.asarray(s["obs"], dtype=np.float32)
    return int(obs.shape[0])


# Must be set before importing deepcfr.* modules that read SPIN_OBS_DIM
OBS_DIM = detect_obs_dim()
os.environ["SPIN_OBS_DIM"] = str(OBS_DIM)

import torch

from deepcfr.networks import PolicyNet, AdvantageNet
from deepcfr.traversal import legal_mask_from_state, regret_matching
from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode


ACTION_NAMES = {
    0: "FOLD",
    1: "CALL/CHECK",
    2: "BET33",
    3: "BET50",
    4: "BET75",
    5: "POT",
    6: "ALLIN",
}


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------
def find_training_checkpoint(root: Path, ckpt_arg: str | None) -> Path:
    """
    O seu treino (scripts/train_deepcfr.py) usa:
      checkpoints/checkpoint_last.pt
    Aqui também aceitamos checkpoint.pt e escolhemos o mais recente se ambos existirem.
    """
    if ckpt_arg:
        p = Path(ckpt_arg)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p

    cand = []
    p_last = root / "checkpoints" / "checkpoint_last.pt"
    p_std = root / "checkpoints" / "checkpoint.pt"
    if p_last.exists():
        cand.append(p_last)
    if p_std.exists():
        cand.append(p_std)

    if not cand:
        return p_last  # default

    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def load_checkpoint_retry(path: Path, tries: int = 8, sleep_s: float = 0.75) -> dict:
    """
    Evita falha quando o treino está salvando ao mesmo tempo.
    """
    last_err: Exception | None = None
    for _ in range(max(1, tries)):
        try:
            return torch.load(str(path), map_location="cpu")
        except Exception as e:
            last_err = e
            time.sleep(float(sleep_s))
    raise RuntimeError(f"Falha ao carregar checkpoint: {path} ({last_err})")


# -----------------------------------------------------------------------------
# Policy helpers
# -----------------------------------------------------------------------------
def masked_softmax_np(logits: np.ndarray, legal: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    legal = legal.astype(np.float32)
    m = legal > 0.0
    out = np.zeros((7,), dtype=np.float32)
    if not m.any():
        out[:] = 1.0 / 7.0
        return out

    z = logits.astype(np.float64) / max(1e-6, float(temperature))
    z[~m] = -1e30
    zmax = np.max(z[m])
    e = np.exp(z - zmax)
    e[~m] = 0.0
    s = float(e.sum())
    if s <= 0.0:
        out[m] = 1.0 / float(m.sum())
        return out
    return (e / s).astype(np.float32)


def pick_action(probs: np.ndarray, raw_legal_actions: list[int], rng: np.random.Generator, greedy: bool) -> int:
    legal = [int(a) for a in raw_legal_actions]
    if not legal:
        return 0
    if greedy:
        return int(max(legal, key=lambda a: float(probs[a])))

    p = np.array([float(probs[a]) for a in legal], dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return int(rng.choice(np.array(legal, dtype=int)))
    p /= s
    return int(rng.choice(np.array(legal, dtype=int), p=p))


def act_avg_policy(
    net: PolicyNet,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    raw_legal_actions: list[int],
    device: torch.device,
    rng: np.random.Generator,
    greedy: bool,
    temperature: float,
) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        x = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
        logits = net(x).squeeze(0).cpu().numpy()
    probs = masked_softmax_np(logits, legal_mask, temperature=temperature)
    a = pick_action(probs, raw_legal_actions, rng=rng, greedy=greedy)
    return a, probs


def act_current_strategy(
    advnet: AdvantageNet,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    raw_legal_actions: list[int],
    device: torch.device,
    rng: np.random.Generator,
    greedy: bool,
) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        x = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
        adv = advnet(x).squeeze(0).cpu().numpy().astype(np.float32)
    sigma = regret_matching(adv, legal_mask)
    a = pick_action(sigma, raw_legal_actions, rng=rng, greedy=greedy)
    return a, sigma


def baseline_action(baseline: str, raw_legal_actions: list[int], rng: np.random.Generator) -> int:
    legal = [int(a) for a in raw_legal_actions]
    if not legal:
        return 0

    if baseline == "random":
        return int(rng.choice(np.array(legal, dtype=int)))

    if baseline == "station":
        if 1 in legal:
            return 1
        for a in [3, 2, 4, 5, 6]:
            if a in legal:
                return a
        if 0 in legal:
            return 0
        return int(rng.choice(np.array(legal, dtype=int)))

    if baseline == "aggro":
        for a in [6, 5, 4, 3, 2, 1, 0]:
            if a in legal:
                return a
        return int(rng.choice(np.array(legal, dtype=int)))

    return int(rng.choice(np.array(legal, dtype=int)))


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
@dataclass
class Running:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2

    @property
    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class EvalOut:
    hands: int
    hu: int
    threeway: int
    bb_hand: list[Running]
    action_counts: list[np.ndarray]  # per seat: (7,)


def evaluate(
    pol_nets: list[PolicyNet],
    adv_nets: list[AdvantageNet],
    strategy: str,
    hero_seat: int | None,
    baseline: str,
    hands: int,
    greedy: bool,
    temperature: float,
    seed: int,
    progress_every: int,
) -> EvalOut:
    device = torch.device("cpu")
    rng = np.random.default_rng(int(seed))

    cfg = {}
    try:
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists() and cfg_path.stat().st_size > 0:
            import yaml  # type: ignore
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}

    sampler = ScenarioSampler(cfg, seed=int(seed) + 7)

    bb_hand = [Running() for _ in range(3)]
    action_counts = [np.zeros((7,), dtype=np.int64) for _ in range(3)]
    hu = 0
    threeway = 0

    t0 = time.time()
    for i in range(int(hands)):
        scen = sampler.sample()
        stacks = list(scen["stacks"])
        sb = int(scen["sb"])
        bb = int(scen["bb"])
        game_is_hu = bool(scen["game_is_hu"])
        if game_is_hu:
            hu += 1
        else:
            threeway += 1

        dealer = int(choose_dealer_id_for_episode(rng, stacks, sb, bb, game_is_hu))
        ep_seed = int(rng.integers(0, 2**31 - 1))

        g = cpoker.PokerGame(3, ep_seed)
        g.reset(stacks, dealer, sb, bb)

        steps = 0
        while not bool(g.is_over()) and steps < 2000:
            p = int(g.get_game_pointer())
            s = g.get_state(p)
            raw_legal = s.get("raw_legal_actions", [])
            if not raw_legal:
                break

            obs = np.array(s["obs"], dtype=np.float32, copy=True)
            obs = np.ascontiguousarray(obs, dtype=np.float32)

            legal = legal_mask_from_state(s)

            is_hero_turn = (hero_seat is None) or (p == int(hero_seat))
            if is_hero_turn:
                if strategy == "avg":
                    a, _ = act_avg_policy(pol_nets[p], obs, legal, raw_legal, device, rng, greedy, temperature)
                else:
                    a, _ = act_current_strategy(adv_nets[p], obs, legal, raw_legal, device, rng, greedy)
            else:
                a = baseline_action(baseline, raw_legal, rng)

            if 0 <= int(a) <= 6:
                action_counts[p][int(a)] += 1

            g.step(int(a))
            steps += 1

        pay = np.asarray(g.get_payoffs(), dtype=np.float64)
        pay_bb = pay / float(max(1, bb))
        for seat in range(3):
            bb_hand[seat].add(float(pay_bb[seat]))

        if progress_every > 0 and (i + 1) % int(progress_every) == 0:
            dt = time.time() - t0
            hps = (i + 1) / max(1e-9, dt)
            print(f"[progress] {i+1}/{hands} hands, {hps:.1f} hands/s")

    return EvalOut(hands=hands, hu=hu, threeway=threeway, bb_hand=bb_hand, action_counts=action_counts)


def print_eval(title: str, e: EvalOut) -> None:
    print("=" * 92)
    print(title)
    print(f"hands={e.hands}, HU={e.hu}, 3way={e.threeway}")
    print("-" * 92)
    for seat in range(3):
        mean = e.bb_hand[seat].mean
        std = e.bb_hand[seat].std
        print(f"Seat {seat}: bb/hand={mean:+.4f}, bb/100={mean*100:+.2f}, std={std:.4f}, n={e.bb_hand[seat].n}")
    print("-" * 92)
    for seat in range(3):
        total = int(e.action_counts[seat].sum())
        if total <= 0:
            continue
        pct = e.action_counts[seat] / max(1, total)
        parts = [f"{ACTION_NAMES[a]}={pct[a]*100:5.1f}%" for a in range(7)]
        print(f"Seat {seat} action% ({total} decisions): " + ", ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--strategy", choices=["avg", "current"], default="avg")
    ap.add_argument("--baseline", choices=["random", "station", "aggro"], default="random")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--all-seats", action="store_true")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    ckpt_path = find_training_checkpoint(ROOT, args.ckpt if args.ckpt else None)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ckpt_path}")

    try:
        ck = load_checkpoint_retry(ckpt_path)
    except Exception as e:
        backups = sorted(
            (ckpt_path.parent).glob(ckpt_path.stem + "_backup_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if backups:
            print(f"[WARN] Falha ao ler {ckpt_path} ({e}). Usando backup: {backups[0]}")
            ck = load_checkpoint_retry(backups[0], tries=3, sleep_s=0.5)
            ckpt_path = backups[0]
        else:
            raise

    iteration = int(ck.get("iteration", -1))

    print(f"Checkpoint carregado: {ckpt_path} (iteration={iteration})")
    print(f"OBS_DIM={OBS_DIM}, strategy={args.strategy}, greedy={args.greedy}, baseline={args.baseline}")

    pol_nets = [PolicyNet(obs_dim=OBS_DIM) for _ in range(3)]
    adv_nets = [AdvantageNet(obs_dim=OBS_DIM) for _ in range(3)]

    if "pol_nets" in ck:
        for p in range(3):
            pol_nets[p].load_state_dict(ck["pol_nets"][p])
    else:
        print("[WARN] checkpoint não tem 'pol_nets'.")

    if "adv_nets" in ck:
        for p in range(3):
            adv_nets[p].load_state_dict(ck["adv_nets"][p])
    else:
        print("[WARN] checkpoint não tem 'adv_nets'.")

    for n in pol_nets + adv_nets:
        n.eval()

    e_self = evaluate(
        pol_nets=pol_nets,
        adv_nets=adv_nets,
        strategy=args.strategy,
        hero_seat=None,
        baseline=args.baseline,
        hands=args.hands,
        greedy=args.greedy,
        temperature=args.temperature,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print_eval(f"SELF-PLAY ({args.strategy})", e_self)

    if args.all_seats:
        for hero in [0, 1, 2]:
            e = evaluate(
                pol_nets=pol_nets,
                adv_nets=adv_nets,
                strategy=args.strategy,
                hero_seat=hero,
                baseline=args.baseline,
                hands=args.hands,
                greedy=args.greedy,
                temperature=args.temperature,
                seed=args.seed + 1000 + hero,
                progress_every=args.progress_every,
            )
            print_eval(f"HERO seat={hero} vs {args.baseline} ({args.strategy})", e)
    else:
        hero = 0
        e = evaluate(
            pol_nets=pol_nets,
            adv_nets=adv_nets,
            strategy=args.strategy,
            hero_seat=hero,
            baseline=args.baseline,
            hands=args.hands,
            greedy=args.greedy,
            temperature=args.temperature,
            seed=args.seed + 1000,
            progress_every=args.progress_every,
        )
        print_eval(f"HERO seat={hero} vs {args.baseline} ({args.strategy})", e)


if __name__ == "__main__":
    main()