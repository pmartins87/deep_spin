from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


G: dict[str, Any] = {}


def _set_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def init_worker(
    project_root: str,
    env_dir: str,
    obs_dim: int,
    base_game_seed: int,
    scen_cfg: dict,
    torch_threads: int,
    worker_seed: int,
) -> None:
    """
    Inicializador do ProcessPool, roda 1 vez por processo (Windows spawn).
    - Coloca paths do projeto.
    - Limita threads do worker (evita oversubscription).
    - Inicializa nets UMA vez e reutiliza, carregando pesos só quando versão muda.
    """
    _set_threads(torch_threads)
    os.environ["SPIN_WORKER"] = "1"  # usado para suprimir logs "CLONE ENABLED" nos workers

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)

    import torch
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    import cpoker
    from deepcfr.networks import AdvantageNet
    from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode
    from deepcfr.traversal import (
        ExternalSamplingTraverser,
        Replayer,
        EpisodeSpec,
        build_sigma_from_advnet,
        legal_mask_from_state,
        sample_action,
    )

    G["torch"] = torch
    G["cpoker"] = cpoker
    G["obs_dim"] = int(obs_dim)
    G["base_game_seed"] = int(base_game_seed)
    G["scen_cfg"] = dict(scen_cfg or {})
    G["ScenarioSampler"] = ScenarioSampler
    G["choose_dealer"] = choose_dealer_id_for_episode
    G["AdvantageNet"] = AdvantageNet
    G["ExternalSamplingTraverser"] = ExternalSamplingTraverser
    G["Replayer"] = Replayer
    G["EpisodeSpec"] = EpisodeSpec
    G["build_sigma_from_advnet"] = build_sigma_from_advnet
    G["legal_mask_from_state"] = legal_mask_from_state
    G["sample_action"] = sample_action

    G["rng"] = np.random.default_rng(int(worker_seed))
    G["device"] = torch.device("cpu")

    # Caches por worker
    G["adv_nets"] = [AdvantageNet(obs_dim=int(G["obs_dim"])).to(G["device"]) for _ in range(3)]
    for n in G["adv_nets"]:
        n.eval()

    G["last_weights_version"] = None
    G["last_weights_path"] = None


class _ListBuffer:
    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.legal: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.pid: list[int] = []

    def add(self, obs, legal_mask, target, player_id) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.legal.append(np.asarray(legal_mask, dtype=np.float32))
        self.target.append(np.asarray(target, dtype=np.float32))
        self.pid.append(int(player_id))

    def to_numpy(self):
        if not self.obs:
            return (
                np.zeros((0, int(G["obs_dim"])), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        return (
            np.stack(self.obs, axis=0),
            np.stack(self.legal, axis=0),
            np.stack(self.target, axis=0),
            np.asarray(self.pid, dtype=np.int32),
        )


def _ensure_weights_loaded(weights_path: str, weights_version: int) -> None:
    """
    Evita custo de (re)criar redes e evita load_state_dict repetido.
    - Cada worker carrega pesos do disco apenas quando (path,version) mudar.
    """
    if G["last_weights_version"] == int(weights_version) and G["last_weights_path"] == str(weights_path):
        return

    torch = G["torch"]
    device = G["device"]
    adv_nets = G["adv_nets"]

    sds = torch.load(weights_path, map_location="cpu")
    if not isinstance(sds, (list, tuple)) or len(sds) != 3:
        raise RuntimeError(f"weights file inválido: esperado list len=3, obtido {type(sds)} len={getattr(sds,'__len__',None)}")

    for net, sd in zip(adv_nets, sds):
        net.load_state_dict(sd)
        net.to(device)
        net.eval()

    G["last_weights_version"] = int(weights_version)
    G["last_weights_path"] = str(weights_path)


@dataclass
class AdvTask:
    iteration: int
    traverser_id: int
    budget: int
    seed: int
    weights_path: str
    weights_version: int


@dataclass
class PolTask:
    iteration: int
    episodes: int
    seed: int
    weights_path: str
    weights_version: int


def run_adv_task(task: AdvTask):
    torch = G["torch"]
    ExternalSamplingTraverser = G["ExternalSamplingTraverser"]
    EpisodeSpec = G["EpisodeSpec"]
    ScenarioSampler = G["ScenarioSampler"]
    choose_dealer = G["choose_dealer"]
    cpoker = G["cpoker"]
    device = G["device"]

    _ensure_weights_loaded(task.weights_path, task.weights_version)

    rng = np.random.default_rng(int(task.seed))

    adv_buffers = [_ListBuffer(), _ListBuffer(), _ListBuffer()]

    sampler = ScenarioSampler(config=dict(G["scen_cfg"]), seed=int(task.seed) ^ 0xA5A5A5)

    traverser = ExternalSamplingTraverser(
        cpoker_module=cpoker,
        base_game_seed=int(G["base_game_seed"]),
        adv_nets=G["adv_nets"],
        adv_buffers=adv_buffers,
        traverser_id=int(task.traverser_id),
        device=device,
        rng=rng,
        torch=torch,
    )

    for _ in range(int(task.budget)):
        # ep_seed determinístico e único por (iter, traverser_id, task.seed, idx_local)
        # Evita colisões massivas quando budget é grande.
        ep_seed = (
            int(G["base_game_seed"])
            + int(task.iteration) * 10_000_000
            + int(task.traverser_id) * 1_000_000
            + (int(task.seed) % 1_000_000)
            + int(_)
        )
        sc = sampler.sample()
        stacks = list(sc["stacks"])
        sb = int(sc["sb"])
        bb = int(sc["bb"])
        game_is_hu = bool(sc.get("game_is_hu", False))

        stacks2 = list(stacks)
        dealer = int(choose_dealer(rng, stacks2, sb, bb, game_is_hu))
        spec = EpisodeSpec(ep_seed=int(ep_seed), stacks=stacks2, dealer_id=dealer, sb=sb, bb=bb, game_is_hu=game_is_hu)

        traverser.traverse(spec, traverser=int(task.traverser_id), history_actions=[])

    obs, legal, target, pid = adv_buffers[int(task.traverser_id)].to_numpy()
    return int(task.traverser_id), obs, legal, target, pid


def run_pol_task(task: PolTask):
    torch = G["torch"]
    Replayer = G["Replayer"]
    EpisodeSpec = G["EpisodeSpec"]
    ScenarioSampler = G["ScenarioSampler"]
    choose_dealer = G["choose_dealer"]
    build_sigma_from_advnet = G["build_sigma_from_advnet"]
    legal_mask_from_state = G["legal_mask_from_state"]
    sample_action = G["sample_action"]
    cpoker = G["cpoker"]
    device = G["device"]

    _ensure_weights_loaded(task.weights_path, task.weights_version)

    rng = np.random.default_rng(int(task.seed))

    pol_buffers = [_ListBuffer(), _ListBuffer(), _ListBuffer()]
    action_counts = [[0] * 7 for _ in range(3)]

    sampler = ScenarioSampler(config=dict(G["scen_cfg"]), seed=int(task.seed) ^ 0x5A5A5A)
    replayer = Replayer(cpoker, base_game_seed=int(G["base_game_seed"]))

    for _ in range(int(task.episodes)):
        # ep_seed determinístico e único por (iter, task.seed, idx_local)
        # Evita colisões em batches grandes de policy_episodes.
        ep_seed = (
            int(G["base_game_seed"])
            + int(task.iteration) * 10_000_000
            + (int(task.seed) % 1_000_000)
            + int(_)
        )
        sc = sampler.sample()
        stacks = list(sc["stacks"])
        sb = int(sc["sb"])
        bb = int(sc["bb"])
        game_is_hu = bool(sc.get("game_is_hu", False))

        stacks2 = list(stacks)
        dealer = int(choose_dealer(rng, stacks2, sb, bb, game_is_hu))
        spec = EpisodeSpec(ep_seed=int(ep_seed), stacks=stacks2, dealer_id=dealer, sb=sb, bb=bb, game_is_hu=game_is_hu)

        g = replayer.make_game(spec)

        while not g.is_over():
            p = int(g.get_game_pointer())
            state = g.get_state(p)

            pid = int(p)  # current player is game pointer
            if pid < 0 or pid >= 3:
                break

            raw_legal = state.get("raw_legal_actions", [])
            if not raw_legal:
                break

            obs = np.array(state["obs"], dtype=np.float32, copy=True)
            raw_legal = list(raw_legal)

            legal_mask = legal_mask_from_state({"raw_legal_actions": raw_legal})

            sigma = build_sigma_from_advnet(
                G["adv_nets"][pid],
                obs=obs,
                legal_mask_or_raw=raw_legal,
                device=device,
                torch=torch,
            )

            a = sample_action(rng, sigma, raw_legal_actions=raw_legal, greedy=False)

            pol_buffers[pid].add(obs, legal_mask, sigma, pid)
            if 0 <= int(a) <= 6:
                action_counts[pid][int(a)] += 1

            g.step(int(a))

    out = []
    for p in range(3):
        out.append(pol_buffers[p].to_numpy())
    return out, action_counts