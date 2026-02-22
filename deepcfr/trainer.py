from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .buffers import ReservoirBuffer
from .networks import AdvantageNet, PolicyNet
from .traversal import (
    ExternalSamplingTraverser,
    Replayer,
    EpisodeSpec,
    build_sigma_from_advnet,
    legal_mask_from_state,
    sample_action,
)

def _atomic_torch_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

@dataclass
class IterMetrics:
    iteration: int
    dt_total_s: float
    dt_collect_adv_s: float
    dt_train_adv_s: float
    dt_collect_pol_s: float
    dt_train_pol_s: float
    adv_losses: list[float]
    pol_loss: float
    adv_added: list[int]
    pol_added: list[int]
    action_counts: list[list[int]]
    adv_buf_size: list[int]
    pol_buf_size: list[int]

class DeepCFRTrainer:
    """
    DeepCFR Trainer compatível com:
      - deepcfr/buffers.py (ReservoirBuffer.add / sample_batch)
      - deepcfr/scenario.py (ScenarioSampler.sample() -> dict)
      - deepcfr/traversal.py (ExternalSamplingTraverser + helpers)

    Importante:
      - Nenhuma chamada a C++ (cpoker) acontece no __init__.
      - Checkpoint salva RNGs (numpy/python/torch) para continuidade.
    """

    def __init__(
        self,
        cpoker_module: Any,
        scenario_sampler: Any,
        choose_dealer_fn: Any,
        adv_nets: list[torch.nn.Module],
        pol_nets: list[torch.nn.Module],
        adv_opts: list[torch.optim.Optimizer],
        pol_opts: list[torch.optim.Optimizer],
        adv_buffers: list[ReservoirBuffer],
        pol_buffers: list[ReservoirBuffer],
        device: torch.device,
        seed: int = 123,
        base_game_seed: int = 123_000,
        traversals_per_player: int = 512,
        policy_episodes: int = 256,
        adv_batch_size: int = 4096,
        pol_batch_size: int = 4096,
        adv_train_steps: int = 1,
        pol_train_steps: int = 1,
    ):
        self.cpoker = cpoker_module
        self.scenario_sampler = scenario_sampler
        self.choose_dealer_fn = choose_dealer_fn

        self.adv_nets = adv_nets
        self.pol_nets = pol_nets
        self.adv_opts = adv_opts
        self.pol_opts = pol_opts
        self.adv_buffers = adv_buffers
        self.pol_buffers = pol_buffers
        self.device = device

        self.num_players = len(adv_nets)
        if self.num_players != 3:
            raise ValueError(
                f"Este projeto assume 3 players no cpoker (HU = 2 ativos + 1 morto). Recebi {self.num_players}."
            )

        self.iteration = 0

        self.base_game_seed = int(base_game_seed)

        self.traversals_per_player = int(traversals_per_player)
        self.policy_episodes = int(policy_episodes)
        self.adv_batch_size = int(adv_batch_size)
        self.pol_batch_size = int(pol_batch_size)
        self.adv_train_steps = int(adv_train_steps)
        self.pol_train_steps = int(pol_train_steps)

        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        self.last_metrics: IterMetrics | None = None

    def _set_eval(self) -> None:
        for n in self.adv_nets:
            n.eval()
        for n in self.pol_nets:
            n.eval()

    def _set_train(self) -> None:
        for n in self.adv_nets:
            n.train()
        for n in self.pol_nets:
            n.train()

    def _sample_episode_spec(self, ep_seed: int) -> EpisodeSpec:
        # ScenarioSampler.sample() retorna dict com stacks/sb/bb/game_is_hu/dead_players...
        sc = self.scenario_sampler.sample()
        stacks = list(sc["stacks"])
        sb = int(sc["sb"])
        bb = int(sc["bb"])
        game_is_hu = bool(sc.get("game_is_hu", False))

        # dealer deve ser escolhido considerando seats vivos (stacks > 0)
        stacks_for_roles = list(stacks)
        dealer = int(self.choose_dealer_fn(self.rng, stacks_for_roles, sb, bb, game_is_hu))
        stacks = stacks_for_roles

        return EpisodeSpec(
            ep_seed=int(ep_seed),
            stacks=stacks,
            dealer_id=dealer,
            sb=sb,
            bb=bb,
            game_is_hu=game_is_hu
        )


    def collect_advantage(self, traverser_id: int, traversals_per_player: int | None = None) -> int:
        traverser_id = int(traverser_id)
        budget = self.traversals_per_player if traversals_per_player is None else int(traversals_per_player)

        before = int(self.adv_buffers[traverser_id].size)
        self._set_eval()

        traverser = ExternalSamplingTraverser(
            cpoker_module=self.cpoker,
            base_game_seed=self.base_game_seed,
            adv_nets=self.adv_nets,
            adv_buffers=self.adv_buffers,
            traverser_id=traverser_id,
            device=self.device,
            rng=self.rng,
            torch=torch,
        )

        for _ in range(budget):
            # ep_seed determinístico e único por (iter, traverser, idx)
            ep_seed = (
                int(self.base_game_seed)
                + int(self.iteration) * 10_000_000
                + int(traverser_id) * 1_000_000
                + int(_)
            )
            spec = self._sample_episode_spec(ep_seed)
            traverser.traverse(spec, traverser=traverser_id, history_actions=[])

        return int(self.adv_buffers[traverser_id].size) - before

    def train_adv_nets(self, steps: int | None = None, batch_size: int | None = None) -> list[float]:
        self._set_train()
        steps = self.adv_train_steps if steps is None else int(steps)
        batch_size = self.adv_batch_size if batch_size is None else int(batch_size)

        losses: list[float] = []
        for p in range(self.num_players):
            buf = self.adv_buffers[p]
            if buf.size < max(1024, batch_size):
                losses.append(float("nan"))
                continue

            net: AdvantageNet = self.adv_nets[p]  # type: ignore
            opt = self.adv_opts[p]

            running = 0.0
            for _ in range(max(1, steps)):
                obs, legal, target, _pid = buf.sample_batch(batch_size)
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                legal_t = torch.as_tensor(legal, device=self.device, dtype=torch.float32)
                target_t = torch.as_tensor(target, device=self.device, dtype=torch.float32)

                pred = net(obs_t)
                loss = AdvantageNet.masked_mse(pred, target_t, legal_t)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

                running += float(loss.detach().cpu().item())

            losses.append(running / float(max(1, steps)))
        return losses

    def collect_policy(self, episodes: int | None = None) -> tuple[list[int], list[list[int]]]:
        self._set_eval()
        episodes = self.policy_episodes if episodes is None else int(episodes)

        added = [0] * self.num_players
        action_counts = [[0] * 7 for _ in range(self.num_players)]

        replayer = Replayer(self.cpoker, base_game_seed=self.base_game_seed)

        for _ in range(episodes):
            # ep_seed determinístico e único por (iter, idx)
            ep_seed = (
                int(self.base_game_seed)
                + int(self.iteration) * 10_000_000
                + int(_)
            )
            spec = self._sample_episode_spec(ep_seed)
            g = replayer.make_game(spec)

            while not g.is_over():
                p = int(g.get_game_pointer())
                state = g.get_state(p)

                # Em alguns builds do env, "current_player" pode não existir.
                # O game_pointer já representa o jogador atual.
                pid = int(state.get("current_player", p))
                if pid < 0 or pid >= self.num_players:
                    break

                raw_legal = state.get("raw_legal_actions", [])
                if not raw_legal:
                    break

                obs = np.array(state["obs"], dtype=np.float32, copy=True)
                raw_legal = list(raw_legal)

                legal_mask = legal_mask_from_state({"raw_legal_actions": raw_legal})

                sigma = build_sigma_from_advnet(
                    self.adv_nets[pid],
                    obs=obs,
                    legal_mask_or_raw=raw_legal,
                    device=self.device,
                    torch=torch,
                )

                a = sample_action(self.rng, sigma, raw_legal_actions=raw_legal, greedy=False)

                self.pol_buffers[pid].add(obs=obs, legal_mask=legal_mask, target=sigma, player_id=pid)
                added[pid] += 1
                if 0 <= int(a) <= 6:
                    action_counts[pid][int(a)] += 1

                g.step(int(a))

        return added, action_counts

    def train_pol_nets(self, steps: int | None = None, batch_size: int | None = None) -> float:
        self._set_train()
        steps = self.pol_train_steps if steps is None else int(steps)
        batch_size = self.pol_batch_size if batch_size is None else int(batch_size)

        losses = []
        for p in range(self.num_players):
            buf = self.pol_buffers[p]
            if buf.size < max(1024, batch_size):
                continue

            net: PolicyNet = self.pol_nets[p]  # type: ignore
            opt = self.pol_opts[p]

            running = 0.0
            for _ in range(max(1, steps)):
                obs, legal, target, _pid = buf.sample_batch(batch_size)
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                legal_t = torch.as_tensor(legal, device=self.device, dtype=torch.float32)
                target_t = torch.as_tensor(target, device=self.device, dtype=torch.float32)

                logits = net(obs_t)
                loss = PolicyNet.masked_cross_entropy(logits, target_t, legal_t)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

                running += float(loss.detach().cpu().item())

            losses.append(running / float(max(1, steps)))

        if not losses:
            return float("nan")
        return float(sum(losses) / len(losses))

    def step_iteration(
        self,
        *,
        traversals_per_player: int | None = None,
        policy_episodes: int | None = None,
        adv_steps: int | None = None,
        pol_steps: int | None = None,
        adv_batch_size: int | None = None,
        pol_batch_size: int | None = None,
    ) -> IterMetrics:
        if traversals_per_player is not None:
            self.traversals_per_player = int(traversals_per_player)
        if policy_episodes is not None:
            self.policy_episodes = int(policy_episodes)
        if adv_steps is not None:
            self.adv_train_steps = int(adv_steps)
        if pol_steps is not None:
            self.pol_train_steps = int(pol_steps)
        if adv_batch_size is not None:
            self.adv_batch_size = int(adv_batch_size)
        if pol_batch_size is not None:
            self.pol_batch_size = int(pol_batch_size)

        t0 = time.time()

        tA0 = time.time()
        adv_added = [self.collect_advantage(p) for p in range(self.num_players)]
        dt_collect_adv = time.time() - tA0

        tB0 = time.time()
        adv_losses = self.train_adv_nets()
        dt_train_adv = time.time() - tB0

        tC0 = time.time()
        pol_added, action_counts = self.collect_policy()
        dt_collect_pol = time.time() - tC0

        tD0 = time.time()
        pol_loss = self.train_pol_nets()
        dt_train_pol = time.time() - tD0

        self.iteration += 1

        m = IterMetrics(
            iteration=self.iteration,
            dt_total_s=time.time() - t0,
            dt_collect_adv_s=float(dt_collect_adv),
            dt_train_adv_s=float(dt_train_adv),
            dt_collect_pol_s=float(dt_collect_pol),
            dt_train_pol_s=float(dt_train_pol),
            adv_losses=[float(x) for x in adv_losses],
            pol_loss=float(pol_loss),
            adv_added=[int(x) for x in adv_added],
            pol_added=[int(x) for x in pol_added],
            action_counts=action_counts,
            adv_buf_size=[int(b.size) for b in self.adv_buffers],
            pol_buf_size=[int(b.size) for b in self.pol_buffers],
        )
        self.last_metrics = m
        return m

    def checkpoint(self, path: str) -> None:
        payload = {
            "iteration": int(self.iteration),
            "base_game_seed": int(self.base_game_seed),
            "rng_np": self.rng.bit_generator.state,
            "rng_py": random.getstate(),
            "rng_torch": torch.get_rng_state(),
            # Scenario RNG (robust): supports numpy Generator (preferred) and python random.Random (fallback)

            "scenario_rng_kind": (

                "numpy" if (

                    getattr(self.scenario_sampler, "rng", None) is not None

                    and getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None) is not None

                    and getattr(getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None), "state", None) is not None

                ) else (

                    "python_random" if (

                        getattr(self.scenario_sampler, "rng", None) is not None

                        and callable(getattr(getattr(self.scenario_sampler, "rng", None), "getstate", None))

                    ) else None

                )

            ),

            "scenario_rng_state": (

                getattr(getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None), "state", None)

                if (

                    getattr(self.scenario_sampler, "rng", None) is not None

                    and getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None) is not None

                    and getattr(getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None), "state", None) is not None

                ) else (

                    getattr(self.scenario_sampler.rng, "getstate")()

                    if (

                        getattr(self.scenario_sampler, "rng", None) is not None

                        and callable(getattr(getattr(self.scenario_sampler, "rng", None), "getstate", None))

                    ) else None

                )

            ),

            # Legacy key: keep numpy state here for backward compatibility with older restores

            "scenario_rng": (

                getattr(getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None), "state", None)

                if (

                    getattr(self.scenario_sampler, "rng", None) is not None

                    and getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None) is not None

                    and getattr(getattr(getattr(self.scenario_sampler, "rng", None), "bit_generator", None), "state", None) is not None

                ) else None

            ),
"adv_nets": [n.state_dict() for n in self.adv_nets],
            "pol_nets": [n.state_dict() for n in self.pol_nets],
            "adv_opts": [o.state_dict() for o in self.adv_opts],
            "pol_opts": [o.state_dict() for o in self.pol_opts],
        }
        _atomic_torch_save(payload, path)

    def restore(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.iteration = int(ckpt.get("iteration", 0))
        self.base_game_seed = int(ckpt.get("base_game_seed", self.base_game_seed))

        if ckpt.get("rng_np") is not None:
            self.rng.bit_generator.state = ckpt["rng_np"]
        if ckpt.get("rng_py") is not None:
            random.setstate(ckpt["rng_py"])
        if ckpt.get("rng_torch") is not None:
            torch.set_rng_state(ckpt["rng_torch"])

        # Scenario RNG restore (robust + backward compatible)


        try:


            sr_kind = ckpt.get("scenario_rng_kind", None)


            sr_state = ckpt.get("scenario_rng_state", None)


        


            # Legacy checkpoints may only have "scenario_rng" (numpy bit_generator.state dict)


            if sr_kind is None and sr_state is None and ckpt.get("scenario_rng", None) is not None:


                sr_kind = "numpy"


                sr_state = ckpt.get("scenario_rng")


        


            if sr_kind == "numpy" and sr_state is not None and getattr(self.scenario_sampler, "rng", None) is not None:


                bg = getattr(self.scenario_sampler.rng, "bit_generator", None)


                if bg is not None:


                    bg.state = sr_state


            elif sr_kind == "python_random" and sr_state is not None and getattr(self.scenario_sampler, "rng", None) is not None:


                if callable(getattr(self.scenario_sampler.rng, "setstate", None)):


                    self.scenario_sampler.rng.setstate(sr_state)


        except Exception:


            pass

        for n, sd in zip(self.adv_nets, ckpt.get("adv_nets", [])):
            n.load_state_dict(sd)
        for n, sd in zip(self.pol_nets, ckpt.get("pol_nets", [])):
            n.load_state_dict(sd)

        for o, sd in zip(self.adv_opts, ckpt.get("adv_opts", [])):
            o.load_state_dict(sd)
        for o, sd in zip(self.pol_opts, ckpt.get("pol_opts", [])):
            o.load_state_dict(sd)
