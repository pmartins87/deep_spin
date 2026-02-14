from __future__ import annotations

"""
scripts/train_deepcfr.py (PARALLEL + DETERMINISTIC MERGE + WEIGHTS DISK-CACHE)

Mudanças principais:
1) Merge determinístico (resume confiável) mantendo o paralelismo nos rollouts.
2) Remove gargalo de IPC: não envia state_dict em cada task.
   - Salva pesos 1 vez por fase/iteração em checkpoints/weights_*.pt (atômico).
   - Workers carregam do disco apenas quando (path,version) muda.
3) Permite separar:
   - --deterministic_merge (default 1): merge ordenado (recomendado manter sempre).
   - --bitwise (default 1): força treino single-thread e algoritmos determinísticos (mais lento).
     Se quiser velocidade, use --bitwise 0 e aumente --main_threads.

Requisitos:
- deepcfr/rollout_workers.py (versão otimizada que usa weights_path + weights_version)
- deepcfr/traversal.py com clone fast path (sem replay O(N^2))
"""

import os
import sys
import time
import json
import shutil
import signal
import traceback
import faulthandler
import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_DIR = PROJECT_ROOT / "env"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))


def _load_config_yaml_or_json(path: Path) -> dict:
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


def ensure_dirs() -> None:
    (PROJECT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def rotate_incompatible_path(path: Path, reason: str) -> None:
    if not path.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}_incompatible_{reason}_{ts}{path.suffix}")
    shutil.move(str(path), str(dst))
    print(f"[WARN] Arquivo incompatível ({reason}), movido para: {dst}")


def rotate_incompatible_folder(folder: Path, reason: str) -> None:
    if not folder.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = folder.with_name(f"{folder.name}_incompatible_{reason}_{ts}")
    shutil.move(str(folder), str(dst))
    print(f"[WARN] Pasta incompatível ({reason}), movida para: {dst}")


def detect_obs_dim(cpoker_module) -> int:
    import numpy as np
    g = cpoker_module.PokerGame(3, 0)
    if hasattr(g, "set_seed"):
        g.set_seed(123)
    g.reset([500, 500, 500], 0, 10, 20)
    p = int(g.get_game_pointer())
    s = g.get_state(p)
    obs = np.asarray(s["obs"], dtype=np.float32)
    return int(obs.shape[0])


def _set_threads_env(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def _torch_save_atomic(torch, obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@dataclass
class TrainCfg:
    device: str = "cpu"
    iterations: int = 10_000_000

    traversals_per_player: int = 16384
    policy_episodes: int = 8192

    adv_batch_size: int = 16384
    pol_batch_size: int = 16384
    adv_steps: int = 2
    pol_steps: int = 1

    adv_capacity: int = 3_000_000
    pol_capacity: int = 2_000_000

    adv_lr: float = 2e-4
    pol_lr: float = 2e-4
    weight_decay: float = 1e-6

    base_seed: int = 123
    log_every: int = 5
    save_every: int = 25

    rollout_workers: int = 24
    worker_threads: int = 1
    main_torch_threads: int = 12

    deterministic_merge: bool = True
    bitwise: bool = True

    ckpt_dir: Path = (PROJECT_ROOT / "checkpoints")

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_dir / "checkpoint.pt"

    @property
    def buffers_dir(self) -> Path:
        return self.ckpt_dir / "buffers"

    @property
    def weights_adv_path(self) -> Path:
        return self.ckpt_dir / "weights_adv.pt"

    @property
    def weights_pol_path(self) -> Path:
        return self.ckpt_dir / "weights_pol.pt"


def save_all(trainer, cfg: TrainCfg) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.buffers_dir.mkdir(parents=True, exist_ok=True)

    trainer.checkpoint(str(cfg.ckpt_path))

    for p in range(3):
        trainer.adv_buffers[p].save(str(cfg.buffers_dir / f"adv_p{p}"))
        trainer.pol_buffers[p].save(str(cfg.buffers_dir / f"pol_p{p}"))


def try_restore_all(trainer, cfg: TrainCfg) -> None:
    if cfg.ckpt_path.exists():
        try:
            trainer.restore(str(cfg.ckpt_path))
            print(f"[OK] Restored {cfg.ckpt_path}, iteration={trainer.iteration}")
        except Exception as e:
            rotate_incompatible_path(cfg.ckpt_path, "restore_failed")
            print(f"[WARN] Restore do checkpoint falhou: {type(e).__name__}: {e}, iniciando do zero.")

    if cfg.buffers_dir.exists():
        try:
            from deepcfr.buffers import ReservoirBuffer
            new_adv = []
            new_pol = []
            for p in range(3):
                adv_p = ReservoirBuffer.load(str(cfg.buffers_dir / f"adv_p{p}"), seed_fallback=cfg.base_seed + 100 + p)
                pol_p = ReservoirBuffer.load(str(cfg.buffers_dir / f"pol_p{p}"), seed_fallback=cfg.base_seed + 200 + p)
                new_adv.append(adv_p)
                new_pol.append(pol_p)

            trainer.adv_buffers[:] = new_adv
            trainer.pol_buffers[:] = new_pol
            print("[OK] Buffers restaurados.")
        except Exception as e:
            rotate_incompatible_folder(cfg.buffers_dir, "buffers_restore_failed")
            print(f"[WARN] Restore dos buffers falhou: {type(e).__name__}: {e}, buffers resetados.")


def _buffer_add_many(buf, obs, legal, target, pid) -> int:
    n = int(obs.shape[0])
    if n == 0:
        return 0
    add_batch = getattr(buf, "add_batch", None)
    if callable(add_batch):
        add_batch(obs, legal, target, pid)
        return n
    for i in range(n):
        buf.add(obs[i], legal[i], target[i], int(pid[i]))
    return n
    
def _buffer_size(buf) -> int:
    """Tenta descobrir quantos itens existem no buffer."""
    try:
        return int(len(buf))
    except Exception:
        pass

    # tenta atributos comuns
    for name in ("size", "n", "count", "num_items", "filled"):
        try:
            v = getattr(buf, name)
            v = v() if callable(v) else v
            if v is not None:
                return int(v)
        except Exception:
            pass
    return 0


def _safe_batch_size(requested: int, available: int, min_bs: int = 1024) -> int:
    """Garante batch <= available e >= min_bs."""
    if available <= 0:
        return 0
    return max(min_bs, min(int(requested), int(available)))


def _fmt_action_dist(counts):
    total = sum(int(x) for x in counts)
    if total <= 0:
        return "no_actions"
    pct = [100.0 * int(x) / total for x in counts]
    return (
        f"F={pct[0]:.1f}% C={pct[1]:.1f}% "
        f"33={pct[2]:.1f}% 50={pct[3]:.1f}% 75={pct[4]:.1f}% "
        f"100={pct[5]:.1f}% AI={pct[6]:.1f}%"
    )


def main() -> None:
    ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--traversals", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--worker_threads", type=int, default=None)
    ap.add_argument("--main_threads", type=int, default=None)
    ap.add_argument("--deterministic_merge", type=int, default=1)
    ap.add_argument("--bitwise", type=int, default=1)
    args = ap.parse_args()

    cfg = TrainCfg()
    if args.workers is not None:
        cfg.rollout_workers = int(args.workers)
    if args.traversals is not None:
        cfg.traversals_per_player = int(args.traversals)
    if args.episodes is not None:
        cfg.policy_episodes = int(args.episodes)
    if args.worker_threads is not None:
        cfg.worker_threads = int(args.worker_threads)
    if args.main_threads is not None:
        cfg.main_torch_threads = int(args.main_threads)
    cfg.deterministic_merge = bool(int(args.deterministic_merge))
    cfg.bitwise = bool(int(args.bitwise))

    # bitwise => single-thread no treino
    if cfg.bitwise:
        cfg.main_torch_threads = 1

    fault_path = PROJECT_ROOT / "logs" / "fault.log"
    with open(fault_path, "w", encoding="utf-8") as ff:
        faulthandler.enable(file=ff, all_threads=True)

        import cpoker
        obs_dim = detect_obs_dim(cpoker)
        os.environ["SPIN_OBS_DIM"] = str(obs_dim)

        # print CLONE status UMA vez, no processo principal
        try:
            g0 = cpoker.PokerGame(3, 0)
            if hasattr(g0, "clone"):
                print("[INFO] CLONE ENABLED (PokerGame.clone detected) [main]")
            else:
                print("[WARN] CLONE DISABLED (fallback replay) [main]")
        except Exception:
            pass
        os.environ["SPIN_WORKER"] = "1"  # suprime logs de clone no processo principal e nos workers


        _set_threads_env(cfg.main_torch_threads)
        import torch

        torch.set_num_threads(max(1, int(cfg.main_torch_threads)))
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        if cfg.bitwise:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        from deepcfr.buffers import ReservoirBuffer, ReservoirConfig
        from deepcfr.networks import AdvantageNet, PolicyNet
        from deepcfr.scenario import ScenarioSampler, choose_dealer_id_for_episode
        # Audit: garante que o arquivo scenario.py importado é o correto (evita versão simplificada/errada)
        try:
            import deepcfr.scenario as scenario_mod
            import hashlib
            _scenario_path = Path(scenario_mod.__file__).resolve()
            _scenario_sha = hashlib.sha256(_scenario_path.read_bytes()).hexdigest()
            print(f"[INFO] deepcfr.scenario file: {_scenario_path}")
            print(f"[INFO] deepcfr.scenario sha256: {_scenario_sha}")
        except Exception as _e:
            print(f"[WARN] Não consegui calcular hash do scenario.py: {_e}")

        from deepcfr.trainer import DeepCFRTrainer
        from deepcfr.rollout_workers import init_worker, run_adv_task, run_pol_task, AdvTask, PolTask

        print(f"OBS_DIM detectado: {obs_dim}")
        print(f"Python: {sys.executable}")
        print(f"cpoker: {getattr(cpoker, '__file__', 'n/a')}")
        print(
            f"workers={cfg.rollout_workers}, worker_threads={cfg.worker_threads}, "
            f"main_threads={cfg.main_torch_threads}, deterministic_merge={int(cfg.deterministic_merge)}, bitwise={int(cfg.bitwise)}"
        )

        scen_cfg = _load_config_yaml_or_json(PROJECT_ROOT / "config.yaml")
        sampler = ScenarioSampler(config=scen_cfg, seed=cfg.base_seed)

        device = torch.device(cfg.device)

        adv_nets = [AdvantageNet(obs_dim=obs_dim).to(device) for _ in range(3)]
        pol_nets = [PolicyNet(obs_dim=obs_dim).to(device) for _ in range(3)]

        adv_opts = [torch.optim.AdamW(n.parameters(), lr=cfg.adv_lr, weight_decay=cfg.weight_decay) for n in adv_nets]
        pol_opts = [torch.optim.AdamW(n.parameters(), lr=cfg.pol_lr, weight_decay=cfg.weight_decay) for n in pol_nets]

        adv_buffers = [
            ReservoirBuffer(ReservoirConfig(capacity=cfg.adv_capacity, obs_dim=obs_dim), seed=cfg.base_seed + 100 + p)
            for p in range(3)
        ]
        pol_buffers = [
            ReservoirBuffer(ReservoirConfig(capacity=cfg.pol_capacity, obs_dim=obs_dim), seed=cfg.base_seed + 200 + p)
            for p in range(3)
        ]

        trainer = DeepCFRTrainer(
            cpoker_module=cpoker,
            scenario_sampler=sampler,
            choose_dealer_fn=choose_dealer_id_for_episode,
            adv_nets=adv_nets,
            pol_nets=pol_nets,
            adv_opts=adv_opts,
            pol_opts=pol_opts,
            adv_buffers=adv_buffers,
            pol_buffers=pol_buffers,
            device=device,
            seed=cfg.base_seed,
            traversals_per_player=cfg.traversals_per_player,
            policy_episodes=cfg.policy_episodes,
            adv_batch_size=cfg.adv_batch_size,
            pol_batch_size=cfg.pol_batch_size,
            adv_train_steps=cfg.adv_steps,
            pol_train_steps=cfg.pol_steps,
        )

        try_restore_all(trainer, cfg)

        stop_flag = {"stop": False}

        def _sigint(_sig, _frm):
            stop_flag["stop"] = True

        try:
            signal.signal(signal.SIGINT, _sigint)
        except Exception:
            pass

        pool = ProcessPoolExecutor(
            max_workers=int(cfg.rollout_workers),
            initializer=init_worker,
            initargs=(
                str(PROJECT_ROOT),
                str(ENV_DIR),
                int(obs_dim),
                int(trainer.base_game_seed),
                dict(scen_cfg),
                int(cfg.worker_threads),
                int(cfg.base_seed ^ 0x123456),
            ),
        )

        print("Training started. CTRL+C salva e sai limpo.")
        t0 = time.time()
        last_log = t0

        try:
            while trainer.iteration < cfg.iterations and not stop_flag["stop"]:
                iter0 = time.time()
                it = int(trainer.iteration)

                # --- escreve pesos (ADV snapshot) 1x por iteração (atômico)
                adv_sds = [n.state_dict() for n in trainer.adv_nets]
                _torch_save_atomic(torch, adv_sds, cfg.weights_adv_path)
                adv_ver = it * 2 + 0

                w = int(cfg.rollout_workers)

                # 1) ADV rollouts
                tA0 = time.time()
                per = max(1, int(cfg.traversals_per_player) // w)
                rem = int(cfg.traversals_per_player) - per * w

                adv_futures = {}
                adv_keys = []
                for p in range(3):
                    for k in range(w):
                        bud = per + (1 if k < rem else 0)
                        seed = int(cfg.base_seed) + 10_000_000 + it * 1000 + p * 100 + k
                        key = (p, k)
                        adv_keys.append(key)
                        fut = pool.submit(
                            run_adv_task,
                            AdvTask(it, p, bud, seed, str(cfg.weights_adv_path), adv_ver),
                        )
                        adv_futures[fut] = key

                adv_results = {}
                try:
                    for fut in as_completed(list(adv_futures.keys())):
                        key = adv_futures[fut]
                        adv_results[key] = fut.result()
                except KeyboardInterrupt:
                    stop_flag["stop"] = True
                    print("\n[WARN] Ctrl+C durante ADV collect. Cancelando futures e salvando estado consistente...")
                    for f in adv_futures.keys():
                        f.cancel()
                    # IMPORTANT: não faz merge parcial
                    raise
                adv_added = [0, 0, 0]
                if cfg.deterministic_merge:
                    adv_iter_keys = sorted(adv_keys)
                else:
                    adv_iter_keys = adv_keys

                for key in adv_iter_keys:
                    pid, obs, legal, target, pp = adv_results[key]
                    adv_added[pid] += _buffer_add_many(trainer.adv_buffers[pid], obs, legal, target, pp)

                dt_collect_adv = time.time() - tA0

                # 2) train ADV
                tB0 = time.time()
                adv_avail = min(_buffer_size(trainer.adv_buffers[p]) for p in range(3))
                adv_bs = _safe_batch_size(cfg.adv_batch_size, adv_avail, min_bs=1024)

                if adv_bs == 0:
                    adv_losses = [float("nan"), float("nan"), float("nan")]
                else:
                    adv_losses = trainer.train_adv_nets(steps=cfg.adv_steps, batch_size=adv_bs)

                dt_train_adv = time.time() - tB0

                # --- escreve pesos pós-treino (POL snapshot)
                adv_sds2 = [n.state_dict() for n in trainer.adv_nets]
                _torch_save_atomic(torch, adv_sds2, cfg.weights_pol_path)
                pol_ver = it * 2 + 1

                # 3) POLICY rollouts
                tC0 = time.time()
                per_ep = max(1, int(cfg.policy_episodes) // w)
                rem_ep = int(cfg.policy_episodes) - per_ep * w

                pol_futures = {}
                pol_keys = []
                for k in range(w):
                    ep = per_ep + (1 if k < rem_ep else 0)
                    seed = int(cfg.base_seed) + 20_000_000 + it * 1000 + k
                    key = (k,)
                    pol_keys.append(key)
                    fut = pool.submit(
                        run_pol_task,
                        PolTask(it, ep, seed, str(cfg.weights_pol_path), pol_ver),
                    )
                    pol_futures[fut] = key

                pol_results = {}
                try:
                    for fut in as_completed(list(pol_futures.keys())):
                        key = pol_futures[fut]
                        pol_results[key] = fut.result()
                except KeyboardInterrupt:
                    stop_flag["stop"] = True
                    print("\n[WARN] Ctrl+C durante POL collect. Cancelando futures e salvando estado consistente...")
                    for f in pol_futures.keys():
                        f.cancel()
                    # IMPORTANT: não faz merge parcial
                    raise
                pol_added = [0, 0, 0]
                action_counts = [[0] * 7 for _ in range(3)]

                pol_iter_keys = sorted(pol_keys) if cfg.deterministic_merge else pol_keys

                for key in pol_iter_keys:
                    out, counts = pol_results[key]
                    for p in range(3):
                        obs, legal, target, pid = out[p]
                        pol_added[p] += _buffer_add_many(trainer.pol_buffers[p], obs, legal, target, pid)
                    for p in range(3):
                        for a in range(7):
                            action_counts[p][a] += int(counts[p][a])

                dt_collect_pol = time.time() - tC0

                # 4) train POLICY (batch dinâmico, evita NaN)
                tD0 = time.time()

                pol_avail = min(_buffer_size(trainer.pol_buffers[p]) for p in range(3))
                pol_bs = _safe_batch_size(cfg.pol_batch_size, pol_avail, min_bs=1024)

                if pol_bs == 0:
                    pol_loss = float("nan")
                    dt_train_pol = 0.0
                else:
                    pol_loss = trainer.train_pol_nets(steps=cfg.pol_steps, batch_size=pol_bs)
                    dt_train_pol = time.time() - tD0


                trainer.iteration += 1
                dt_total = time.time() - iter0

                adv_samples = sum(int(x) for x in adv_added)
                pol_samples = sum(int(x) for x in pol_added)
                adv_sps = adv_samples / max(1e-9, dt_collect_adv)
                pol_sps = pol_samples / max(1e-9, dt_collect_pol)

                if trainer.iteration == 1 or (trainer.iteration % cfg.log_every == 0):
                    now = time.time()
                    elapsed = now - t0
                    print(
                        f"iter={trainer.iteration} dt_iter={dt_total:.1f}s elapsed={elapsed/60:.1f}m "
                        f"ADV: collect={dt_collect_adv:.1f}s ({adv_sps:.0f} sps) train={dt_train_adv:.1f}s bs={adv_bs} "
                        f"POL: collect={dt_collect_pol:.1f}s ({pol_sps:.0f} sps) train={dt_train_pol:.1f}s bs={pol_bs} "
                        f"adv_added={adv_added} pol_added={pol_added} "
                        f"adv_loss={[round(float(x), 4) for x in adv_losses]} pol_loss={round(float(pol_loss), 4)}"
                    )
                    print(
                        "actions p0: " + _fmt_action_dist(action_counts[0]) + " | "
                        "p1: " + _fmt_action_dist(action_counts[1]) + " | "
                        "p2: " + _fmt_action_dist(action_counts[2])
                    )
                    last_log = now

                if trainer.iteration % cfg.save_every == 0:
                    print("[INFO] Salvando checkpoint + buffers...")
                    save_all(trainer, cfg)
                    print("[OK] Salvo.")

        except KeyboardInterrupt:
            stop_flag["stop"] = True
            print("\n[INFO] Interrompido pelo usuário (CTRL+C). Salvando estado consistente...")
        finally:
            print("\nParando, salvando checkpoint + buffers...")
            try:
                save_all(trainer, cfg)
                print("[OK] Salvo. Saindo.")
            except Exception as _e:
                print(f"[WARN] Falha ao salvar checkpoint/buffers: {_e}")
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário (CTRL+C).")
    except Exception as e:
        print("\n[ERRO] O script terminou com exceção:")
        print(f"{type(e).__name__}: {e}")
        print("\nTraceback completo:\n")
        traceback.print_exc()
        if os.name == "nt":
            try:
                os.system("pause")
            except Exception:
                pass
        raise