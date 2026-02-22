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


def ensure_dirs(base_dir: Path | None = None) -> None:
    base = base_dir if base_dir is not None else PROJECT_ROOT
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


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
    
def _fixed_base_game_seed(cfg: TrainCfg) -> int:
    # Seed estável entre execuções e resumes.
    # (Qualquer fórmula determinística serve, desde que não dependa de PID/tempo.)
    return int(cfg.base_seed) ^ 0x9E3779B1


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
    
    # B034: Warm-up, não grava buffers nas primeiras iterações
    warmup_iters: int = 10
    skip_saves_during_warmup: bool = True   

    # B019: Salvar por tempo, reduz overhead de IO.
    # - save_every: mantém compatibilidade (0 desativa)
    # - save_every_seconds: intervalo por tempo (8h default)
    save_every: int = 0
    save_every_seconds: int = 8 * 60 * 60

    rollout_workers: int = 24
    worker_threads: int = 1
    main_torch_threads: int = 12

    deterministic_merge: bool = True
    bitwise: bool = True

    ckpt_dir: Path = (PROJECT_ROOT / "checkpoints")  # pode ser sobrescrito no main() via RUN_DIR

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

    @property
    def rng_path(self) -> Path:
        return self.ckpt_dir / "rng_state.json"

def save_all(trainer, cfg: TrainCfg) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.buffers_dir.mkdir(parents=True, exist_ok=True)

    trainer.checkpoint(str(cfg.ckpt_path))

    for p in range(3):
        trainer.adv_buffers[p].save(str(cfg.buffers_dir / f"adv_p{p}"))
        trainer.pol_buffers[p].save(str(cfg.buffers_dir / f"pol_p{p}"))

    # --- SALVA RNG STATES (determinismo no resume) ---
    import numpy as np
    import torch
    import random

    # --- SALVA RNG STATES sem pickle (compatível com torch 2.6+) ---
    import base64

    torch_bytes = torch.get_rng_state().cpu().numpy().tobytes()
    rng_state = {
        "torch_rng_state_b64": base64.b64encode(torch_bytes).decode("ascii"),
        "numpy_global_state": list(np.random.get_state()),  # (str, ndarray, int, int, float)
        "python_random_state": list(random.getstate()),
    }

    # numpy_global_state[1] é ndarray, precisa virar lista
    if isinstance(rng_state["numpy_global_state"][1], np.ndarray):
        rng_state["numpy_global_state"][1] = rng_state["numpy_global_state"][1].tolist()

    # salva atômico
    cfg.rng_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.rng_path.with_suffix(cfg.rng_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rng_state, f)
    os.replace(tmp, cfg.rng_path)


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

    # --- RESTAURA RNG STATES sem torch.load (compatível com torch 2.6+) ---
    if cfg.rng_path.exists():
        try:
            import numpy as np
            import torch
            import random
            import base64

            with open(cfg.rng_path, "r", encoding="utf-8") as f:
                st = json.load(f)

            # torch rng
            b = base64.b64decode(st["torch_rng_state_b64"].encode("ascii"))
            torch_state = np.frombuffer(b, dtype=np.uint8).copy()
            torch.set_rng_state(torch.from_numpy(torch_state))

            # numpy global
            ng = st["numpy_global_state"]
            # reconstrói tuple: (str, ndarray, int, int, float)
            ng0 = ng[0]
            ng1 = np.asarray(ng[1], dtype=np.uint32)
            ng2 = int(ng[2]); ng3 = int(ng[3]); ng4 = float(ng[4])
            np.random.set_state((ng0, ng1, ng2, ng3, ng4))

            # python random (JSON transforma tuplas em listas, então re-tuple recursivo)
            def _to_tuple(x):
                if isinstance(x, list):
                    return tuple(_to_tuple(i) for i in x)
                return x

            pr = st["python_random_state"]
            random.setstate(_to_tuple(pr))

            print("[OK] RNG states restaurados.")
        except Exception as e:
            rotate_incompatible_path(cfg.rng_path, "rng_restore_failed")
            print(f"[WARN] Restore do RNG falhou: {type(e).__name__}: {e}")

def _sanitize_policy_targets(legal: "np.ndarray", target: "np.ndarray") -> "np.ndarray":
    """
    Garante que target da policy:
      - zera ações ilegais,
      - renormaliza para somar 1 dentro das ações legais,
      - evita NaN/inf.
    """
    import numpy as np

    t = np.asarray(target, dtype=np.float32, copy=True)
    l = np.asarray(legal, dtype=np.float32, copy=False)

    # zera ilegais
    t *= l

    # trata NaN/inf
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    # renormaliza por linha
    s = t.sum(axis=1, keepdims=True)
    # se linha zerou, volta para uniforme nas legais
    bad = (s <= 0.0)
    if np.any(bad):
        cnt = l.sum(axis=1, keepdims=True)
        cnt = np.maximum(cnt, 1.0)
        t[bad[:, 0]] = l[bad[:, 0]] / cnt[bad[:, 0]]
        s = t.sum(axis=1, keepdims=True)

    t /= np.maximum(s, 1e-12)
    return t

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

def _sample_adv_target_stats(buf, n: int = 8192):
    import numpy as np
    sz = _buffer_size(buf)
    if sz <= 0:
        return None
    b = int(min(n, sz))
    _, legal, target, _ = buf.sample_batch(b)
    legal = np.asarray(legal, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    t = np.abs(target * legal).reshape(-1)
    if t.size == 0:
        return None

    naninf = bool(np.isnan(t).any() or np.isinf(t).any())
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "abs_p50": float(np.percentile(t, 50)),
        "abs_p95": float(np.percentile(t, 95)),
        "abs_p99": float(np.percentile(t, 99)),
        "naninf": naninf,
    }


def _sample_pol_entropy(buf, n: int = 8192):
    import numpy as np
    sz = _buffer_size(buf)
    if sz <= 0:
        return None
    b = int(min(n, sz))
    _, legal, target, _ = buf.sample_batch(b)
    legal = np.asarray(legal, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    t = target * legal
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    s = t.sum(axis=1, keepdims=True)
    bad = (s <= 0.0)
    if np.any(bad):
        cnt = legal.sum(axis=1, keepdims=True)
        cnt = np.maximum(cnt, 1.0)
        t[bad[:, 0]] = legal[bad[:, 0]] / cnt[bad[:, 0]]
        s = t.sum(axis=1, keepdims=True)

    t = t / np.maximum(s, 1e-12)
    naninf = bool(np.isnan(t).any() or np.isinf(t).any())

    ent = -(t * np.log(np.maximum(t, 1e-12))).sum(axis=1)
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_p05": float(np.percentile(ent, 5)),
        "entropy_p50": float(np.percentile(ent, 50)),
        "naninf": naninf,
    }
    
def _sample_policy_action_support(buf, n: int = 8192):
    """
    Para cada ação a:
      - count_legal[a] = quantas vezes a ação estava legal (no batch amostrado)
      - mean_prob[a] = média do target_prob[a] apenas nos estados onde legal[a]==1
    """
    import numpy as np
    sz = _buffer_size(buf)
    if sz <= 0:
        return None

    b = int(min(n, sz))
    _, legal, target, _ = buf.sample_batch(b)
    legal = np.asarray(legal, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # sanitiza rapidamente (igual B011, mas sem dependência)
    t = target * legal
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    s = t.sum(axis=1, keepdims=True)
    bad = (s <= 0.0)
    if np.any(bad):
        cnt = np.maximum(legal.sum(axis=1, keepdims=True), 1.0)
        t[bad[:, 0]] = legal[bad[:, 0]] / cnt[bad[:, 0]]
        s = t.sum(axis=1, keepdims=True)
    t = t / np.maximum(s, 1e-12)

    num_actions = t.shape[1]
    count_legal = legal.sum(axis=0)  # (A,)
    mean_prob = np.zeros((num_actions,), dtype=np.float32)

    for a in range(num_actions):
        m = legal[:, a] > 0.0
        if np.any(m):
            mean_prob[a] = float(t[m, a].mean())
        else:
            mean_prob[a] = 0.0

    return {
        "count_legal": count_legal.astype(np.int64).tolist(),
        "mean_prob": mean_prob.tolist(),
    }

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
    # B043: permite isolar execuções (verificadores) em um diretório próprio
    _run_dir = os.environ.get("RUN_DIR", "").strip()
    base_dir = Path(_run_dir) if _run_dir else PROJECT_ROOT
    ensure_dirs(base_dir=base_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--traversals", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--worker_threads", type=int, default=None)
    ap.add_argument("--main_threads", type=int, default=None)
    ap.add_argument("--deterministic_merge", type=int, default=1)
    ap.add_argument("--bitwise", type=int, default=1)

    # B042: permite limitar o treino para verificadores (resume integrity)
    ap.add_argument("--max_iters", type=int, default=None)

    args = ap.parse_args()

    cfg = TrainCfg()
    # B043: redireciona checkpoints/logs quando RUN_DIR estiver definido
    cfg.ckpt_dir = base_dir / "checkpoints"
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
    # B042: max iters via CLI (para verificadores)
    if args.max_iters is not None:
        cfg.iterations = int(args.max_iters)

    # B042: override por ambiente (para verify_resume_integrity.py)
    _env_max = os.environ.get("MAX_ITERS_OVERRIDE", "").strip()
    if _env_max:
        try:
            cfg.iterations = int(_env_max)
        except Exception:
            pass

    # bitwise => single-thread no treino
    if cfg.bitwise:
        cfg.main_torch_threads = 1

    fault_path = base_dir / "logs" / "fault.log"
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
        import numpy as np
        import random

        # Seeds globais (tem que acontecer antes de qualquer treino/amostragem)
        random.seed(int(cfg.base_seed))
        np.random.seed(int(cfg.base_seed))
        torch.manual_seed(int(cfg.base_seed))

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

        sampler = ScenarioSampler(seed=int(cfg.base_seed))
        scen_cfg = dict(getattr(sampler, "config", {}))

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

        # Força estabilidade total: nunca depender de base_game_seed interno do trainer
        try:
            trainer.base_game_seed = int(cfg.base_seed)
        except Exception:
            pass

        stop_flag = {"stop": False}

        def _sigint(_sig, _frm):
            stop_flag["stop"] = True

        try:
            signal.signal(signal.SIGINT, _sigint)
        except Exception:
            pass
            
        # Determinismo forte no resume: não usar PID no seed do worker
        os.environ["SPIN_DETERMINISTIC_WORKERS"] = "1"

        # Base seed do jogo TEM que ser estável para resume bit-a-bit.
        fixed_bgs = _fixed_base_game_seed(cfg)

        # Força também no trainer (afeta Replayer e Traverser no processo principal)
        trainer.base_game_seed = int(fixed_bgs)

        pool = ProcessPoolExecutor(
            max_workers=int(cfg.rollout_workers),
            initializer=init_worker,
            initargs=(
                str(PROJECT_ROOT),
                str(ENV_DIR),
                int(obs_dim),
                int(cfg.base_seed),  # FIXO e estável entre execuções
                dict(scen_cfg),
                int(cfg.worker_threads),
                int(cfg.base_seed ^ 0x123456),
            ),
        )

        print("Training started. CTRL+C salva e sai limpo.")
        t0 = time.time()
        last_log = t0

        try:
            last_save_ts = time.time()
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
                # Divide exatamente o budget, sem forçar mínimo 1 por worker.
                total_trav = int(cfg.traversals_per_player)
                per = total_trav // w
                rem = total_trav % w

                adv_futures = {}
                adv_keys = []
                for p in range(3):
                    for k in range(w):
                        bud = per + (1 if k < rem else 0)
                        if bud <= 0:
                            continue

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
                    print("\n[WARN] Ctrl+C durante ADV collect. Cancelando futures e encerrando pool para salvar estado consistente...")

                    # Cancela futures já submetidos (não bloqueia)
                    try:
                        for f in list(adv_futures.keys()):
                            f.cancel()
                    except Exception:
                        pass

                    # Encerra o pool imediatamente para evitar spam de KeyboardInterrupt nos workers
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass

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
                
                total_ep = int(cfg.policy_episodes)
                per_ep = total_ep // w
                rem_ep = total_ep % w

                pol_futures = {}
                pol_keys = []
                for k in range(w):
                    ep = per_ep + (1 if k < rem_ep else 0)
                    if ep <= 0:
                        continue

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
                    print("\n[WARN] Ctrl+C durante POL collect. Cancelando futures e encerrando pool para salvar estado consistente...")

                    # Cancela futures já submetidos (não bloqueia)
                    try:
                        for f in list(pol_futures.keys()):
                            f.cancel()
                    except Exception:
                        pass

                    # Encerra o pool imediatamente para evitar spam de KeyboardInterrupt nos workers
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass

                    # IMPORTANT: não faz merge parcial
                    raise
                pol_added = [0, 0, 0]
                action_counts = [[0] * 7 for _ in range(3)]

                pol_iter_keys = sorted(pol_keys) if cfg.deterministic_merge else pol_keys

                for key in pol_iter_keys:
                    out, counts = pol_results[key]
                    for p in range(3):
                        obs, legal, target, pid = out[p]
                        target = _sanitize_policy_targets(legal, target)
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

                # B036: Warm-up correto
                # Durante warm-up treinamos normalmente, mas quando termina, limpamos buffers
                # para não contaminar o dataset "oficial" com dados do início aleatório.
                if cfg.warmup_iters and trainer.iteration == cfg.warmup_iters:
                    print(f"[INFO] Warm-up finalizado em iter={trainer.iteration}. Limpando buffers...")
                    for p in range(3):
                        trainer.adv_buffers[p].clear(reseed=int(cfg.base_seed) + 1000 + p)
                        trainer.pol_buffers[p].clear(reseed=int(cfg.base_seed) + 2000 + p)
                    print("[OK] Buffers limpos. A partir daqui o dataset é 'limpo'.")

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
                    
                    # Stats de escala do sinal, detecta saturação e NaN cedo
                    adv_s = [_sample_adv_target_stats(trainer.adv_buffers[p], n=8192) for p in range(3)]
                    pol_s = [_sample_pol_entropy(trainer.pol_buffers[p], n=8192) for p in range(3)]

                    def _fmt_adv(i):
                        s = adv_s[i]
                        if not s:
                            return "empty"
                        return f"p50={s['abs_p50']:.3f} p95={s['abs_p95']:.3f} p99={s['abs_p99']:.3f} naninf={int(s['naninf'])}"

                    def _fmt_pol(i):
                        s = pol_s[i]
                        if not s:
                            return "empty"
                        return f"mean={s['entropy_mean']:.3f} p05={s['entropy_p05']:.3f} p50={s['entropy_p50']:.3f} naninf={int(s['naninf'])}"

                    print("adv_abs BB stats, p0: " + _fmt_adv(0) + " , p1: " + _fmt_adv(1) + " , p2: " + _fmt_adv(2))
                    print("pol_entropy stats, p0: " + _fmt_pol(0) + " , p1: " + _fmt_pol(1) + " , p2: " + _fmt_pol(2))                  
                    
                    pol_sup = [_sample_policy_action_support(trainer.pol_buffers[p], n=8192) for p in range(3)]

                    def _fmt_sup(i):
                        s = pol_sup[i]
                        if not s:
                            return "empty"
                        cl = s["count_legal"]
                        mp = s["mean_prob"]
                        parts = []
                        for a in range(len(cl)):
                            parts.append(f"a{a}: legal={cl[a]} mean={mp[a]:.4f}")
                        return " | ".join(parts)

                    print("pol_action_support p0: " + _fmt_sup(0))
                    print("pol_action_support p1: " + _fmt_sup(1))
                    print("pol_action_support p2: " + _fmt_sup(2))
                    
                    print(
                        "actions p0: " + _fmt_action_dist(action_counts[0]) + " | "
                        "p1: " + _fmt_action_dist(action_counts[1]) + " | "
                        "p2: " + _fmt_action_dist(action_counts[2])
                    )
                    last_log = now

                do_save = False

                # Compatibilidade: save por iteração, apenas se save_every > 0
                if cfg.save_every and (trainer.iteration % cfg.save_every == 0):
                    do_save = True

                # B019: save por tempo (default 8h)
                now_ts = time.time()
                if cfg.save_every_seconds and (now_ts - last_save_ts) >= float(cfg.save_every_seconds):
                    do_save = True

                if do_save:
                    if cfg.skip_saves_during_warmup and (trainer.iteration < cfg.warmup_iters):
                        pass
                    else:
                        print("[INFO] Salvando checkpoint + buffers...")
                        save_all(trainer, cfg)
                        print("[OK] Salvo.")
                        last_save_ts = now_ts

        except KeyboardInterrupt:
            stop_flag["stop"] = True
            print("\n[INFO] Interrompido pelo usuário (CTRL+C). Salvando estado consistente...")
        finally:
            # B037: encerra workers antes de salvar para evitar spam de KeyboardInterrupt nos processos
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

            print("\nParando, salvando checkpoint + buffers...")
            try:
                save_all(trainer, cfg)
                print("[OK] Salvo. Saindo.")
            except Exception as _e:
                print(f"[WARN] Falha ao salvar checkpoint/buffers: {_e}")


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