from __future__ import annotations
import os, json, tempfile
import numpy as np
from dataclasses import dataclass

@dataclass
class ReservoirConfig:
    obs_dim: int = 292
    num_actions: int = 7
    capacity: int = 2_000_000
    dtype: str = "float32"

class ReservoirBuffer:
    """
    Reservoir sampling buffer com:
      - add() item-a-item
      - add_batch() rápido quando ainda não encheu (cópia blocada)
      - save() atômico (evita checkpoint corrompido em queda de energia)
    """
    VERSION = 2

    def __init__(self, cfg: ReservoirConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        self.capacity = int(cfg.capacity)
        self.obs_dim = int(cfg.obs_dim)
        self.num_actions = int(cfg.num_actions)
        self.dtype = np.float32 if cfg.dtype == "float32" else np.float64

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=self.dtype)
        self.legal = np.zeros((self.capacity, self.num_actions), dtype=self.dtype)
        self.target = np.zeros((self.capacity, self.num_actions), dtype=self.dtype)
        self.player_id = np.zeros((self.capacity,), dtype=np.int16)

        self.n_seen = 0
        self.size = 0

    def add(self, obs: np.ndarray, legal_mask: np.ndarray, target: np.ndarray, player_id: int):
        obs = np.asarray(obs, dtype=self.dtype)
        legal_mask = np.asarray(legal_mask, dtype=self.dtype)
        target = np.asarray(target, dtype=self.dtype)

        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs shape {obs.shape} != ({self.obs_dim},)")
        if legal_mask.shape != (self.num_actions,):
            raise ValueError(f"legal shape {legal_mask.shape} != ({self.num_actions},)")
        if target.shape != (self.num_actions,):
            raise ValueError(f"target shape {target.shape} != ({self.num_actions},)")

        self.n_seen += 1

        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            j = int(self.rng.integers(0, self.n_seen))
            if j >= self.capacity:
                return
            idx = j

        self.obs[idx] = obs
        self.legal[idx] = legal_mask
        self.target[idx] = target
        self.player_id[idx] = int(player_id)

    def add_batch(self, obs: np.ndarray, legal: np.ndarray, target: np.ndarray, player_id: np.ndarray):
        """
        Fast path:
          - Enquanto o buffer ainda não encheu, copia blocos inteiros (muito mais rápido).
          - Depois que encheu, cai no add() item-a-item (reservoir sampling).
        """
        obs = np.asarray(obs, dtype=self.dtype)
        legal = np.asarray(legal, dtype=self.dtype)
        target = np.asarray(target, dtype=self.dtype)
        player_id = np.asarray(player_id, dtype=np.int16)

        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs batch shape {obs.shape} != (N,{self.obs_dim})")
        if legal.ndim != 2 or legal.shape[1] != self.num_actions:
            raise ValueError(f"legal batch shape {legal.shape} != (N,{self.num_actions})")
        if target.ndim != 2 or target.shape[1] != self.num_actions:
            raise ValueError(f"target batch shape {target.shape} != (N,{self.num_actions})")
        if player_id.ndim != 1 or player_id.shape[0] != obs.shape[0]:
            raise ValueError("player_id shape mismatch")

        n = int(obs.shape[0])
        if n == 0:
            return

        # 1) bloco inicial até encher
        if self.size < self.capacity:
            free = self.capacity - self.size
            k = min(n, free)
            s = self.size
            e = s + k
            self.obs[s:e] = obs[:k]
            self.legal[s:e] = legal[:k]
            self.target[s:e] = target[:k]
            self.player_id[s:e] = player_id[:k]
            self.size += k
            self.n_seen += k
            if k == n:
                return
            # sobrou
            obs = obs[k:]
            legal = legal[k:]
            target = target[k:]
            player_id = player_id[k:]
            n = int(obs.shape[0])

        # 2) buffer cheio: reservoir item-a-item
        for i in range(n):
            self.add(obs[i], legal[i], target[i], int(player_id[i]))

    def sample_batch(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("buffer vazio")
        b = int(batch_size)
        idx = self.rng.integers(0, self.size, size=b, dtype=np.int64)
        return (self.obs[idx], self.legal[idx], self.target[idx], self.player_id[idx])

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "cfg": {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "capacity": self.capacity,
                "dtype": "float32" if self.dtype == np.float32 else "float64",
            },
            "n_seen": int(self.n_seen),
            "size": int(self.size),
            "rng_state": self.rng.bit_generator.state,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta_path = path + ".json"
        npz_path = path + ".npz"

        # escreve em temporário e faz replace (atômico)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(meta_path), encoding="utf-8") as tf:
            json.dump(self.state_dict(), tf)
            tmp_meta = tf.name

        tmp_npz = npz_path + ".tmp.npz"
        np.savez_compressed(
            tmp_npz,
            obs=self.obs[:self.size],
            legal=self.legal[:self.size],
            target=self.target[:self.size],
            player_id=self.player_id[:self.size],
        )

        os.replace(tmp_meta, meta_path)
        os.replace(tmp_npz, npz_path)

    @classmethod
    def load(cls, path: str, seed_fallback: int = 0) -> "ReservoirBuffer":
        meta_path = path + ".json"
        npz_path = path + ".npz"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg_d = meta["cfg"]
        cfg = ReservoirConfig(
            obs_dim=int(cfg_d["obs_dim"]),
            num_actions=int(cfg_d["num_actions"]),
            capacity=int(cfg_d["capacity"]),
            dtype=str(cfg_d["dtype"]),
        )

        buf = cls(cfg, seed=seed_fallback)
        buf.n_seen = int(meta["n_seen"])
        buf.size = int(meta["size"])

        data = np.load(npz_path)
        buf.obs[:buf.size] = data["obs"]
        buf.legal[:buf.size] = data["legal"]
        buf.target[:buf.size] = data["target"]
        buf.player_id[:buf.size] = data["player_id"]

        try:
            buf.rng.bit_generator.state = meta["rng_state"]
        except Exception:
            pass

        return buf
