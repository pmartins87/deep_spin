# -*- coding: utf-8 -*-
"""
Scenario sampler para Spin & Go, com stacks realistas baseados em frequências reais.

Inclui:
- choose_dealer_id_for_episode, exigida pelo train_deepcfr.py
- Tabela COMPLETA (sem simplificações) de frequências por blind level, para 3 jogadores e Heads-Up

Observação sobre a linha "OUTROS":
A tabela original tem uma categoria agregada "Outros". Como ela não informa o intervalo exato,
esta implementação amostra um valor que NÃO cai em nenhum dos intervalos explícitos, dentro de [0, total_chips].
Isso preserva a massa de probabilidade da categoria, sem inventar um intervalo específico.

Gerado em 2026-02-14T02:10:35.735070Z
"""
from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ======================================================================================
#  Helper exigida pelo train_deepcfr.py
# ======================================================================================
def choose_dealer_id_for_episode(rng, stacks, sb, bb, is_hu):
    """Compat, para o treino e a avaliação.
    
    - Sorteia o dealer (BTN) entre jogadores com stack > 0.
    - Não altera stacks.
    - Mesmo que algum stack seja menor que o blind, o engine já faz clamp
      no post (Player.bet), então não precisamos de ajustes artificiais aqui.
    """
    alive = [i for i, s in enumerate(stacks) if s > 0]
    if not alive:
        return 0
    try:
        return int(rng.choice(alive))
    except Exception:
        # fallback para random.Random
        return int(alive[int(rng.random() * len(alive))])

# ======================================================================================
#  Tabelas completas de frequência, extraídas da tabela de mãos reais enviada pelo usuário
# ======================================================================================

# Blind levels suportados pelo projeto
BLIND_LEVELS: List[Tuple[int, int]] = [
  (10, 20),
  (15, 30),
  (20, 40),
  (30, 60),
  (40, 80),
  (50, 100),
  (60, 120),
  (80, 160),
  (100, 200),
]

# Pesos de blind level, a partir do "% do nível" da sua tabela
# 3 jogadores: 19877, 11425, 5321, 1432, 205, 31, 3, 0, 0
REAL_BLIND_WEIGHTS_3P: List[float] = [51.91, 29.83, 13.9, 3.74, 0.54, 0.08, 0.01, 0.0, 0.0]

# Heads-up: 2897, 7895, 9494, 7219, 3345, 895, 156, 21, 5
REAL_BLIND_WEIGHTS_HU: List[float] = [9.07, 24.73, 29.74, 22.61, 10.48, 2.8, 0.49, 0.07, 0.02]

# Tabela COMPLETA, sem simplificações, percentuais por faixa e por blind level
REAL_STACK_TABLE_3P: List[Dict[str, Any]] = [
  {
    "range": [
      500,
      500
    ],
    "pct": {
      "10/20": 23.0,
      "15/30": 0.04,
      "20/40": 0.0,
      "30/60": 0.0,
      "40/80": 0.0,
      "50/100": 0.0,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      470,
      530
    ],
    "pct": {
      "10/20": 15.59,
      "15/30": 2.45,
      "20/40": 1.22,
      "30/60": 0.91,
      "40/80": 0.49,
      "50/100": 0.0,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      440,
      560
    ],
    "pct": {
      "10/20": 18.01,
      "15/30": 5.93,
      "20/40": 3.33,
      "30/60": 2.23,
      "40/80": 1.95,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      410,
      590
    ],
    "pct": {
      "10/20": 11.68,
      "15/30": 9.09,
      "20/40": 4.75,
      "30/60": 3.49,
      "40/80": 3.9,
      "50/100": 12.9,
      "60/120": 33.33,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      380,
      620
    ],
    "pct": {
      "10/20": 8.83,
      "15/30": 10.56,
      "20/40": 5.94,
      "30/60": 5.73,
      "40/80": 7.32,
      "50/100": 3.23,
      "60/120": 33.33,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      350,
      650
    ],
    "pct": {
      "10/20": 5.46,
      "15/30": 10.12,
      "20/40": 7.59,
      "30/60": 6.49,
      "40/80": 5.85,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      320,
      680
    ],
    "pct": {
      "10/20": 4.1,
      "15/30": 9.96,
      "20/40": 8.18,
      "30/60": 9.78,
      "40/80": 5.37,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      290,
      710
    ],
    "pct": {
      "10/20": 2.6,
      "15/30": 8.62,
      "20/40": 7.84,
      "30/60": 6.91,
      "40/80": 7.8,
      "50/100": 6.45,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      260,
      740
    ],
    "pct": {
      "10/20": 1.97,
      "15/30": 7.33,
      "20/40": 9.23,
      "30/60": 9.08,
      "40/80": 7.32,
      "50/100": 6.45,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      230,
      770
    ],
    "pct": {
      "10/20": 1.57,
      "15/30": 7.05,
      "20/40": 9.08,
      "30/60": 7.89,
      "40/80": 9.27,
      "50/100": 0.0,
      "60/120": 33.33,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      200,
      800
    ],
    "pct": {
      "10/20": 0.89,
      "15/30": 5.58,
      "20/40": 8.01,
      "30/60": 7.68,
      "40/80": 6.83,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      170,
      830
    ],
    "pct": {
      "10/20": 0.75,
      "15/30": 4.43,
      "20/40": 6.73,
      "30/60": 8.1,
      "40/80": 7.8,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      140,
      860
    ],
    "pct": {
      "10/20": 0.84,
      "15/30": 3.54,
      "20/40": 6.43,
      "30/60": 6.08,
      "40/80": 4.39,
      "50/100": 6.45,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      110,
      890
    ],
    "pct": {
      "10/20": 0.65,
      "15/30": 3.4,
      "20/40": 5.04,
      "30/60": 4.96,
      "40/80": 7.32,
      "50/100": 6.45,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      80,
      920
    ],
    "pct": {
      "10/20": 0.79,
      "15/30": 3.27,
      "20/40": 3.51,
      "30/60": 3.98,
      "40/80": 5.85,
      "50/100": 9.68,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      50,
      950
    ],
    "pct": {
      "10/20": 0.73,
      "15/30": 2.26,
      "20/40": 3.25,
      "30/60": 4.96,
      "40/80": 4.88,
      "50/100": 6.45,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      20,
      980
    ],
    "pct": {
      "10/20": 1.14,
      "15/30": 2.31,
      "20/40": 2.88,
      "30/60": 4.82,
      "40/80": 3.9,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      1,
      999
    ],
    "pct": {
      "10/20": 0.34,
      "15/30": 0.95,
      "20/40": 1.65,
      "30/60": 1.89,
      "40/80": 0.49,
      "50/100": 3.23,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      1,
      1499
    ],
    "pct": {
      "10/20": 1.07,
      "15/30": 3.06,
      "20/40": 5.36,
      "30/60": 5.03,
      "40/80": 9.27,
      "50/100": 19.35,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  }
]

REAL_STACK_TABLE_HU: List[Dict[str, Any]] = [
  {
    "range": [
      700,
      800
    ],
    "pct": {
      "10/20": 2.07,
      "15/30": 9.08,
      "20/40": 10.63,
      "30/60": 12.33,
      "40/80": 11.96,
      "50/100": 12.18,
      "60/120": 11.54,
      "80/160": 9.52,
      "100/200": 20.0
    }
  },
  {
    "range": [
      650,
      850
    ],
    "pct": {
      "10/20": 2.73,
      "15/30": 9.09,
      "20/40": 10.71,
      "30/60": 11.75,
      "40/80": 11.96,
      "50/100": 11.62,
      "60/120": 10.26,
      "80/160": 19.05,
      "100/200": 0.0
    }
  },
  {
    "range": [
      600,
      900
    ],
    "pct": {
      "10/20": 5.35,
      "15/30": 9.18,
      "20/40": 10.48,
      "30/60": 11.75,
      "40/80": 12.08,
      "50/100": 13.97,
      "60/120": 12.82,
      "80/160": 9.52,
      "100/200": 0.0
    }
  },
  {
    "range": [
      550,
      950
    ],
    "pct": {
      "10/20": 10.11,
      "15/30": 10.94,
      "20/40": 11.03,
      "30/60": 10.86,
      "40/80": 10.94,
      "50/100": 10.73,
      "60/120": 13.46,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      500,
      1000
    ],
    "pct": {
      "10/20": 24.4,
      "15/30": 11.11,
      "20/40": 10.7,
      "30/60": 10.56,
      "40/80": 11.84,
      "50/100": 10.84,
      "60/120": 4.49,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      450,
      1050
    ],
    "pct": {
      "10/20": 29.96,
      "15/30": 11.82,
      "20/40": 9.59,
      "30/60": 9.0,
      "40/80": 9.51,
      "50/100": 8.27,
      "60/120": 9.62,
      "80/160": 14.29,
      "100/200": 20.0
    }
  },
  {
    "range": [
      400,
      1100
    ],
    "pct": {
      "10/20": 13.29,
      "15/30": 11.45,
      "20/40": 9.14,
      "30/60": 8.64,
      "40/80": 7.71,
      "50/100": 7.93,
      "60/120": 8.33,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      350,
      1150
    ],
    "pct": {
      "10/20": 5.42,
      "15/30": 9.51,
      "20/40": 7.15,
      "30/60": 7.3,
      "40/80": 6.76,
      "50/100": 5.25,
      "60/120": 4.49,
      "80/160": 14.29,
      "100/200": 40.0
    }
  },
  {
    "range": [
      300,
      1200
    ],
    "pct": {
      "10/20": 2.97,
      "15/30": 6.95,
      "20/40": 6.86,
      "30/60": 5.94,
      "40/80": 5.53,
      "50/100": 5.14,
      "60/120": 7.69,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      250,
      1250
    ],
    "pct": {
      "10/20": 1.38,
      "15/30": 4.41,
      "20/40": 5.28,
      "30/60": 4.03,
      "40/80": 3.98,
      "50/100": 4.02,
      "60/120": 3.85,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": [
      200,
      1300
    ],
    "pct": {
      "10/20": 1.0,
      "15/30": 2.88,
      "20/40": 3.73,
      "30/60": 3.21,
      "40/80": 2.84,
      "50/100": 4.02,
      "60/120": 7.05,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      150,
      1350
    ],
    "pct": {
      "10/20": 0.35,
      "15/30": 1.7,
      "20/40": 2.16,
      "30/60": 1.69,
      "40/80": 2.0,
      "50/100": 2.12,
      "60/120": 4.49,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      100,
      1400
    ],
    "pct": {
      "10/20": 0.21,
      "15/30": 1.08,
      "20/40": 1.16,
      "30/60": 1.29,
      "40/80": 1.46,
      "50/100": 1.79,
      "60/120": 0.64,
      "80/160": 4.76,
      "100/200": 0.0
    }
  },
  {
    "range": [
      50,
      1450
    ],
    "pct": {
      "10/20": 0.55,
      "15/30": 0.46,
      "20/40": 0.66,
      "30/60": 0.87,
      "40/80": 0.96,
      "50/100": 1.34,
      "60/120": 0.64,
      "80/160": 0.0,
      "100/200": 20.0
    }
  },
  {
    "range": [
      0,
      1500
    ],
    "pct": {
      "10/20": 0.21,
      "15/30": 0.34,
      "20/40": 0.73,
      "30/60": 0.78,
      "40/80": 0.48,
      "50/100": 0.78,
      "60/120": 0.64,
      "80/160": 0.0,
      "100/200": 0.0
    }
  },
  {
    "range": "OUTROS",
    "pct": {
      "10/20": 0.0,
      "15/30": 0.0,
      "20/40": 0.0,
      "30/60": 0.0,
      "40/80": 0.0,
      "50/100": 0.0,
      "60/120": 0.0,
      "80/160": 0.0,
      "100/200": 0.0
    }
  }
]


def _normalize(weights: List[float]) -> List[float]:
    s = sum(weights)
    if s <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / s for w in weights]


def _range_contains(r: Tuple[int, int], x: int) -> bool:
    return r[0] <= x <= r[1]


def _sample_stack_from_table(
    rng,
    table: List[Dict[str, Any]],
    blind_key: str,
    total_chips: int,
) -> int:
    """Amostra UM stack (marginal) para um jogador, usando a tabela completa e o blind_key.

    Requisitos importantes:
    - determinismo/resume: o RNG deve ser restaurável via checkpoint (preferencialmente numpy Generator)
    - a linha OUTROS preserva massa de probabilidade, sem inventar intervalos
    """
    weights: List[float] = []
    ranges: List[Optional[Tuple[int, int]]] = []

    for row in table:
        pct = float(row.get('pct', {}).get(blind_key, 0.0) or 0.0)
        weights.append(pct)

        rr = row.get('range', None)
        if rr is None:
            ranges.append(None)
        else:
            lo, hi = int(rr[0]), int(rr[1])
            ranges.append((lo, hi))

    probs = _normalize(weights)

    # rng.choice (numpy Generator) is preferred; fallback keeps compatibility.
    try:
        idx = int(rng.choice(len(probs), p=probs))
    except Exception:
        # random.Random fallback
        idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])

    r = ranges[idx]

    if r is not None:
        lo, hi = r
        if lo == hi:
            return lo
        try:
            # numpy Generator: high is exclusive
            return int(rng.integers(lo, hi + 1))
        except Exception:
            return int(rng.randint(lo, hi))

    # OUTROS: amostra um valor fora de TODOS os intervalos explícitos
    explicit = [rr for rr in ranges if rr is not None]
    for _ in range(50):
        try:
            x = int(rng.integers(0, total_chips + 1))
        except Exception:
            x = int(rng.randint(0, total_chips))
        if all(not _range_contains(rr, x) for rr in explicit):
            return x

    # fallback, se algo muito estranho acontecer
    try:
        return int(rng.integers(0, total_chips + 1))
    except Exception:
        return int(rng.randint(0, total_chips))


class ScenarioSampler:
    """
    Gera cenários para um episódio, o episódio aqui é UMA MÃO.

    Diferença principal para o ScenarioSampler antigo:
    - agora stacks são amostrados por frequência real, condicionados ao blind level e ao modo (3P ou HU)
    - total_chips é fixo em 1500, para Spin & Go padrão com 500 iniciais
    """

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.config = dict(config) if config is not None else {}

        # RNG: use numpy Generator to enable robust checkpoint/restore via bit_generator.state.
        # (trainer.py also has a fallback for random.Random checkpoints.)
        self.rng = np.random.default_rng(int(seed))

        # total fixo, Spin & Go 500 cada, 3 jogadores
        self.total_chips: int = int(self.config.get("force_total_chips", 1500))

        # Probabilidade de o episódio ser HU, mantém compatibilidade
        # Se não existir, usa a fração implícita na sua amostra real, que dá ~0.4548
        self.heads_up_prob: float = float(self.config.get("heads_up_prob", 0.4548))

        # Blind levels, usa config se existir, senão usa padrão acima
        self.blind_levels: List[Tuple[int, int]] = list(self.config.get("blind_levels", BLIND_LEVELS))

        # Usa pesos reais por padrão, para refletir o que aparece nas mãos reais
        self.blind_weights_3p: List[float] = list(self.config.get("blind_weights_3hand", REAL_BLIND_WEIGHTS_3P))
        self.blind_weights_hu: List[float] = list(self.config.get("blind_weights_hu", REAL_BLIND_WEIGHTS_HU))

        # Garante alinhamento com blind_levels
        if len(self.blind_weights_3p) != len(self.blind_levels):
            self.blind_weights_3p = [1.0] * len(self.blind_levels)
        if len(self.blind_weights_hu) != len(self.blind_levels):
            self.blind_weights_hu = [1.0] * len(self.blind_levels)

        # Precompute normalized probabilities (float64) for fast sampling
        self._blind_probs_3p = np.asarray(_normalize(self.blind_weights_3p), dtype=np.float64)
        self._blind_probs_hu = np.asarray(_normalize(self.blind_weights_hu), dtype=np.float64)

        # Precomputa map blind -> key string "sb/bb"
        self._blind_key = {(sb, bb): f"{sb}/{bb}" for sb, bb in self.blind_levels}

        # Para 3P, a sua tabela não tem mãos em 80/160 e 100/200, então as colunas são 0.
        # Para evitar distribuição vazia se algum blind for sorteado nesses níveis, fazemos fallback para 60/120.
        self._fallback_3p_blind = "60/120"

    def _sample_blinds(self, is_hu: bool) -> Tuple[int, int]:
        probs = self._blind_probs_hu if is_hu else self._blind_probs_3p
        idx = int(self.rng.choice(len(self.blind_levels), p=probs))
        sb, bb = self.blind_levels[idx]
        return int(sb), int(bb)

    def _sample_stacks_hu(self, blind_key: str) -> List[int]:
        # Amostra um stack e usa o complemento, garante stacks válidos
        for _ in range(50):
            a = _sample_stack_from_table(self.rng, REAL_STACK_TABLE_HU, blind_key, self.total_chips)
            if 1 <= a <= self.total_chips - 1:
                b = self.total_chips - a
                if b > 0:
                    stacks = [a, b, 0]
                    # BUGFIX: não usar slice, pois cria cópia e não embaralha in-place.
                    if float(self.rng.random()) < 0.5:
                        stacks[0], stacks[1] = stacks[1], stacks[0]
                    return stacks
        # fallback
        a = int(self.rng.integers(1, self.total_chips))
        return [a, self.total_chips - a, 0]

    def _sample_stacks_3p(self, blind_key: str) -> List[int]:
        # Se a coluna do blind_key for toda 0, cai no fallback 60/120
        # Isso pode acontecer em 80/160 e 100/200, pois você não tem mãos 3P nesses níveis na amostra.
        # (Na prática, a chance já é praticamente zero, pois REAL_BLIND_WEIGHTS_3P também zera esses níveis.)
        if blind_key in ("80/160", "100/200"):
            blind_key = self._fallback_3p_blind

        # Amostra marginalmente e normaliza para somar total_chips
        for _ in range(50):
            raw = [
                _sample_stack_from_table(self.rng, REAL_STACK_TABLE_3P, blind_key, self.total_chips),
                _sample_stack_from_table(self.rng, REAL_STACK_TABLE_3P, blind_key, self.total_chips),
                _sample_stack_from_table(self.rng, REAL_STACK_TABLE_3P, blind_key, self.total_chips),
            ]
            # Evita 0
            raw = [max(1, int(x)) for x in raw]
            s = sum(raw)
            if s <= 0:
                continue

            # Escala para total
            scaled = [int(round(x * self.total_chips / s)) for x in raw]
            # Ajusta soma
            diff = self.total_chips - sum(scaled)
            if diff != 0:
                imax = max(range(3), key=lambda i: scaled[i])
                scaled[imax] += diff

            # Garante >=1
            for i in range(3):
                if scaled[i] <= 0:
                    j = max(range(3), key=lambda k: scaled[k])
                    if scaled[j] > 1:
                        scaled[j] -= 1
                        scaled[i] = 1

            if sum(scaled) == self.total_chips and min(scaled) > 0:
                perm = self.rng.permutation(3)
                scaled = [scaled[int(i)] for i in perm]
                return scaled

        # fallback bem seguro
        a = int(self.rng.integers(1, self.total_chips - 1))
        b = int(self.rng.integers(1, self.total_chips - a))
        c = self.total_chips - a - b
        out = [a, b, c]
        perm = self.rng.permutation(3)
        out = [out[int(i)] for i in perm]
        return out

    def sample(self) -> Dict[str, Any]:
        is_hu = (self.rng.random() < self.heads_up_prob)

        sb, bb = self._sample_blinds(is_hu=is_hu)
        blind_key = self._blind_key.get((sb, bb), f"{sb}/{bb}")

        if is_hu:
            stacks = self._sample_stacks_hu(blind_key)
            dead = [i for i, s in enumerate(stacks) if s <= 0]
        else:
            stacks = self._sample_stacks_3p(blind_key)
            dead = []

        return {
            "total_chips": self.total_chips,
            "is_heads_up": bool(is_hu),
            "sb": sb,
            "bb": bb,
            "stacks": stacks,
            "dead_players": dead,
        }
