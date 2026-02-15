# Deep Spin, Manual Completo (Spin & Go 3-max, DeepCFR, CPU-only)

Este documento explica o projeto Deep Spin em nível de “manual de produto”, com foco em:
- como compilar e rodar,
- como treinar por meses com resume confiável,
- como interpretar logs e desempenho,
- como o ambiente C++ funciona,
- como o ScenarioSampler funciona,
- como o vetor de observação (OBS_DIM=338) está organizado.

---

## 1. Visão geral

O Deep Spin treina um agente de Poker (Spin & Go 3 jogadores, Winner-Takes-All) usando DeepCFR / External Sampling.
O objetivo é estabilidade e desempenho em CPU, com checkpoint/restore confiável por longos períodos.

Componentes principais:
- `env/` C++: motor do jogo (cpoker via pybind11)
- `deepcfr/` Python: buffers, redes, traversal, trainer (DeepCFR)
- `scripts/` Python: treino, sanity checks, avaliação e debug

---

## 2. Estrutura resumida do repositório

- `env/`
  - `poker_env.cpp`, `poker_env.h` (motor C++ + binding pybind11)
  - `setup.py` (build do módulo cpoker)
- `deepcfr/`
  - `scenario.py` (ScenarioSampler, frequências reais, choose_dealer_id_for_episode)
  - `trainer.py` (DeepCFRTrainer, checkpoint/restore)
  - `traversal.py` (External Sampling / DeepCFR traversal)
  - `rollout_workers.py` (workers de coleta)
  - `buffers.py` (reservatórios ADV/POL)
  - `networks.py` (redes, heads, forward)
- `scripts/`
  - `train_deepcfr.py` (treino principal)
  - `sanity_check.py` (sanidade do motor + sampler)
  - `eval_policy.py` (avaliação de policy)
  - `hand_history_debug.py` (histórico completo de uma mão, estilo PT4)

- `ASSISTANT_CONTEXT.md` (índice + regras do projeto)
- `docs/PROJECT_LOG.md` (histórico/versionamento + roadmap)

---

## 3. Pré-requisitos e setup (Windows, CPU-only)

Recomendado:
- Python (o mesmo usado no build do cpoker)
- Visual Studio Build Tools (MSVC)
- pybind11, numpy, torch (CPU)

Crie e ative o venv, instale dependências (exemplo):
- `pip install -r requirements.txt` (se houver)
ou manualmente:
- `pip install pybind11 numpy torch`

---

## 4. Compilação do motor C++ (cpoker)

Dentro de `env/`:

- `python setup.py build_ext --inplace`

Isso gera um arquivo `.pyd` (Windows) que será importado como `cpoker`.

Dica: sempre compile com o MESMO Python do venv que você usa pra rodar treino/sanity.

---

## 5. Execução rápida: sanity antes de qualquer treino longo

No root do projeto:

- `venv\Scripts\python.exe -u scripts\sanity_check.py --episodes 5000`

O sanity_check reporta:
- showdown com board < 5 (deve ser 0),
- violações de affordability (deve ser 0),
- pot/stack negativos (deve ser 0),
- distribuição simples de ações por diff (to_call==0 vs >0).

---

## 6. Treino: `scripts/train_deepcfr.py`

Exemplo típico (CPU Ryzen 9):
- `venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0`

Conceitos:
- `workers`: processos/atores que coletam (rollouts/traversals)
- `worker_threads`: threads internas por worker (geralmente 1 em Windows pra evitar overhead)
- `main_threads`: threads de torch no processo principal
- `traversals`: quantos traversals por iteração
- `episodes`: quantos episódios por iteração (mãos)
- `deterministic_merge=1`: merge determinístico (ordena antes de inserir buffers)
- `bitwise`: controle interno de determinismo (depende do código)

Recomendação prática CPU:
- comece com `workers` alto e `worker_threads=1`
- ajuste `main_threads` de forma que não “brigue” com workers

---

## 7. Checkpoint/restore e resume longo

O checkpoint deve incluir:
- pesos das redes,
- estado dos buffers (ADV/POL),
- métricas essenciais,
- RNG do ScenarioSampler.

Padrão esperado:
- você interrompe com Ctrl+C,
- o treino salva checkpoint consistente,
- ao reiniciar, o treino continua sem “pular” estados essenciais.

---

## 8. Action space (7 ações)

O motor e a rede trabalham com 7 ações discretas (índices 0..6), com semântica No-Limit:

0. FOLD
1. CHECK/CALL
2. BET/RAISE 33% pot
3. BET/RAISE 50% pot
4. BET/RAISE 75% pot
5. BET/RAISE pot
6. ALL-IN

Notas importantes:
- O custo real de raise é sempre: `diff (to_call) + raise_add`.
- `legal_actions` só inclui raise se `diff + raise_add <= remained`.
- Se `diff > remained`, o jogador não consegue pagar o call: ações devem ser limitadas apropriadamente (não listar raise impossível).

---

## 9. Observação do agente: OBS_DIM = 338 (layout v2)

A observação é um vetor `float[338]` organizado assim:

### 9.1 Base (235)
1) Cards: 104
- [0..51]   = mão do herói (one-hot 52)
- [52..103] = board público (one-hot 52)

Card index:
- índice = suit*13 + (rank-2)
- suit: Spades=0, Hearts=1, Diamonds=2, Clubs=3
- rank: 2..A

2) Numeric: 8  (tudo normalizado em BB quando aplicável)
- [104..106] stacks (3 jogadores) em BB
- [107..109] current_bets/raised (3 jogadores) em BB (na street atual)
- [110] pot em BB
- [111] hero_stack / (pot+eps)

3) Street one-hot: 4
- [112..115] preflop/flop/turn/river

4) Position scenario: 11
- [116] HUSB
- [117] HUBB
- [118] 3wBTNvBB (SB fold)
- [119] 3wBBvBTN
- [120] 3wBTNvSB (BB fold)
- [121] 3wSBvBTN
- [122] 3wSBvBB (BTN fold)
- [123] 3wBBvSB
- [124] 3pBTN
- [125] 3pSB
- [126] 3pBB

5) Hand strength (granular): 17  -> [127..143]
Esta é uma heurística rápida (não é avaliação perfeita de todas as categorias).
Buckets (resumo):
- [127] air/low high-card
- [128] K/A high-card
- [129] bottom pair
- [130] underpair (pocket pair abaixo do board)
- [131] middle pair
- [132] pocket pair médio / (também usado preflop para PP baixo)
- [133] top pair weak kicker
- [134] top pair good kicker
- [135] overpair / premium pocket pair
- [136] two pair
- [137] trips (1 hole card)
- [138] set (pocket pair + board)
- [139] straight/flush “não usando 2 hole cards” ou board-made
- [140] straight usando 2 hole cards
- [141] flush usando 1 hole card
- [142] flush usando 2 hole cards
- [143] quads

Observação: full house e straight flush não aparecem como buckets explícitos aqui. Isso é uma heurística, útil como feature, mas não uma “hand evaluator perfeita”.

6) Draws: 5  -> [144..148]
- [144] none
- [145] gutshot
- [146] open-ended straight draw (OESD)
- [147] flush draw
- [148] combo draw (straight + flush)

7) Board texture: 31 -> [149..179]
Sinais avançados de textura do board:
- flush possible (1 suit >=2 no flop, >=3 turn/river)
- straight possible (conectividade/4-run etc)
- paired board, trips on board
- monotone/2-tone/rainbow
- high/low composition
- gaps e “wetness” aproximada
(Ver `analyze_advanced_board_texture` no C++ para a lista completa por índice.)

8) Hero relative to board: 4 -> [180..183]
- [180] hero has pair with board
- [181] hero has 2 pair+ with board
- [182] hero blocks flush suit (tem carta do naipe dominante)
- [183] hero has overcards (duas overcards vs top card)

9) Action context: 51 -> [184..234]
A) Contínuo (7) [184..190]
- to_call_bb / (pot_bb+eps)
- hero_stack_bb / (pot_bb+eps)
- eff_stack_bb / (pot_bb+eps)
- normalized_pot = log(1+pot_bb)/log(1+total_chips_bb)
- aggressor_flag (1 se herói foi agressor na street)
- num_active_scaled
- is_heads_up flag

B) Postflop categórico (13) [191..203]
- [191] to_call == 0
- [192..196] faced bet buckets por to_call/pot: verylow/low/normal/high/over
- [197..202] facing raise buckets: depende do tamanho do último bet do herói e se o raise é >2x
- [203] vs_reraise flag

C) Preflop categórico (31) [204..234]
Inclui:
- posição (BTN/SB/BB)
- HU/3way
- estado do pote (unopened, limped, open-raised, 3bet, 4bet+, iso, limp-raised)
- bucket do tamanho enfrentado
- bucket do “hero_prev_action”
- posição do last raiser
- flags especiais: unopened, got_isolated, facing limp-raise, facing 4bet+

---

### 9.2 History (96)
Três “StreetSummary” fixos:
- Preflop: 30 dims
- Flop: 33 dims
- Turn: 33 dims

Cada StreetSummary inclui:
- contexto básico (ctx)
- action_counts (7)
- aggressor_flag (1)
- stacks_snapshot (4)
- pot_snapshot (4)
- bet_snapshot (4)

---

### 9.3 Legal action mask (7)
Final do vetor:
- 7 floats {0,1} indicando ações legais no estado atual.

---

## 10. ScenarioSampler (frequências reais)

Arquivo: `deepcfr/scenario.py`

Responsável por gerar cenários realistas:
- blind level (sb/bb),
- stacks iniciais (3p ou HU),
- total_chips,
- flag is_heads_up.

Requisitos de integridade:
- tabelas completas (sem simplificação),
- RNG reprodutível (numpy Generator),
- state save/restore compatível com checkpoint.

A função `sample()` retorna um dict com:
- `is_heads_up`
- `stacks`
- `sb`
- `bb`
- `total_chips`
- `blind_level`

---

## 11. Debug de mão (hand history)

Use:
- `venv\Scripts\python.exe -u scripts\hand_history_debug.py --seed 123 --policy random`

Ele imprime:
- stacks, blinds, dealer
- ações por street com pot/to_call
- board e hole cards de todos os jogadores (modo debug)
- payoffs no final

---

## 12. Monitoramento de desempenho

O log do treino deve te mostrar, por iteração:
- tempo em collect vs train
- sps / throughput
- adv_added / pol_added
- losses
- distribuição de ações (fold/call/33/50/75/pot/allin)

Regras práticas:
- se collect domina muito, gargalo está no traversal/env
- se train domina, ajuste batch sizes/threads
- se CPU fica baixa, verifique threads, spawn e afinidade

---

## 13. Roadmap recomendado (alto nível)

Fase 1 (agora):
- estabilidade absoluta (sanity limpo, Ctrl+C limpo, resume ok)
- treinos longos ChipEV por mão

Fase 2:
- avaliação robusta (torneios completos ou proxies melhores)
- opcional: PrizeEV shaping (WTA)
- tuning de performance (cache por worker, evitar overhead)

---

## 14. Troubleshooting rápido

- “ImportError deepcfr”: rodar scripts sempre a partir do root, ou ajustar sys.path
- sanity mostra raise affordability >0: problema no C++ legal_actions/custo de raise
- checkpoint crash em scenario_rng: RNG não é numpy Generator ou restore incompatível
- board<5 em showdown: bug no fast-forward de all-in

Fim.
