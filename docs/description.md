# Deep Spin, Manual Completo (Spin & Go 3-max, DeepCFR, CPU-only)

Este documento explica o projeto Deep Spin em nível de “manual de produto”, com foco em:
- como compilar e rodar,
- como treinar por meses com resume confiável,
- como interpretar logs e desempenho,
- como o ambiente C++ funciona,
- como o ScenarioSampler funciona,
- como o vetor de observação (OBS_DIM=338) está organizado.

---

Objetivo: Criar um bot de poker, na modalidade Spin Go, de altíssimo nível usando DeepCFR.
O treino ocorrerá num Ryzen 9 9950 X, com 64GB de RAM e sem GPU.
Como deve ser um treino de longa duração é muito provável que haja interrupções no treino, por isso os checkpoints devem recuperar, com o máximo de fidelidade, o estado anterior.
O ambiente é derivado do código Open Source do RLcard que é originalmente feito para Cash Game Six Max, mas aqui foi adaptado para torneios Spin Go.
Uma das adaptações mais críticas foi a mudança no número de jogadores, aqui devemos ter muito cuidado em separar mãos que começam headsup (O jogador no SB também é o BTN), de mãos que estão headsup porque um jogador foldou na mão.
Outro desafio da adaptação forão as blinds variáveis e a simulação de torneio treinando apenas mãos independentes e não torneios completos.
Para as simlações foram usadas estatísticas de base de dados reais sobre a probabilidade de uma mão ser HU e sobre a probabilidade dos jogadores terem determinadas faixas de Stacks.
Alguns bugs críticos e erros de lógica que surgiram nas primeiras versões foram: 
1- a simulação de HU sempre "matar" o mesmo jogador, enviesando o treinamento dos players. 
2- ações de raise por fração do pote ignoravam o call embutido
3- All-in runout incompleto
4- Cálculo errado de Side pots multiway
5- Features que guardavam as stacks dos jogares e outras informações mas sem associar um jogador a uma posições.
6- Muitos outros bugs menores!

Considero que a principal dificuldade de treinar um DeepCFR com essa dificuldade, seja não perceber bugs ou falhas de lógicas escondidas, eeu cheguei a treinar com um conjunto de features que não diferencia as cartas do board das cartas da mão, com outro que calculava errado tamanhos de raises; outro que tratava jogos HU como se fossem 3-way, e tudo isso passa quase despercebido, não gera nenhum alerta no LOG.
Vc já editou o poker_env e está rodando aparentemente sem erros.
Agora vou compartilhar todos os outros arquivos do projeto, quero que vc procure por erros, falhas de lógica ou qualquer coisa que possa prejudicar o desenvolvimento da IA.
Para isso, vc deve estar atento às regras do Poker, à proposta do código, a integração entre os scripts, de forma a criarmos o melhor Bot possível, sem bugs e falhas de lógica escondidas, com as ferramentas que temos.
Novos Anexos:
1- train_deepcfr.py
2- eval_policy.py
3- traversal.py
4- trainer.py
5- scenario.py
6- rollout_workers.py
7- networks.py
8- buffers.py
9- log_manus.txt (o Log do treino com o poker_env que vc editou)

Resumo da Tarefa:
1- Encontrar bugs e corrigí-los.
2- Encontrar falhas de lógicas que prejudiquem o treinamento como ele foi idealizado.
3- Melhorar os scripts com foco em performance e eficiência de treino, elevação do teto de aprendizado e, principalmente detecção de bugs e falhas por meio de logs e scripts de debug ou evaluation, pois não adianta treinar por meses para depois descobrir que treinou errado.
4- Verificar se com o log gerado após sua edição no poker_env tudo parece estar correndo bem e se devo continuar com o treino.
5- Sua opinião sobre a capacidade do projeto de gerar um bot de altíssimo nível, capaz até mesmo de vencer humanos profissionais em competições homem x máquina.



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

## 9. Observação do agente: OBS_DIM = 292 (layout v2)

### Dimensões da IA (FEATURES)
# 1) CARDS: (104)
- [0..51]   = mão do herói (one-hot 52)
- [52..103] = board público (one-hot 52)

Card index:
- índice = suit*13 + (rank-2)
- suit: Spades=0, Hearts=1, Diamonds=2, Clubs=3
- rank: 2..A

# 2) NUMERIC: 20  (tudo normalizado em BB quando aplicável) *** AQUI FOI MODIFICADO PARA RELACIONAR OS STACK OU BETS A UMA POSIÇÃO (Não sei se relaciona antes) ***
- [104] big_blind_chips 
- [105] effective_stack_bb = min(stack_total) entre jogadores ativos (não folded)
- [106] stack behind do jogador dealer em BB
- [107] stack behind do jogador no small em BB
- [108] stack behind do jogador no bigblind em BB
- [109] max_stack_total_bb
- [110] min_stack_total_bb
- [111] current_bets/raised do jogador dealer em BB (na street atual)
- [112] current_bets/raised do jogador no small em BB (na street atual)
- [113] current_bets/raised do jogador no bigblind em BB (na street atual)
- [114] pot em BB (pot é o pot total, diferente de potcommon pois potcommon não considera as apostas atuais!)
- [115] potcommon (pot - atapostas atuais) ***AQUI FOI MODIFICADO PARA MOSTRA O POTCOMMON, ANTES MOSTRAVA O STACK DO HERO< MAS JÁ TEMOS O STACK POR POSIÇÃO E A POSIÇÃO DO HERO!***
- [116] amount_to_call_bb = round_.to_call - round_.raised[my_id] em BB (clamp >= 0)
- [117] amount_to_call_over_pot = call / pot (ótimo pra IA entender “call barato vs caro”)
- [118] spr_effective = effective_stack_bb / pot_bb_total
- [119] last_bet_size_bb (a última aposta/raise enfrentada pelo herói em BB, quando aplicável)
- [120] num_active_players (2 ou 3)
- [121] num_allin_players
- [122] num_folded_players
- [123] raises_this_street (raises + bets)

# 3) STREET: one-hot: (4)
- [124..127] preflop/flop/turn/river

# 4)POSIÇÕES REALATIVAS: (11):
  [128] game_is_hu_HUSB - hero no SB
  [129] game_is_hu_HUBB - hero no BB
  [130] hand_is_hu_3wBTNvBB (SB fold) - hero no BTN
  [131] hand_is_hu_3wBBvBTN (SB fold) - hero no BB
  [132] hand_is_hu_3wBTNvSB (BB fold) - hero no BTN
  [133] hand_is_hu_3wSBvBTN (BB fold) - hero no SB
  [134] hand_is_hu_3wSBvBB (BTN fold) - hero no SB
  [135] hand_is_hu_3wBBvSB (BTN fold) - hero no BB
  [136] 3wBTN - hero no BTN
  [137] 3wSB - hero no SB
  [138] 3wBB - hero no BB
       

# 5) HAND STRENGH:  – one-hot –(40)

HS00 air/low high-card (≤ Q-high)
HS01 K/A high-card
HS02 pair onboard (best hand uses 0 hole, pair category)
HS03 bottom pair
HS04 middle pair
HS05 top pair weak kicker (kicker ≤ T)
HS06 top pair good kicker (kicker ≥ J)
HS07 underpair (pocket pair not above board top)
HS08 overpair (pocket pair above board top)
HS09 two pair onboard (uses 0 hole)
HS10 two pair uses 1 hole
HS11 two pair uses 2 hole
HS12 trips onboard (uses 0 hole)
HS13 trips uses 1 hole (board paired + one hole)
HS14 set (pocket pair + board)
HS15 straight 1-hole LOW (hole used rank 2–6)
HS16 straight 1-hole MID (7–T)
HS17 straight 1-hole HIGH (J–A)
HS18 straight 2-hole LOW (max hole used 2–6)
HS19 straight 2-hole MID (7–T)
HS20 straight 2-hole HIGH (J–A)
HS21 board straight, you do NOT improve
HS22 board straight, you DO improve
HS23 flush 1-hole LOW (2–6)
HS24 flush 1-hole MID (7–T)
HS25 flush 1-hole HIGH (J–A)
HS26 flush uses 2 hole
HS27 board flush, you do NOT improve
HS28 board flush, you DO improve
HS29 board full house, you do NOT improve
HS30 board full house, you DO improve
HS31 full house uses 1 hole
HS32 full house uses 2 hole
HS33 board quads, you do NOT improve kicker
HS34 board quads, you DO improve kicker
HS35 quads uses 1 hole
HS36 quads uses 2 hole
HS37 straight flush onboard, you do NOT improve
HS38 straight flush uses 1 hole
HS39 straight flush uses 2 hole


# 6) DRAWS AND OUTS (12)
	// [187] gutshot
	// [188] oesd
	// [189] flush_draw
	// [190] combo_draw
	// [191] overcards_1   (== 1 OVER CARD)
	// [192] overcards_2   (== 2 OVER CARDS)
	// [193] bd_flush_draw
	// [194] bd_straight_draw
	// [195] middle_over_1
	// [196] middle_over_2
	// [197] under_1
	// [198] under_2


# 7) BOARD TEXTURE: (28) 
vec[0] – flush_possible
vec[1] – straight_possible
vec[2] – turn_is_middle_vs_flop_top2
vec[3] – river_is_middle_vs_first4_top2
vec[4] – both_turn_and_river_between_flop_second_and_flop_bottom
vec[5] – turn_is_under_or_equal_flop_min
vec[6] – river_is_under_or_equal_flop_min
vec[7] – turn_pairs_flop_any
vec[8] – river_pairs_any_previous
vec[9] – turn_pairs_flop_top
vec[10] – river_pairs_pre_top
vec[11] – turn_completes_board_straight_from_flop
vec[12] – turn_completes_board_flush_from_flop
vec[13] – river_completes_board_straight_from_turn
vec[14] – river_completes_board_flush_from_turn
vec[15] – river_does_not_complete_existing_flush_draw_suit
vec[16] – flop_has_3_broadways
vec[17] – flop_has_2_broadways
vec[18] – flop_has_1_broadway
vec[19] – flop_has_0_broadways
vec[20] – flop_has_zero_middle_ranks_6_to_9
vec[21] – flop_is_rainbow
vec[22] – flop_top_is_A_single
vec[23] – flop_top_is_K_single
vec[24] – flop_top_is_Q_single
vec[25] – turn_is_A_single_in_first4
vec[26] – river_is_A_single_in_all5
vec[27] – flush_possible_duplicate *** REMOVER ***
vec[28] – straight_possible_duplicate *** REMOVER ***
vec[29] – board_has_any_duplicate_rank *** REMOVER ***
vec[30] – flop_is_monotone
vec[31] – flop_is_twotone (no monotone, no raiwnbow)


# 8) CURRENT ACTION CONTEXT: (26 = total = 1 +13 +12)
A) Contínuo (1) *** AQUI PRECISO VERIFICAR O QUE FOI TRANSFERIDO PARA NUMERIC PARA REMOVER AS DUPLICIDADES ***
- aggressor_flag (1 se herói foi agressor na street passada! *** TEM QUE SER EM RELAÇÃO A STREET PASSADA! PARECE QUE ESTAVA ATUAL AQUI***) ***Aqui eram 7 dimensões, mas foram movidas pra numéric, restou apenas uma***


B) Postflop categórico (13) 
0_act_first - Street acabou de começar, ninguém agiu ainda.
1_vs_check - Não é a primeira ação da street, e to_call_bb == 0, então alguém checou e você está diante de check.
2_vs_lowbet - “facing a bet”, herói ainda não apostou nesta street, não é reraise! Enfrentando aposta <= 0.40 do potcommon ***parece que aqui estava do "pot" mas aí já incluiria a própria aposta que estamos enfrentando, o que seria um erro!***
3_vs_normalbet - “facing a bet”, herói ainda não apostou nesta street, não é reraise! Enfrentando aposta <= 0.60 do potcommon ***parece que aqui estava do "pot" mas aí já incluiria a própria aposta que estamos enfrentando, o que seria um erro!***
4_vs_highbet - “facing a bet”, herói ainda não apostou nesta street, não é reraise! Enfrentando aposta <= 1.1 do potcommon ***parece que aqui estava do "pot" mas aí já incluiria a própria aposta que estamos enfrentando, o que seria um erro!***
5_vs_overbet - “facing a bet”, herói ainda não apostou nesta street, não é reraise! Enfrentando aposta > 1.1 do potcommon ***parece que aqui estava do "pot" mas aí já incluiria a própria aposta que estamos enfrentando, o que seria um erro!***
6_did_lowbet_got_raised_Normal
7_did_normalbet_got_raised_Normal
8_did_highbet_got_raised_Normal
9_did_lowbet_got_raised_Over
10_did_normalbet_got_raised_Over
11_did_highbet_got_raised_Over
12_vs_reraise

C) Preflop categórico (12)
Inclui:
- posição (BTN/SB/BB) *** REMOVER POIS  JÁ ESTÁ EM POSIÇÕES RELATIVAS ***
- HU/3way *** REMOVER POIS  JÁ ESTÁ EM POSIÇÕES RELATIVAS ***
- estado do pote (6) (unopened, limped, open-raised, 3bet+, iso, limp-raised)  
- bucket do tamanho enfrentado *** REMOVER POIS A IA SÓ IRÁ TREINAR CONTRA MINI RAISE / LIMP ou ALL_IN COMO ABERTURA, O QUE CONSIDERO UMA ABSTRAÇÃO BEM EFETIVA DADO AO QUE É PRATICA NOS JOGOS REAIS)***
- bucket do “hero_prev_action” (6) 0 none / 1 limp/call blind / 2 call raise / 3 open / 4 3bet+ / 5 iso *** As 4bets devem ser sempre ALL-in ENTão Não tem pq registrar aqui!
- posição do last raiser *** REMOVER POIS JÁ ESTÁ EM POSIÇÕES RELATIVAS ***
- is_heads_up_now flag (num_active==2, inclui game_is_hu e hand_is_hu) *** REMOVER POIS JÁ ESTÁ EM POSIÇÕES RELATIVAS ***

---

# 9) HISTORY (39) *** AQUI NÃO ENTRA RIVER POIS REFERE-SE A STREETS PASSADAS! PARA A STREET ATUAL TEMOS O CURRENT ACTION CONTEXT!***
Três “StreetSummary” fixos:
- Preflop: 11 dims ( 10  + 1 ) ***ACTION_COUNTS E SNAPSHOTS REMOVIDOS POIS CONSIDERO INFORMAÇÃO GENÉRICA POUCO ÚTIL, ME CORRIJA SE EU ESTIVER ERRADO)***
- Flop: 14 dims ( 13  + 1 )  ***ACTION_COUNTS E SNAPSHOTS REMOVIDOS POIS CONSIDERO INFORMAÇÃO GENÉRICA POUCO ÚTIL, ME CORRIJA SE EU ESTIVER ERRADO)***
- Turn: 14 dims ( 13  + 1 )  ***ACTION_COUNTS E SNAPSHOTS REMOVIDOS POIS CONSIDERO INFORMAÇÃO GENÉRICA POUCO ÚTIL, ME CORRIJA SE EU ESTIVER ERRADO)***

Cada StreetSummary inclui:
- contexto básico (ctx)
- action_counts (7) ***ACTION_COUNTS REMOVIDOS POIS CONSIDERO INFORMAÇÃO GENÉRICA POUCO ÚTIL, ME CORRIJA SE EU ESTIVER ERRADO)***
- aggressor_flag (1)
- stacks_snapshot (4) *** REMOVER ***
- pot_snapshot (4) *** REMOVER ***
- bet_snapshot (4) *** REMOVER ***

Detalhamento dos contextos:
Preflop:
One-hot do “contexto enfrentado” no preflop, calculado por
[0] UNOPENED, ninguém abriu ainda (sem ações registradas no hist_pre, além dos blinds)
[1] VS_LIMP_SINGLE, 1 limp antes de você agir
[2] VS_LIMP_MULTI, 2+ limps antes de você agir
[3] VS_OPEN_RAISE_BTN, enfrenta open raise do BTN
[4] VS_OPEN_RAISE_SB, enfrenta open raise do SB
[5] VS_RAISE_CALL, enfrenta open raise que teve pelo menos 1 call (raise, call)
[6] VS_RAISE_3BET, você não abriu, há 2+ raises antes de você agir (3bet, 4bet+, etc)
[7] VS_3BET, você abriu e está enfrentando 3bet ou 4bet+
[8] LIMP_RAISED, você limpou e alguém isolou, você está enfrentando esse iso
[9] ISO_RAISED, você isolou e o limper re-aumentou (limp-raise vs iso)

Flop e Turn
One-hot do “contexto básico de ação na street” (pós-flop) em 13 buckets,
[0] act_first, herói age primeiro na street (nenhuma ação registrada ainda)
[1] vs_check, alguém deu check e você está com ação (nada a pagar)
[2] vs_bet_small, enfrenta bet pequena (por exemplo, to_call/pot <= 0.40)
[3] vs_bet_mid, enfrenta bet média (<= 0.60)
[4] vs_bet_big, enfrenta bet grande (<= 1.10)
[5] vs_bet_over, enfrenta overbet (> 1.10)
[6] raise_normal_vs_small, você betou small e enfrenta raise “normal” (não over)
[7] raise_normal_vs_mid
[8] raise_normal_vs_big
[9] raise_over_vs_small, você betou small e enfrenta raise “over”
[10] raise_over_vs_mid
[11] raise_over_vs_big
[12] vs_reraise, há pelo menos um re-raise na street (reraise detectado)


---

# 10) LEGAL ACTION MASK (7) *** Mantém como está ***
Final do vetor:
- 7 floats {0,1} indicando ações legais no estado atual.

---

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