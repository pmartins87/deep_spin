# Deep Spin, Project Log e Versionamento

Este arquivo é o “diário técnico” do projeto:
- bugs críticos encontrados e corrigidos,
- decisões arquiteturais,
- compatibilidade de checkpoints,
- roadmap curto e longo.

Ele NÃO substitui o `docs/MANUAL.md`.
O Manual explica “como usar”. Aqui explica “o que mudou e por quê”.

---

## Release atual

### Nome sugerido
**DeepSpin R1, Baseline Sanity-Passed**  
(Primeira release que passa sanity_check sem violações de affordability e sem inconsistências óbvias.)

Data: 2026-02-14

Critério de release:
- `scripts/sanity_check.py --episodes 5000` sem violações
- compilação cpoker ok
- Ctrl+C no treino salva checkpoint consistente (validação manual)

---

## Bugs críticos corrigidos (histórico resumido)

### 1) Raise com diff > 0 (BUG CRÍTICO)
Sintoma:
- ações de raise por fração do pote ignoravam o call embutido (diff),
- EV de ações caras ficava brutalmente errado,
- política podia colapsar para fold/call.

Correção:
- custo real do raise = diff + raise_add
- legal_actions valida (diff + raise_add) <= remained
- raised[player] reflete mx + raise_add (não “apenas bet(q)”)

Impacto:
- Muda o MDP efetivo. Checkpoints gerados antes disso NÃO são confiáveis.

---

### 2) Checkpoint crash: scenario_rng NoneType.state
Sintoma:
- checkpoint tentava salvar `rng.bit_generator.state`, mas rng não era numpy Generator.

Correção:
- ScenarioSampler usa `np.random.default_rng(seed)`
- checkpoint salva/restaura RNG com fallback robusto

Impacto:
- Corrige estabilidade de long run e resume.

---

### 3) Bug silencioso de shuffle em slice (viesava stacks)
Sintoma:
- `rng.shuffle(stacks[:2])` não alterava a lista original.

Correção:
- shuffle feito em uma estrutura que realmente afeta stacks.

Impacto:
- Remove viés silencioso de posição x stack.

---

### 4) All-in runout incompleto
Status:
- Recheck confirmou que o fast-forward completa flop/turn/river antes de encerrar.
- sanity_check confirma showdown com board<5 = 0.

---

### 5) Side pots multiway
Status:
- motor implementa side pots por níveis de contribuição e distribui por elegibilidade.

---

## Compatibilidade de checkpoints

### Quando reiniciar do zero
- Se seu checkpoint/buffers foram gerados ANTES da correção do raise diff+raise_add.

### Quando pode continuar
- Se o checkpoint foi gerado DEPOIS das correções críticas e você confirma:
  - sanity_check limpo
  - sha256 do scenario compatível com a versão atual
  - treino consegue Ctrl+C e resume sem inconsistência

---

## Sanity checks atuais

- board<5 em showdown: esperado 0
- raise affordability (postflop pot-fraction): esperado 0
- preflop raise quando não pode call: esperado 0
- pot/stack negativo: esperado 0

---

## Roadmap curto (próximos passos)

1) Treino curto de validação (10–30 min)
- validar Ctrl+C e resume
- validar logs e throughput

2) Instrumentação extra
- log de distribuição de ações por street e por posição
- alertas automáticos se colapso de ação (ex.: fold extremo)

3) Debug tools
- `hand_history_debug.py` para reproduzir mãos e discutir bugs “no detalhe”

---

## Roadmap longo

1) Melhorar objetivo para WTA
- manter ChipEV por mão como baseline
- fase 2: PrizeEV shaping (winprob do stack) ou simulação de torneio completo

2) Melhorias de features
- hand strength heurística pode ganhar buckets extras (ex.: full house explícito)
- mais sinais de textura/linha do pote

3) Performance (CPU-only)
- cache de objetos por worker
- reduzir overhead de pickle/spawn no Windows
- métricas por iteração (collect/train/sps) sempre visíveis

################### V2


# PROJECT_LOG

## v2 (cpoker v51 C++), build estável + correções críticas de integridade

### Objetivo da v2
Consolidar a versão C++ (v51) como um port fiel da lógica v50 (Python), mantendo:
- Progressão de rounds/streets consistente
- Legal-actions consistentes com as abstrações
- Estado (obs) com dimensão fixa (OBS_DIM=338), sem escrita fora do buffer
- Determinismo de RNG (Dealer e PokerGame) para checkpoints robustos
- Correções de HU dentro de ambiente 3-handed com assento morto (dead seat)

---

## 1) Correções de integridade do motor (Game/Round)

### 1.1 Dead seat correto (HU dentro de 3-handed)
Foi implementado o conceito de **dead seat v50-style**:
- `status == FOLDED`
- `remained_chips == 0`
- `in_chips == 0`
- `hand.empty()`

Isso é usado para distinguir:
- `game_is_hu`: mão já começou HU (assento morto estrutural)
- `hand_is_hu`: mão começou 3-way e alguém foldou durante a mão

### 1.2 Round::is_over corrigido para não travar em HU com dead seat
`Round::is_over()` passa a contar FOLDED e ALLIN como “not playing”.
Sem isso, HU dentro de um env 3P pode entrar em loop / não finalizar street.

### 1.3 proceed_round() agora pula corretamente ALLIN/FOLDED
Após aplicar uma ação, o `game_pointer` avança e **pula qualquer jogador não ALIVE**.
Isso evita pedir ação para ALLIN.

### 1.4 All-in fast-forward sem board incompleto
`advance_stage_if_needed()` foi corrigido para NÃO encerrar a mão com board incompleto quando todos estão ALLIN:
- Se ninguém pode agir (ALLIN/FOLDED), ainda assim são dealadas as cartas restantes até completar 5 comunitárias.
Isso corrige um bug extremamente destrutivo de EV (showdown com board incompleto).

---

## 2) Determinismo e checkpoints

### 2.1 RNG do Dealer e do PokerGame serializáveis
Foram adicionadas funções:
- `Dealer::get_rng_state()` / `Dealer::set_rng_state()`
- `PokerGame::get_rng_state()` / `PokerGame::set_rng_state()`
- `PokerGame::get_dealer_rng_state()` / `PokerGame::set_dealer_rng_state()`

Objetivo: permitir checkpoint “perfeito” (reprodutível).

### 2.2 Segurança contra cópia/move (ponteiros internos)
Como `Round` guarda ponteiro para `Dealer`, foi endurecido:
- Copy ctor / move ctor / assignments ajustam `round_.dealer = &dealer_`
- `clone()` retorna `std::unique_ptr<PokerGame>` e rewire do ponteiro do dealer no clone

---

## 3) Preflop: raise custom e sincronização de ponteiros

### 3.1 Rota de “raise custom” (BB-based sizing)
No preflop, as ações de raise (2..5) são mapeadas para sizings BB:
- Unopened: 2bb
- Vs limp(s): 2.5bb + 1bb por limper extra
- Vs open: 5bb + 2bb por caller pós-open
- Vs 3bet+: sem raise não-allin, usar ALL_IN

### 3.2 Fix crítico no bloco de raise custom
No caminho de raise custom foram corrigidos:
- Sincronização `round_.game_pointer = game_pointer_`
- Avanço pulando FOLDED e ALLIN
- Se raiser vira ALLIN: `round_.to_act -= 1` para evitar double-count na lógica de término do round
- Se q <= 0: remap para CHECK/CALL
- Se q >= stack: remap para ALL_IN
- Em todos os remaps: histórico e summary registram a ação aplicada (applied_action)

---

## 4) Legal actions (máscara + regras)

### 4.1 Máscara legal_mask dentro da obs
A obs inclui `legal_mask[7]` no final, alinhada com ações 0..6.

### 4.2 get_legal_actions() com regras específicas do preflop
Preflop não usa pot-fraction raises (pot pequeno), então:
- Retorna set base {FOLD, CHECK_CALL, raises(2..5), ALL_IN}
- Filtra ações inválidas (não deixa 7/8 escapar)
- Se to_call >= stack: remove raises e ALL_IN (evita “all-in raise” ambíguo quando é só call-allin)
- Contra 4bet+: shove-or-fold (call só se for call-allin)
- Mantém apenas 1 raise não-allin permitido por cenário (unopened/limp/open)

---

## 5) Estado (obs) v2: OBS_DIM = 338

### 5.1 Layout consolidado
- Cards: 104
- Numeric: 8
- Street: 4
- Position scenario: 11 (com distinção game_is_hu vs hand_is_hu)
- Hand strength: 17 (heurística)
- Draws: 5
- Board texture: 31
- Hero vs board: 4
- Action context: 51
- History v2: 96 (street summaries preflop/flop/turn por jogador)
- legal_mask: 7
Total: 338

### 5.2 debug_raw_obs desligado por padrão
`debug_raw_obs_ = false` e exposto via setter/getter.

---

## 6) Notas conhecidas (para v3)
- Numeric[8] ainda está baseado em “seat index 0..2”, não em posições (BTN/SB/BB).
- Hand strength (17) ainda é heurística, precisa virar avaliação perfeita por categoria (conforme spec v3).
- potcommon na v2 é calculado em step() para faced_ctx, mas na obs o slot [111] ainda não está fixo como potcommon (vai ser alinhado na v3).
