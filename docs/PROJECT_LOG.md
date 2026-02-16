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

Fim.