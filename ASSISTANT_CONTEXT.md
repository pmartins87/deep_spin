# ASSISTANT_CONTEXT.md

## 1) Objetivo do projeto

Construir um agente de Poker **Spin & Go (3-handed com mistura realista de HU)** usando **DeepCFR (External Sampling MCCFR + redes neurais)**, rodando **CPU-only** por enquanto, com arquitetura e logs “blindados” para:
- Criar um bot de altíssimo nível, capaz de vencer humanos proffisonais gabaritados do clube de poker da faculdade, que desafiaram os alunos de Ciências da Computação a criar um bot de Poker que os vencesse ao longo de uma grande amostragem.
- Treino longo e monitorável (evitar “2 meses depois descobrir bug oculto”).
- Determinismo por seed (reprodutibilidade).
- Evoluir para GPU futuramente sem reescrever a base do projeto.
- Abstração de ações fixa (7 ações) com regras consistentes de pré-flop e pós-flop.
- Observação (infoset) rica e estável (atualmente **OBS_DIM varia conforme versão do env**, ver seção 4).

## 2) Como rodar

### 2.1 Ambiente Python (Windows)
1) Ativar venv:
- `C:\PokerAI\spin_deepcfr\venv\Scripts\activate`

2) (Re)compilar o módulo C++ (cpoker) quando houver mudanças no `poker_env.cpp/.h`:
- `cd C:\PokerAI\spin_deepcfr\env`
- `python setup.py build_ext --inplace`

> Observação: o `cpoker.cp314-win_amd64.pyd` existe em **duas possíveis localizações**:
- `C:\PokerAI\spin_deepcfr\env\cpoker.cp314-win_amd64.pyd` (onde é gerado)
- `C:\PokerAI\spin_deepcfr\cpoker.cp314-win_amd64.pyd` (onde é executado)

3) O treino irá rodar num Ryzen9 9950x com 64GB de RAM sem GPU.

### 2.2 Treino recomendado (CPU-only)
Com equilíbrio bom entre velocidade e reprodutibilidade “prática”:

cd C:\PokerAI\spin_deepcfr
venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0

Notas:
- `--deterministic_merge 1`: merge **ordenado** dos resultados dos workers antes de inserir nos buffers, melhora confiança no resume.
- `--bitwise 0`: não exige “bitwise determinism” total (muito caro em CPU-only). O treino fica bem mais rápido.
- `--main_threads 12`: libera o torch para usar mais CPU no treino (apesar de hoje o gargalo ser o collect).

**Checkpoint:**
- CTRL+C deve fazer “saída limpa”, salvando checkpoint e buffers.

### 2.3 Parar e retomar (checkpoint)

- Pressione **CTRL+C**: o script cancela futures, evita merge parcial, salva checkpoint+búnker e sai sem stacktrace.
- Para retomar: rode o mesmo comando. O script deve imprimir:

```
[OK] Restored ... checkpoint.pt, iteration=...
[OK] Buffers restaurados.
```

---

## 3) Árvore de pastas resumida


```
spin_deepcfr/
ASSISTANT_CONTEXT.md
last_log.txt
validar_env.py
smoke_test_env.py
  env/
    poker_env.cpp
    poker_env.h
    setup.py
    cpoker*.pyd
  deepcfr/
    __init__.py
    buffers.py
    networks.py
    scenario.py
    trainer.py
    traversal.py
    rollout_workers.py
  scripts/
    train_deepcfr.py
    eval_policy.py
  checkpoints/
    checkpoint.pt
    buffers/
      adv_p0.npz ...
      pol_p0.npz ...
```

---

## 4) Decisões importantes já tomadas

### 4.1 Ações abstratas (NUM_ACTIONS = 7)
Índices e labels:
0. FOLD
1. CHECK/CALL
2. BET_33
3. BET_50
4. BET_75
5. BET_POT
6. ALL_IN

### 4.2 Regras de pré-flop (BB-based)
- Pré-flop NÃO usa pot-percentage para sizings; usa tamanhos em BB e mapeia para labels.
- A abstração é fixa, 7 ações, o env mapeia para ações legais reais de cada estado.
- O objetivo do env é ser consistente e “poker-correto”, não maximizar complexidade de sizing.

**Estado do pote / contexto pré-flop (ctx10):**
0 UNOPENED  
1 VS_LIMP_SINGLE  
2 VS_LIMP_MULTI  
3 VS_OPEN_RAISE_BTN  
4 VS_OPEN_RAISE_SB  
5 VS_RAISE_CALL  
6 VS_RAISE_3BET  
7 VS_3BET  
8 LIMP_RAISED  
9 ISO_RAISED  

**Sizings pré-flop simplificados (foco em eficiência):**
- unopened: raise = 2bb (mapeia para BET_33) + ALL_IN
- vs limp: raise = 2.5bb + ( +1bb por caller adicional ) (mapeia para BET_75) + ALL_IN
- vs raise: raise = 5bb + ( +2bb por caller adicional ) (mapeia para BET_50) + ALL_IN
- vs 4bet+: {FOLD, ALL_IN} e CHECK/CALL só se call for all-in

> Importante: BTN pode fazer limp (não vamos bloquear), porque queremos o bot aprendendo a enfrentar estilos reais.

### 4.3 HU vs 3-handed “de verdade”
Não tratar HU como “3-handed com um foldado”.
O modo HU deve ser gerado/representado de forma consistente para não confundir:
- seats realmente fora do jogo vs folded
- estado correto de blinds e posições
- ordem de ação correta

### 4.4 Observação (infoset) e OBS_DIM e detecção automática
- O env C++ retorna `state["obs"]` já como vetor float32.
- O Python deve **detectar OBS_DIM do env**, não hardcodar.
- Mudanças no vetor (ex.: 260 → 293 → 338) quebram checkpoint (esperado e aceito).

### 4.5 legal_mask dentro do obs
O obs inclui **legal_mask[7]** (0/1) alinhada com as 7 ações, para facilitar estabilidade e evitar ações impossíveis.

### 4.6 Histórico
Histórico é essencial.
A direção do projeto é evoluir de “contadores pobres” para um histórico que capture:
- contexto das ações anteriores por street (tipo “action_index” + resposta),
- ordem (quem apostou primeiro, quem respondeu, etc),
- e agregados úteis (investido por street, agressor, etc).

### 4.7 Cenário
O cenário passou a usar estatísticas reais (hardcoded) para frequencias de stacks, nível de blinds e jogadores na mão.
- Foco em torneios com **500 fichas iniciais**.
- Usar frequências reais de stacks observadas (tabela de mãos reais), com versões:
  - 3-handed (distribuição por níveis de blind)
  - heads-up (distribuição por níveis de blind)

**Importante:** `scenario.py` precisa expor também a função:
- `choose_dealer_id_for_episode(...)` (o treino exige isso)

### 4.8 Episódio e objetivo do RL

- **Episódio = 1 mão**, termina quando `g.is_over()` fica True:
  - showdown, ou todos foldaram exceto 1.
- Objetivo atual: **ChipEV** por mão (ganhar fichas na mão).

Observação (Spin & Go winner-takes-all):
- ChipEV é um ótimo proxy e tende a funcionar bem, mas **não é matematicamente idêntico** a PrizeEV quando:
  - o torneio tem dinâmica de blinds, stacks evoluem, e o “valor marginal” de fichas muda com a fase.
- Caminho recomendado:
  1) primeiro estabilizar um **bom ChipEV** (rápido e menos variância),
  2) depois (se necessário) adicionar um modo PrizeEV/torneio ou uma aproximação (ex.: value function por stack-state).

### 4.9 Regras de “quase all-in”

Converter para ALL_IN **só** quando:
- a ação deixaria “restinho” muito pequeno (ex.: **<10%** do stack efetivo), ou
- a ação já é praticamente all-in (**>90%** do efetivo).

Isso reduz “rugosidade” no espaço de ações sem distorcer demais.

### 4.10 Performance: gargalo atual

O gargalo é o **ADV collect**, não o treino:
- ADV collect típico: ~2.0k–2.6k samples/s (dependendo de traversals/episodes).
- Train adv/pol: ~1–2s por iteração (muito pequeno comparado ao collect).

### 4.11 Otimização principal: clone() vs replay

- Foi ativado **PokerGame.clone()** no C++ e usado no traversal para evitar replay O(N²).
- Log “CLONE ENABLED” existe para confirmar clone.

### 4.12 Segurança do C++ (bug crítico evitado)

- Evitar ponteiros instáveis atravessando C++↔Python (ex.: `Dealer*` dentro de `Round`).
- Round deve permanecer “stateless” (sem ponteiro para objeto externo) para evitar crash/illegal action após horas/dias.

### 4.13 Workers: não recriar redes a cada task

- `rollout_workers.py` deve manter redes em memória global por processo:
  - inicializa uma vez no init do worker,
  - em cada task, carrega apenas `state_dict`.

---

### 4.13 Checkpoint e atomic save (Windows)

- Salvamento usa `.tmp` e renomeia para `.npz`.
- Se aparecer erro do tipo:
  - `WinError 2 ... adv_p0.npz.tmp -> adv_p0.npz`
  isso normalmente indica que o `.tmp` não chegou a ser criado (interrupção no meio do save, path faltando, ou concorrência).

Regras:
- Diretórios `checkpoints/` e `checkpoints/buffers/` devem existir antes de salvar.
- No CTRL+C, o script deve:
  - **não fazer merge parcial**,
  - cancelar futures,
  - salvar uma vez em estado consistente.

---

## 5) Convenções (nomes, logs, formatos)

### 5.1 Seeds e determinismo
- Existe `base_seed` global (do trainer) e `ep_seed` por episódio/mão.
- Validações devem checar consistência: mesma seed -> mesma sequência.

### 5.2 Logs de treino
Os logs precisam mostrar (no mínimo):
- iteração atual e tempo
- taxa de geração de samples (samples/seg)
- tamanho dos buffers por player
- loss médio por rede (advantage nets) e, quando existir, average policy net
- estatísticas de ação (freq por ação, %fold, %all-in, etc)
- checks de sanidade (NaN/Inf, grads exploding, etc)

### 5.3 Checkpoints
- Checkpoints devem registrar:
  - iteração
  - pesos das redes
  - otimizadores
  - buffers (ou path/estado deles)
  - seeds/estado RNG
  - OBS_DIM esperado (para recusar carregar quando incompatível)
  - Preferir escrita atômica, salvar em arquivo temporário, depois rename.
  - `checkpoint_last.pt` deve ser o “padrão de retomada”, `checkpoint.pt` pode ser snapshot periódico.
  - Para o eval rodar enquanto o treino salva, manter backups, por exemplo `checkpoint_last_backup_YYYYMMDD_HHMMSS.pt`.


### 5.4 Formatos
- Obs é `np.float32` contíguo.
- `legal_mask` é `np.float32` contíguo.
- Ações são `int` 0..6.

## 5.5 Logs: o que significa e como julgar “saudável”

Linha exemplo:

```
iter=25 dt_iter=... ADV: collect=... (sps) train=... bs=...
POL: collect=... (sps) train=... bs=...
adv_added=[...] pol_added=[...] adv_loss=[...] pol_loss=...
actions p0: ... | p1: ... | p2: ...
```

Checks:
- `pol_loss` deve ser **finito** (evitar NaN).
- `actions` não deve colapsar para uma ação única por longos períodos (um pouco de viés no começo é normal).
- `adv_added/pol_added` devem ser >0 com frequência, e coerentes com o batch.
- `dt_iter` deve escalar aproximadamente com `traversals/episodes`.

---

## 6) Problemas/atenções atuais (para não esquecer)
	1) viés forte para fold, vide last_log.txt
	2) Quando “todo mundo está all-in”, advance_stage_if_needed() força stage_ = SHOWDOWN e finaliza, mas não completa as 5 cartas comunitárias.
	3) O showdown então avalia a mão com board incompleto, e o seu eval7_best() não faz equity de runout, ele só avalia o “melhor 5-card possível com as cartas que existem agora”. Isso destrói o EV real de all-in, draws, blefes e raises, e frequentemente empurra a política para “não colocar ficha”, ou seja, fold/call passivo.
	4) As ações de raise/bet por fração do pote são tratadas de um jeito que tende a ficar incoerente quando existe diff > 0 (aposta a pagar). Se raise não inclui corretamente o call embutido, ou se o legal action não checa diff + raise_add, isso bagunça custo real das ações e também empurra para fold.
	5) fold alto quando to_call == 0 (ninguém deveria foldar de graça, nunca!)
	6) desaparecimento quase total de bets/raises ao longo das iterações.

## 7) Próximos upgrades planejados (curto prazo)
  1) Corrigir os problemas descritos no item 6.	
  2) Sanity check do motor quando all-in acontece
		Rode 10.000 mãos random e logue: quantas vezes terminou em showdown com len(public_cards) < 5.
		Se isso acontecer, a dinâmica está errada e o treino vai enviesar.
  3) payoff consistente (chips conservation / payoff sanity), sem perder performance e treinando mão avulsas, não torneio completo.

## 8) Próximos upgrades planejados (longo prazo)
	1) Poder jogar torneio SpinGo contra dois bots via CMD (aqui o foco será verificar na prática se tudo está ocorrendo conforme o esperado)
	2) Integrar o bot ao OpenHoldem via user.dll para que o bot seja capaz de jogar na interface do Clube de Poker da Faculdade.
	

## 9) Arquivos “fonte da verdade” usados na última correção

- disponíveis no repositório público https://github.com/pmartins87/deep_spin

Atualizado em 2026-02-14 14:07:12.

