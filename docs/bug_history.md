//DeepSpin v60

# Bug History

## B000 – ^M (CR) aparecendo em linhas do Python no git diff/show
- Sintoma: linhas exibindo `^M` no `git show`, diffs “sujos”.
- Causa: normalização/edição com EOL inconsistente (core.autocrlf true e/ou CR inserido).
- Correção:
  - Ajuste local: `core.autocrlf=false`, `core.eol=lf`.
  - Normalização: `git add --renormalize .` e commit.
  - Prevenção: `.gitattributes` forçando LF (exceto `.bat`).
- Status: resolvido.

## B001 (CRÍTICO) - Showdown com board incompleto quando todos estão ALL-IN
- Arquivo: env/poker_env.cpp
- Local: PokerGame::is_over()
- Problema: se ninguém pode agir, a mão pode terminar sem runout do board (5 cartas), gerando payoffs errados.
- Fix: em is_over(), completar public_cards_ até 5 cartas quando any_can_act == false e fechar round_counter_ = 4.

## B002 - Min-raise removido indevidamente em get_nolimit_legal_actions
- Arquivo: env/poker_env.cpp
- Local: Round::get_nolimit_legal_actions()
- Problema: condição usa raise_add <= diff, o que pode eliminar raise mínimo válido.
- Fix: trocar para raise_add < diff.

## B003 - (NOT A BUG!) Effective stack e SPR usando stack total ao invés de stack behind
- Arquivo: env/poker_env.cpp
- Local: PokerGame::get_state()
- Problema: eff_stack_bb calculado com (remained + in_chips) distorce features pós-aposta.
- Fix: Não foi considerado bug, é o comportamento esperado! Digamos que o jogador no bigblind tem 3bbs, 
1bb ele colocou em jogo, outros 2bb ficarão pra trás, se eu for all-in contra ele, vou perder 3bbs. 
Então o Stack Effetivo é quanto eu posso perder. Conta a aposta atual e o stack pra trás.

## B004 (CRÍTICO) - sanity_check usa índices errados do obs para calcular diff (bets)
- Arquivo: scripts/sanity_check.py
- Problema: bets_bb = obs[107:110] não representa bets, gera diff incorreto e validações enganosas.
- Fix: calcular diff via raw_obs["in_chips"] e ativar set_debug_raw_obs(True).

## B006 (CRÍTICO), train_deepcfr.py pode exceder traversals ou episodes quando workers > budget
- Arquivo: scripts/train_deepcfr.py
- Problema: per = max(1, total//workers) força no mínimo 1 por worker, excede total quando total < workers.
- Fix: usar divisão exata per=total//w, rem=total%w e pular tasks com bud/ep <= 0.

## B005 (ALTA) - traversal.py inconsistências (OBS_DIM fallback, type hint, keyword errado)
- Arquivo: deepcfr/traversal.py
- Problemas: fallback 338, List[int] sem import, EpisodeSpec chamado com is_hu (keyword inválido).
- Fix: fallback 292, usar list[int], trocar is_hu -> game_is_hu.

## B007 (ALTA), trainer.py usa current_player que pode não existir, encerra policy cedo
- Arquivo: deepcfr/trainer.py
- Fix: fallback para game_pointer, pid = state.get("current_player", p)

## B008 (CRÍTICO), PolicyNet.masked_cross_entropy normalizava por número de ações legais
- Arquivo: deepcfr/networks.py
- Problema: loss dividia por legal.sum(), distorcendo o gradiente por estado e enviesando treino.
- Fix: usar cross entropy padrão, loss = -(target * log(probs)).sum(dim=-1).mean()
- Verificação: scripts/verify_policy_mask.py

## B009 (CRÍTICO), colisões de ep_seed gerando muitas mãos repetidas por iteração
- Arquivo: deepcfr/rollout_workers.py
- Problema: ep_seed usava rng.integers(0, 1_000_000). Com dezenas de milhares de episódios/traversals, ocorre birthday problem e muitas colisões.
- Impacto: repete mãos, reduz diversidade, enviesando treino silenciosamente.
- Fix: ep_seed determinístico e único por task + índice local.
- Verificação: scripts/verify_epseed_uniqueness.py

## B010 (ALTA), trainer.py ainda gerava ep_seed com rng pequeno, repetindo mãos
- Arquivo: deepcfr/trainer.py
- Fix: ep_seed determinístico e único, igual ao pipeline paralelo.

## B011 (HARDENING), sanitização do target da Policy no merge
- Arquivo: scripts/train_deepcfr.py
- Fix: zera ilegais, renormaliza, remove NaN/inf antes de gravar no buffer.
- Verificação: scripts/verify_buffer_targets.py

## B012 (HARDENING), monitoramento de escala do sinal de treino (ADV e POL)
- Arquivo: scripts/train_deepcfr.py
- Mudança: logs de percentis do abs regret target (BB) e entropia da policy, mais detecção de NaN ou inf.
- Script: scripts/verify_signal_scale.py

## B013 (HARDENING), monitoramento de suporte de ações no buffer de policy
- Arquivo: scripts/train_deepcfr.py
- Mudança: logs por ação, frequência de legalidade e mean_prob do target quando legal.
- Script: scripts/verify_action_support.py
- Objetivo: detectar ações “mortas” e composição de dataset enviesada (ex: só CHECK/CALL).

## B014 (VERIFICAÇÃO), semântica de ações legais (CHECK/CALL/RAISE) validada via clone+step
- Script: scripts/verify_legal_semantics.py
- Objetivo: detectar bugs raros onde CHECK aparece com diff>0, falta CHECK com diff==0, raises que não aumentam max(in_chips), ou ausência de ação que paga quando diff>0.

## B015 (VERIFICAÇÃO), auditoria de composição do dataset (street, HU/3-way, diff0)
- Script: scripts/verify_dataset_composition.py
- Objetivo: detectar dominância de pré-flop/diff0 e falta de turn/river, que prejudica aprendizado sem crash.

## B016 (CRÍTICO), min-raise inexistente permitia micro-raises e inflava pré-flop no buffer
- Arquivo: env/poker_env.h/.cpp
- Problema: get_nolimit_legal_actions não aplicava min-raise corretamente (não rastreava last_raise_size, não validava mx==0).
- Impacto: loops de micro-raises, muitos mais pontos de decisão no pré-flop, buffer de policy dominado por pré-flop.
- Fix: adicionar last_raise_size, inicializar a cada betting round, exigir raise_add >= min_raise e atualizar last_raise_size quando ocorrer raise (incluindo ALL_IN com raise).
- Verificação: scripts/verify_legal_semantics.py e scripts/verify_dataset_composition.py

## B017 (ALTA), Mini raise vs isolation era permitido
- Arquivo: env/poker_env.cpp
- Motivo: evitar raises que gerassem ramificações artificiais.
- Mudança: Raise vs isolation é sempre all-in.
- Impacto esperado: mais shoves em spots onde apenas aumentar não faz muito sentido, menos linhas ruins como abstração.

## B018 (ALTA), near-all-in collapse ajustado para 60%
- Arquivo: env/poker_env.cpp
- Motivo: evitar raises que comprometem grande parte do stack e deixam “migalhas” para trás, reduzindo ramificações artificiais.
- Mudança: threshold de 90% para 60% (e leftover equivalente).
- Impacto esperado: mais shoves em spots de commit alto, menos linhas ruins como abstração.

## B019 (ALTA), salvar checkpoints por tempo (8h) para reduzir overhead de IO
- Arquivo: scripts/train_deepcfr.py
- Mudança: save_every=0 (desativa por iteração) e novo save_every_seconds=8h.
- Mantém: save no Ctrl+C.

## B020 (CRÍTICO), pré-flop permitia FOLD com to_call == 0 (fold grátis ilegal)
- Arquivo: env/poker_env.cpp
- Local: PokerGame::get_legal_actions (stage_==0)
- Problema: lista base incluía FOLD e não removia quando to_call_chips == 0.
- Impacto: reduz reach de flop, aumenta término pré-flop, distorce dataset e treino.
- Fix: remover FOLD quando to_call_chips == 0.
- Verificação: scripts/verify_legal_semantics.py, scripts/verify_hand_reach_rates.py, scripts/verify_dataset_composition.py

## B021 (RECUSADO) ALL_IN sempre legal” inflaciona morte pré-flop nas iterações iniciais
- Sugestão: Open shove / shove em pot limped só quando effective stack ≤ 12bb
- Decisão: Eu descarto completamente a aplicação da B021! Há muitos all-ins com stacks de 20bb+ que devem ser feitos! 
Não quero enviesar as decisões da IA, quero fornecer um cenário sem falhar, que corresponda ao jogo real, para que a IA aprenda o jogo sem interferências diretas na estratégia como foi essa sugestão da B021.

## B022 (VERIFICAÇÃO) tornar verify_preflop_end_causes.py capaz de achar a config do ScenarioSampler automaticamente

## B023/B024, expor ENV_CONFIG no train_deepcfr.py para verificadores e consistência
- Arquivo: scripts/train_deepcfr.py
- Problema: config do ScenarioSampler estava inline e não acessível para scripts de verificação.
- Fix: criar ENV_CONFIG global e usar ENV_CONFIG ao instanciar ScenarioSampler.
- Impacto: nenhum no comportamento, apenas melhora integridade e auditabilidade.

## B034 (ALTA), warm-up sem gravar buffers para evitar dataset inicial aleatório
- Arquivo: scripts/train_deepcfr.py
- Problema: no início a policy é quase uniforme e ALL_IN é sempre legal, resultando em shove-fest e dataset contaminado.
- Fix: warmup_iters (default 10) onde coletamos/treinamos, mas não gravamos nos buffers nem salvamos.
- Verificação: scripts/verify_preflop_policy_mass.py e scripts/verify_dataset_composition.py

## B036 (ALTA), warm-up anterior bloqueava merge e impedia treino (buffers vazios -> loss nan)
- Arquivo: scripts/train_deepcfr.py
- Problema: warm-up bloqueava gravação nos buffers, então não havia batch para treinar.
- Fix: durante warm-up treina normalmente, e ao final do warm-up limpa buffers (ReservoirBuffer.clear()).
- Verificação: scripts/verify_preflop_policy_mass.py e scripts/verify_dataset_composition.py.

## B037 (ALTA), Ctrl+C gerava spam de KeyboardInterrupt nos workers durante save
- Arquivo: scripts/train_deepcfr.py
- Fix: pool.shutdown() antes de save_all() no finally.

## B039 (ALTA), thresholds do verify_dataset_composition ajustados para "pontos de decisão"
- Arquivo: scripts/verify_dataset_composition.py
- Motivo: river/turn por decisão é naturalmente baixo em Spin; alertas antigos geravam falso positivo.
- Fix: thresholds mais realistas (river<0.2%, turn<1%, pre>95%).

## B040 (ALTA), Ctrl+C cancela pool imediatamente para evitar spam de KeyboardInterrupt dos workers
- Arquivo: scripts/train_deepcfr.py
- Fix: no KeyboardInterrupt, cancelar futures e shutdown(wait=False) antes de re-raise.