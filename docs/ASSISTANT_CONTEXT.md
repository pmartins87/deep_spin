# Deep Spin, Índice e Regras do Projeto

Este arquivo é um ÍNDICE prático + regras de integridade.
Documentação detalhada fica em:
- `docs/MANUAL.md` (manual completo de uso e explicação técnica)
- `docs/PROJECT_LOG.md` (versionamento, bugs, roadmap)

Repositório (fonte da verdade):
- https://github.com/pmartins87/deep_spin

---

## Comandos essenciais

### Build do motor C++ (cpoker)
Dentro de `env/`:
- `python setup.py build_ext --inplace`

### Sanity check (rodar sempre antes de treino longo)
No root:
- `venv\Scripts\python.exe -u scripts\sanity_check.py --episodes 5000`

Esperado:
- showdown com board<5: 0
- violações affordability: 0
- pot/stack negativos: 0

### Treino (exemplo)
- `venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0`

### Debug de uma mão (hand history estilo PT4)
- `venv\Scripts\python.exe -u scripts\hand_history_debug.py --seed 123 --policy random`

---

## Regras de integridade (não negociar)

1) Resume confiável
- Ctrl+C salva checkpoint consistente e sai sem stacktrace
- deterministic_merge=1: merge determinístico, sem depender da ordem de futures

2) Sem “simplificações invisíveis”
- tabelas/frequências devem permanecer completas e auditáveis
- mudanças grandes devem vir com sanity checks

3) Motor C++ seguro e consistente
- ações legais compatíveis com custo real (diff + raise_add)
- pot/stack nunca negativos
- side pots corretos

---

## Estado atual (baseline)

- Release: “DeepSpin R1, Baseline Sanity-Passed”
- sanity_check passa sem violações (affordability + negativos)
- correções críticas aplicadas:
  - raise diff+raise_add
  - RNG do scenario (numpy Generator + checkpoint robusto)
  - shuffle slice bug removido

Para detalhes do histórico: `docs/PROJECT_LOG.md`.
Para detalhes completos do sistema: `docs/MANUAL.md`.
