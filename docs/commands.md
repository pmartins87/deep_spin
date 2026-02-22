## Comandos essenciais

# Ativar venv
cd C:\PokerAI\spin_deepcfr
call venv\Scripts\activate.bat

# Compilar
cd env
..\venv\Scripts\python.exe setup.py build_ext --inplace
cd ..

## Rodar train_deepcfr.py:

# Modo “mais confiável” (bitwise, mais lento)
venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 1

# Modo “produção CPU-only” (bem mais rápido, ainda com merge determinístico)
cd C:\PokerAI\spin_deepcfr
venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0

# Modo Agressivo:
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
cd C:\PokerAI\spin_deepcfr
venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0

# play_cli_vs_checkpoint.py:
venv\Scripts\python.exe -u scripts\play_cli_vs_checkpoint.py --checkpoint checkpoints\checkpoint.pt --human 0

# find_debug_examples.py
venv\Scripts\python.exe -u scripts\find_debug_examples.py --seed 7 --max_hands 200000

# sanity_check.py
venv\Scripts\python.exe -u scripts\sanity_check.py --episodes 5000

# scripts/verify_budgets.py
python scripts/verify_budgets.py --workers 24 --traversals 10 --episodes 10

# scripts/verify_policy_mask.py
python scripts/verify_policy_mask.py

# verify_epseed_uniqueness.py
python scripts/verify_epseed_uniqueness.py --n 50000

# python scripts/verify_scenarios.py
python scripts/verify_scenarios.py

# python scripts/verify_env_invariants.py
python scripts/verify_env_invariants.py

# python scripts/verify_buffer_targets.py
python scripts/verify_buffer_targets.py

# python scripts/verify_fold_opportunities.py
python scripts/verify_fold_opportunities.py

# python scripts/verify_signal_scale.py
python scripts/verify_signal_scale.py

# python scripts/verify_action_support.py
python scripts/verify_action_support.py

# python scripts/verify_legal_semantics.py
set VERIFY_HANDS=10000
python scripts/verify_legal_semantics.py

# python scripts/verify_dataset_composition.py
python scripts/verify_dataset_composition.py

# python scripts/verify_obs_alignment.py
python scripts/verify_obs_alignment.py

# python scripts/verify_hand_reach_rates.py
python scripts/verify_hand_reach_rates.py

# python scripts/verify_resume_integrity.py
python scripts/verify_resume_integrity.py