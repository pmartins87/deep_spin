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
cd C:\PokerAI\spin_deepcfr
venv\Scripts\python.exe -u scripts\train_deepcfr.py --workers 24 --worker_threads 1 --main_threads 12 --traversals 16384 --episodes 8192 --deterministic_merge 1 --bitwise 0

# play_cli_vs_checkpoint.py:
venv\Scripts\python.exe -u scripts\play_cli_vs_checkpoint.py --checkpoint checkpoints\checkpoint.pt --human 0

# find_debug_examples.py
venv\Scripts\python.exe -u scripts\find_debug_examples.py --seed 7 --max_hands 200000

# sanity_check.py
venv\Scripts\python.exe -u scripts\sanity_check.py --episodes 5000