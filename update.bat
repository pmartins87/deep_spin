@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM update.bat
REM - Por padrao: sincroniza APENAS arquivos ja rastreados (tracked)
REM - Se houver novos arquivos (untracked) nao-ignorados, pergunta se quer versionar
REM - Comando extra: update.bat add caminho\arquivo "mensagem"
REM - Mensagem (sem add): update.bat "mensagem"
REM ============================================================

cd /d "%~dp0" || exit /b 1

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ERRO: esta pasta nao parece ser um repositorio Git.' -ForegroundColor Red"
  exit /b 1
)

REM Ignoraveis padrao (somente LOCAL, nao mexe no .gitignore do repo)
call :ensure_exclude "venv/"
call :ensure_exclude "runs/"
call :ensure_exclude "logs/"
call :ensure_exclude "build/"
call :ensure_exclude "dist/"
call :ensure_exclude "checkpoints/"
call :ensure_exclude "*.pyd"
call :ensure_exclude "__pycache__/"
call :ensure_exclude "*.pyc"
call :ensure_exclude "*.log"

REM -------------------------
REM Parse argumentos
REM -------------------------
set "MODE=%~1"
set "ADD_PATH="
set "MSG="

if /I "%MODE%"=="add" (
  set "ADD_PATH=%~2"
  set "MSG=%~3"
  if "%ADD_PATH%"=="" (
    powershell -NoProfile -Command "Write-Host 'ERRO: use: update.bat add caminho\arquivo \"mensagem\"' -ForegroundColor Red"
    exit /b 1
  )
) else (
  REM Mensagem simples: primeiro argumento (use aspas se tiver espacos)
  set "MSG=%~1"
)

REM Se MSG nao foi fornecida, cria automática
if not defined MSG (
  for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmmss"') do set "MSG=auto: update %%I"
)

powershell -NoProfile -Command "Write-Host '[SYNC] pull --rebase --autostash...' -ForegroundColor Cyan"
git pull --rebase --autostash
if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ERRO: pull --rebase falhou. Resolva conflitos e rode novamente.' -ForegroundColor Red"
  exit /b 1
)

REM Stage somente TRACKED
powershell -NoProfile -Command "Write-Host '[STAGE] tracked (arquivos ja versionados)...' -ForegroundColor Cyan"
git add -u
if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ERRO: git add -u falhou.' -ForegroundColor Red"
  exit /b 1
)

REM Se modo add: adiciona um caminho específico para virar TRACKED
if defined ADD_PATH (
  powershell -NoProfile -Command "Write-Host ('[ADD] ' + '%ADD_PATH%') -ForegroundColor Yellow"
  git add -- "%ADD_PATH%"
  if errorlevel 1 (
    powershell -NoProfile -Command "Write-Host 'ERRO: nao consegui adicionar esse caminho. Confira o nome/caminho.' -ForegroundColor Red"
    exit /b 1
  )
)

REM Conta untracked nao-ignorados e ignorados
for /f %%C in ('powershell -NoProfile -Command "(git ls-files --others --exclude-standard ^| Measure-Object).Count"') do set "UNTRACKED=%%C"
for /f %%C in ('powershell -NoProfile -Command "(git ls-files --others -i --exclude-standard ^| Measure-Object).Count"') do set "IGNORED=%%C"

powershell -NoProfile -Command "Write-Host ('[INFO] novos (untracked): %UNTRACKED% , ignorados: %IGNORED%') -ForegroundColor DarkYellow"

REM Se houver novos arquivos untracked (e nao ignorados), pergunta se quer versionar TODOS
if not "%UNTRACKED%"=="0" (
  powershell -NoProfile -Command "Write-Host '[NOVOS] Existem arquivos/pastas no PC que NAO estao no Git:' -ForegroundColor Yellow"
  powershell -NoProfile -Command "git ls-files --others --exclude-standard | Select-Object -First 40 | ForEach-Object { '  ' + $_ }"
  powershell -NoProfile -Command "Write-Host 'Quer passar a VERSIONAR (track) TODOS eles agora? (S/N)' -ForegroundColor Yellow"
  choice /c SN /n >nul
  if errorlevel 2 (
    powershell -NoProfile -Command "Write-Host 'OK: mantendo os novos fora do Git.' -ForegroundColor DarkYellow"
  ) else (
    powershell -NoProfile -Command "Write-Host '[ADD] adicionando novos (git add -A)...' -ForegroundColor Yellow"
    git add -A
    if errorlevel 1 (
      powershell -NoProfile -Command "Write-Host 'ERRO: git add -A falhou.' -ForegroundColor Red"
      exit /b 1
    )
  )
)

REM Se houver DELECOES staged, pede confirmacao
git diff --cached --name-status | findstr /b "D " >nul 2>&1
if not errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ATENCAO: ha delecoes prontas para commit:' -ForegroundColor Red"
  git diff --cached --name-status | findstr /b "D "
  powershell -NoProfile -Command "Write-Host 'Continuar mesmo assim? (S/N)' -ForegroundColor Red"
  choice /c SN /n >nul
  if errorlevel 2 (
    powershell -NoProfile -Command "Write-Host 'ABORTADO: nada foi commitado.' -ForegroundColor Yellow"
    exit /b 1
  )
)

REM Se nada staged, sai
git diff --cached --quiet --exit-code
if not errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'OK: nada para commitar.' -ForegroundColor Green"
  exit /b 0
)

powershell -NoProfile -Command "Write-Host ('[COMMIT] ' + '%MSG%') -ForegroundColor Cyan"
git commit -m "%MSG%"
if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ERRO: commit falhou.' -ForegroundColor Red"
  exit /b 1
)

for /f "delims=" %%B in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%B"

powershell -NoProfile -Command "Write-Host ('[PUSH] origin ' + '%BRANCH%') -ForegroundColor Cyan"
git push origin "%BRANCH%"
if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host 'ERRO: push falhou.' -ForegroundColor Red"
  exit /b 1
)

powershell -NoProfile -Command "Write-Host 'OK: sincronizado (GitHub <-> PC via pull+push).' -ForegroundColor Green"
exit /b 0

:ensure_exclude
set "PAT=%~1"
if not exist ".git\info" mkdir ".git\info" >nul 2>&1
if not exist ".git\info\exclude" type nul > ".git\info\exclude"
findstr /x /c:"%PAT%" ".git\info\exclude" >nul 2>&1
if errorlevel 1 (
  >>".git\info\exclude" echo %PAT%
)
exit /b 0
