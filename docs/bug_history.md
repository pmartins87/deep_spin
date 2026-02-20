//DeepSpin v60

# Bug History

## BUG-001 – ^M (CR) aparecendo em linhas do Python no git diff/show
- Sintoma: linhas exibindo `^M` no `git show`, diffs “sujos”.
- Causa: normalização/edição com EOL inconsistente (core.autocrlf true e/ou CR inserido).
- Correção:
  - Ajuste local: `core.autocrlf=false`, `core.eol=lf`.
  - Normalização: `git add --renormalize .` e commit.
  - Prevenção: `.gitattributes` forçando LF (exceto `.bat`).
- Status: resolvido.