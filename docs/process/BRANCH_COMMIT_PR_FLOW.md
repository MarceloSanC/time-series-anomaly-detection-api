# BRANCH, COMMIT E PR — FLUXO ROBUSTO

Guia prático para executar o fluxo completo após um merge.

## Cenário de partida

- merge acabou de ser feito
- você quer começar uma nova task
- não há alterações locais pendentes

## 1) Sincronizar e conferir estado

```bash
git fetch origin --prune
git checkout main
git pull --ff-only origin main
git status
```

Esperado:
- branch atual: `main`
- working tree limpa (`nothing to commit, working tree clean`)

## 2) Deletar branch antiga (local)

```bash
git branch -d <branch_antiga>
```

Se aparecer que a branch não foi mergeada localmente mas você sabe que foi mergeada no GitHub:

```bash
git fetch origin --prune
git branch -d <branch_antiga>
```

Use `-D` somente em último caso (forçado).

## 3) Criar e mudar para a branch nova

Padrão: `<tipo>/<area>-<objetivo>`

```bash
git checkout -b <nova_branch>
```

Exemplos:
- `feature/stage-a-task7-models-tests`
- `fix/models-version-validation`
- `docs/readme-models-endpoints`

## 4) Implementar mudanças e revisar escopo do commit

Verificar arquivos alterados:

```bash
git status --short
```

Ver diff:

```bash
git diff
```

## 5) Commit (com template)

Formato obrigatório:

`<tipo>(<escopo>): <resumo no imperativo>`

Template:

```bash
git add <arquivos_do_escopo>
git commit -m "<tipo>(<escopo>): <resumo no imperativo>"
```

Exemplos:
- `feat(models): add introspection routes for model metadata`
- `test(models): add integration coverage for /models endpoints`
- `docs(readme): add usage examples for models endpoints`

## 6) Subir branch para o remoto

```bash
git push -u origin <nova_branch>
```

## 7) Preparar PR

Você vai definir o título e a descrição. Use este checklist para montar o texto:

- contexto/objetivo
- mudanças principais (bullet points)
- impacto em contrato/API (se houver)
- validações executadas
- risco conhecido (se houver)

### Template de descrição de PR

```md
## Summary
- ...
- ...

## Changes
- ...
- ...

## Validation
- ...
- ...

## Notes
- ...
```

## 8) Criar PR

Opção A (GitHub Web):
- abra a URL sugerida pelo `git push`:
  - `https://github.com/<owner>/<repo>/pull/new/<nova_branch>`

Opção B (CLI `gh`):

```bash
gh pr create --base main --head <nova_branch> --title "<titulo>" --body-file <arquivo_descricao.md>
```

## 9) Validar se está pronto para merge

Checklist objetivo:

1. Escopo correto no PR (sem arquivos fora da task)
2. Branch base correta (`main`)
3. CI/checks verdes
4. Testes locais relevantes passaram
5. Sem conflito de merge
6. Sem comentários pendentes bloqueantes

Comandos úteis locais:

```bash
git status --short
git log --oneline -n 5
```

Comandos úteis no GitHub:

```bash
gh pr view --json number,title,state,mergeable,headRefName,baseRefName
gh pr checks
```

## 10) Pós-merge (limpeza)

Depois do merge no GitHub:

```bash
git checkout main
git pull --ff-only origin main
git branch -d <nova_branch>
git fetch origin --prune
```

---

## Sequência curta (resumo)

```bash
git fetch origin --prune
git checkout main
git pull --ff-only origin main
git branch -d <branch_antiga>
git checkout -b <nova_branch>
# implementar
git add <arquivos>
git commit -m "<tipo>(<escopo>): <mensagem>"
git push -u origin <nova_branch>
# abrir PR
```

