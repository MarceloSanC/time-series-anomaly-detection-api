# GIT_PROTOCOL.md — Protocolo de Versionamento

Este documento define o padrão oficial de versionamento para o projeto **Time Series Anomaly Detection API**.

## 1. Estratégia de Branches

- `main`: sempre estável e pronta para uso.
- Desenvolvimento começa com o **scaffolding base** diretamente na `main`.
- Cada nova feature deve ser desenvolvida em uma branch dedicada.
- Só fazer merge para `main` quando a feature estiver funcional e validada.

## 2. Regra de Fluxo

Para cada feature:

1. Atualizar `main` local.
2. Criar branch de feature.
3. Implementar a feature com commits pequenos e descritivos.
4. Executar validações (testes/lint aplicáveis).
5. Fazer merge da feature em `main`.
6. Excluir branch após merge.

## 3. Padrão de Nome de Branch

Formato obrigatório:

`<tipo>/<dia>-<escopo-curto>`

Tipos permitidos:

- `feature`: nova funcionalidade
- `fix`: correção de bug
- `refactor`: refatoração sem mudança de comportamento
- `test`: criação/ajuste de testes
- `docs`: documentação
- `chore`: manutenção técnica

Exemplos:

- `feature/day1-domain-schemas`
- `feature/day1-model-repository`
- `feature/day1-model-service`
- `fix/day2-train-endpoint-error-handling`
- `docs/day5-readme-final`

## 4. Padrão de Mensagens de Commit

Formato obrigatório:

`<tipo>(<escopo>): <resumo no imperativo>`

Tipos de commit:

- `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

Exemplos:

- `feat(domain): add pydantic schemas for time series payloads`
- `feat(repository): implement atomic index write with os.replace`
- `feat(service): add model training and version resolution`
- `fix(service): handle missing version with explicit exception`
- `test(repository): add save-load roundtrip coverage`
- `docs(roadmap): document day 1 merge sequence`

Boas práticas:

- Um commit deve representar uma mudança lógica única.
- Evitar commits genéricos como `update` ou `ajustes`.
- Mensagens curtas (ideal até 72 caracteres no resumo).
- Se necessário, usar corpo explicativo após uma linha em branco.

## 5. Regra de Merge

- Preferência: `--no-ff` para preservar histórico da feature.
- Alternativa: squash merge apenas quando houver muitos commits ruidosos.
- Não fazer merge com testes quebrando.

Comando recomendado:

```bash
git checkout main
git merge --no-ff <branch-feature>
```

## 6. Sequência Recomendada para Day 1

Após scaffolding inicial na `main`, seguir branches por feature na ordem do roadmap:

1. `feature/day1-config`
2. `feature/day1-domain`
3. `feature/day1-repository`
4. `feature/day1-services`
5. `feature/day1-unit-tests`

## 7. Checklist Antes do Merge

- Feature funcional conforme `docs/ai/skills.md` e `docs/project/ROADMAP.md`
- Sem violação de camadas (`domain`, `services`, `repository`, `api`)
- Tipagem aplicada nas assinaturas públicas
- Testes relevantes passando
- Commit messages no padrão definido

## 8. Convenção de Tags

- Releases estáveis: `vMAJOR.MINOR.PATCH`
- Exemplo inicial de entrega final: `v1.0.0`

