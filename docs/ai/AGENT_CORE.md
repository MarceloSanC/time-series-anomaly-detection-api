# AGENT_CORE.md

Instrucoes obrigatorias para qualquer agente que trabalhe neste repositorio.

## Ordem de leitura (inicio de sessao)

1. `docs/context/openapi_spec.yaml`
2. `docs/ai/skills.md`
3. `docs/project/ROADMAP.md`
4. `docs/ai/LLM_USAGE.md`
5. `docs/project/ARCHITECTURE.md`
6. `docs/process/GIT_PROTOCOL.md`

## Regras operacionais

- Seguir estritamente o MVP antes de qualquer enhancement.
- Contrato HTTP (rotas, requests e responses) deve seguir `docs/context/openapi_spec.yaml`.
- Respeitar boundaries de camadas: `domain`, `services`, `repository`, `api`, `utils`.
- Toda configuracao tunavel deve sair de `.env` via `app/config.py`.
- Sempre trabalhar com branch por feature e merge para `main` apenas quando funcional.

## Versionamento e qualidade

- Nomes de branch e commits devem seguir `docs/process/GIT_PROTOCOL.md`.
- Qualquer arquivo gerado deve passar por review com o protocolo de `docs/ai/LLM_USAGE.md`.
