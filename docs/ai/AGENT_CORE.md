# AGENT_CORE.md

Instrucoes obrigatorias para qualquer agente que trabalhe neste repositorio.

## Ordem de leitura (inicio de sessao)

1. `docs/ai/skills.md`
2. `docs/project/ROADMAP.md`
3. `docs/ai/LLM_USAGE.md`
4. `docs/project/ARCHITECTURE.md`
5. `docs/process/GIT_PROTOCOL.md`

## Regras operacionais

- Seguir estritamente o MVP antes de qualquer enhancement.
- Respeitar boundaries de camadas: `domain`, `services`, `repository`, `api`, `utils`.
- Toda configuracao tunavel deve sair de `.env` via `app/config.py`.
- Sempre trabalhar com branch por feature e merge para `main` apenas quando funcional.

## Versionamento e qualidade

- Nomes de branch e commits devem seguir `docs/process/GIT_PROTOCOL.md`.
- Qualquer arquivo gerado deve passar por review com o protocolo de `docs/ai/LLM_USAGE.md`.

