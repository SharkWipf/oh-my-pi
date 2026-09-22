# @oh-my-pi/pi-coding-agent

Core implementation package for the `omp` coding agent in the `oh-my-pi` monorepo.

For installation, setup, provider configuration, model roles, slash commands, and full CLI reference, see:
- [Monorepo README (local)](../../README.md)
- [Monorepo README (GitHub)](https://github.com/can1357/oh-my-pi#readme)

Package-specific references:
- [CHANGELOG](./CHANGELOG.md)
- [MCP configuration guide](../../docs/mcp-config.md)
- [MCP runtime lifecycle](../../docs/mcp-runtime-lifecycle.md)
- [MCP server/tool authoring](../../docs/mcp-server-tool-authoring.md)
- [DEVELOPMENT](./DEVELOPMENT.md)

## Transient inline image accounting

For a prepared provider `Context`, `getInlineFrameAccounting(image)` and `getInlineTextAccounting(textBlock)` from `@oh-my-pi/pi-coding-agent/session/snapcompact-inline` identify actual inline-rendered frames and their control notes. Owners are `system`, `context` (loaded context instructions), or `tool`; tool facts also carry `toolCallId`. Frame `estimatedTokens` comes from the rendering shape and is a local estimate, not exact provider billing. Ordinary original images have no inline fact.

Count the transformed system-prompt stub and emitted text, not the replaced prompt. When starting from `Tokenizer.countMessages`, replace each inline image baseline with its frame estimate by adding only `estimatedTokens - IMAGE_TOKEN_ESTIMATE`. Irreducible prompt accounting includes only the `system`/`context` frames and notes, not ordinary history or tool-result frames.

Facts use the existing source-origin sidecar and survive explicit image-normalization and blob-decoration clones. Arbitrary untracked clones, changed image/text fields, invalidated hook output, and persisted/reloaded origin maps do not provide current inline facts. Inspect the actual prepared pre-hook context; do not infer ownership from equal text or treat missing facts as a known zero cost.

## Memory backends

The agent supports three mutually-exclusive memory backends, selected via the `memory.backend` setting (Settings → Memory tab, or `~/.omp/config.yml`):

- `off` (default) — no memory subsystem runs.
- `local` — existing rollout-summarisation pipeline; writes `memory_summary.md` and consolidated artifacts under the agent dir.
- `hindsight` — talks to a [Hindsight](https://hindsight.vectorize.io) server (Cloud or self-hosted Docker), retains transcripts every Nth user turn, recalls memories on the first turn of a session, and exposes `retain`, `recall`, and `reflect`.

### Hindsight quickstart

1. Run a Hindsight server (Cloud or `docker run -p 8888:8888 ghcr.io/vectorize-io/hindsight:latest`).
2. Set `memory.backend = "hindsight"` and `hindsight.apiUrl = "http://localhost:8888"` (or your Cloud URL).
3. Optional environment overrides (env wins over settings):
   - `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN` — connection
   - `HINDSIGHT_BANK_ID`, `HINDSIGHT_DYNAMIC_BANK_ID`, `HINDSIGHT_AGENT_NAME` — bank addressing
   - `HINDSIGHT_AUTO_RECALL`, `HINDSIGHT_AUTO_RETAIN`, `HINDSIGHT_RETAIN_MODE` — lifecycle
   - `HINDSIGHT_RECALL_BUDGET`, `HINDSIGHT_RECALL_MAX_TOKENS` — recall sizing
   - `HINDSIGHT_BANK_MISSION`, `HINDSIGHT_DEBUG`

Switching backends mid-session immediately replaces the live backend, memory tools, listeners, and system-prompt context. Existing users with `memories.enabled = true|false` are migrated to `memory.backend = "local"|"off"` exactly once on first launch; afterward, `memory.backend` is the sole runtime selector.
