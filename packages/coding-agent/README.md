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

## Source preservation policy API

`session/preserved-messages` exposes a branch-local `PreservedMessageQuery`. Build it cooperatively from the active journal branch and a currentness predicate; it ignores entries before the latest clear boundary. Window-only settings changes query compact count/price aggregates without retokenizing source history. Rows and selected candidate messages are materialized on demand.

Use `appendEntries` for an already validated same-branch journal suffix rather than collecting the whole branch after every send. Reset or divergent ancestry requires a new cooperative build. `getManualGroup`/`getManualGroups` expose complete atom members and existing override journal IDs for reset revalidation; `invalidateClassifications` clears changed-input facts before fresh decisions are published. `readPreservedUserMessageClassificationMasks` provides the same real-user/post-clear fact interpretation for asynchronous backfill without pricing history. Selection memberships and aggregate totals are lazy; explicitly materialize the required candidates at the owning operation snapshot boundary before awaiting method work.

First/recent/hard-recent limits independently accept Off, All, message count, tokens, or a percentage of the effective model maximum context. Finite token/percentage zero is not Off. Heuristics only remove candidates; ordinary regex, eleven-category policy, Final regex, then manual state resolve in order. Auto is neutral and Keep wins within a stage. Hard-recent temporarily bypasses Never and pruning. Manual Always also bypasses pruning, but every Always source shares the configured linked cap.

When no effective model maximum is available, percentage-derived windows are reported in `selection.unavailableLimits`; the result is a provisional preview, not a finite-zero cap or complete compaction input. Other configured units and manual/source inspection still work. `selection.blockers` names the actual source that stopped each finite edge. Non-Auto reset enumeration captures its source boundary when the iterator is requested and avoids a synchronous all-history sort.

The result separates selected users `P`, complete admitted non-user atoms `N`, and temporary hard-recent `H`. Quota estimates are determined from source content before ordinary allocation, including the base estimate for original images. Methods must leave the ordinary user cut unchanged and precharge complete `N` once inside their own allocator using `prechargeNonUsers`; physical representation overlap never refunds source quota. Candidate spans use durable source IDs and UTF-16 text intervals. Installed compaction bytes and provider accounting remain method/lifecycle-owned.

Visible user-attributed `custom_message` journal entries support manual state as non-user `N` sources. Their current durable content is normalized through the existing custom-message helper while retaining the journal ID; they never enter automatic user windows, hard-recent, or classification. Hidden and agent-attributed custom injections do not become manual user rows.

Saved manual overrides and successful eleven-bit classification metadata keep their established v1 codecs. Settings compose layers before interpreting legacy paired message-zero limits; opening settings does not write normalized values.
