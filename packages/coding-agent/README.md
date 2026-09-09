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

## Goal objective delivery

`goal.injectAsUserMessage` (default `false`) is available under Settings → Tasks → Modes. When enabled, each successful `goal` tool `create` queues only the trimmed objective as an ordinary user follow-up. Budget, status, and progress remain in the existing goal runtime; they are not added to the user message.

Delivery uses the normal session queue: a streaming run consumes the follow-up after its current work, while an idle session follows ordinary queue-drain rules. The message becomes durable when delivered, not when queued; restarting does not reconstruct undelivered objectives from saved goal state. Delivered messages keep their journal identity and tool-producer metadata on reload, with unchanged user-role rendering and billing attribution.

`get`, `resume`, `complete`, `drop`, and direct `/goal` commands do not inject another objective. Turning the setting off affects future creates only; it does not rewrite history or add special goal retention. SDK tool hosts outside `createAgentSession` must provide `ToolSession.sendUserMessage` when enabling this option.

## Long-session runtime reads

`SessionManager` keeps cumulative usage in its existing journal index. `getUsageStatistics()` includes task and background model usage; `getAssistantUsageStatistics()` is the top-level assistant-only subtotal displayed by the footer. Branch navigation does not reset either cumulative total.

Credential pins and retained-context controls use one active branch fold and one reset/compaction checkpoint. Recent sibling branches reuse that checkpoint; navigating to unrelated older history may still require a full ancestry walk. `buildSessionContext({ transcript: true })` intentionally retains full-history export semantics. No historical messages are evicted by these optimizations.

After modifying persisted entries in place, call `await sessionManager.rewriteEntries()` before reading derived state. It rebuilds the index while preserving the selected leaf, including for in-memory sessions. Returned credential maps and usage snapshots may be changed without changing the index.

## Rewind viewport

With an empty editor, press Escape twice to open rewind at the recent tail. Up/Down choose rendered turns; Left/Right move between sibling branches at a fork or between user turns otherwise. Enter rewinds to the outlined source entry; Escape cancels. Home/End, PageUp/PageDown, and the mouse wheel inspect history without changing the selected rewind destination.

Rewind uses journal-entry anchors rather than guessed global row numbers. Edge arrows indicate more history; cold history remains accessible without pre-rendering it. The first open indexes source descriptors cooperatively with a cancellable loading view. Only demanded transcript components are instantiated, and offscreen components are released. Reopening unchanged history reuses the source index and bounded last-visible window; source rewrites, branch changes, and relevant presentation changes invalidate that view. A single large message or grouped tool card still costs the work required by its own renderer.
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

## Context controls and source directives

Interactive `/context` opens the fullscreen source manager; `/context usage` and `/context details` open the physical usage and inventory views. Headless `/context` reports usage, with the same explicit `usage` and `details` arguments. The manager starts at the newest real user in chronological order. Source rows stay one terminal-cell-truncated line; first/recent badges describe admitted policy positions rather than filtered row numbers.

Use the visible Actions menu (`a`) or `n` for Never, `y` for Always, `-`/Backspace for Auto, and Space/Enter to cycle. Independent role and stored-state filters (`f`) compose with chronological search (`/`). Inspect (`i`) distinguishes stored manual state, effective policy, eleven-category facts, pending/failed work, source quota and installed representation. Source settings (`l`) links the relevant stage and limits to native settings controls. Never disables additional preservation, not ordinary retention or deletion; hard-recent may temporarily protect a stored Never source. Manual tool exchanges are indivisible.

Reset row (`r`) affects the selected complete source group. Reset all (`R`) captures non-Auto groups throughout the current branch after its clear boundary, including hidden rows; confirmation identifies the scope. Newer edits are skipped, and neither action removes successful classification facts or settings. Classify selected (`c`) and missing-only backfill (`C`) show the resolved model and request-cost warning; backfill offers a positive worker count, progress, cancellation and explicit resume. Closing the menu does not cancel session jobs.

Context settings use typed limit, enum, number, model and regex submenus. First/recent limits are independent; the shared Always cap can link to either edge. Regex rules expose separate Final and case-insensitive controls. Policy edits update the next-compaction preview immediately but do not rewrite already installed PNG/native artifacts. Percentage limits remain visibly unavailable without an effective model maximum.

Prefix ordinary input with `/keep` or `/once` to store Always or Never source state. The directive is not emitted as authored body text; literal remainder, original image references and source-capture identity survive submission, steering/follow-up queues and editor restoration across TUI, ACP, RPC and collaboration. There are no `/pin` or `/unpin` source commands.

## Source preservation policy API

`session/preserved-messages` exposes a branch-local `PreservedMessageQuery`. Build it cooperatively from the active journal branch and a currentness predicate; it ignores entries before the latest clear boundary. Window-only settings changes query compact count/price aggregates without retokenizing source history. Rows and selected candidate messages are materialized on demand.

Use `appendEntries` for an already validated same-branch journal suffix rather than collecting the whole branch after every send. Reset or divergent ancestry requires a new cooperative build. `getManualGroup`/`getManualGroups` expose complete atom members and existing override journal IDs for reset revalidation; `invalidateClassifications` clears changed-input facts before fresh decisions are published. `readPreservedUserMessageClassificationMasks` provides the same real-user/post-clear fact interpretation for asynchronous backfill without pricing history. Selection memberships and aggregate totals are lazy; explicitly materialize the required candidates at the owning operation snapshot boundary before awaiting method work.

First/recent/hard-recent limits independently accept Off, All, message count, tokens, or a percentage of the effective model maximum context. Finite token/percentage zero is not Off. Heuristics only remove candidates; ordinary regex, eleven-category policy, Final regex, then manual state resolve in order. Auto is neutral and Keep wins within a stage. Hard-recent temporarily bypasses Never and pruning. Manual Always also bypasses pruning, but every Always source shares the configured linked cap.

When no effective model maximum is available, percentage-derived windows are reported in `selection.unavailableLimits`; the result is a provisional preview, not a finite-zero cap or complete compaction input. Other configured units and manual/source inspection still work. `selection.blockers` names the actual source that stopped each finite edge. Non-Auto reset enumeration captures its source boundary when the iterator is requested and avoids a synchronous all-history sort.

The result separates selected users `P`, complete admitted non-user atoms `N`, and temporary hard-recent `H`. Quota estimates are determined from source content before ordinary allocation, including the base estimate for original images. Methods must leave the ordinary user cut unchanged and precharge complete `N` once inside their own allocator using `prechargeNonUsers`; physical representation overlap never refunds source quota. Candidate spans use durable source IDs and UTF-16 text intervals. Installed compaction bytes and provider accounting remain method/lifecycle-owned.

Visible user-attributed `custom_message` journal entries support manual state as non-user `N` sources. Their current durable content is normalized through the existing custom-message helper while retaining the journal ID; they never enter automatic user windows, hard-recent, or classification. Hidden and agent-attributed custom injections do not become manual user rows.

Saved manual overrides and successful eleven-bit classification metadata keep their established v1 codecs. Settings compose layers before interpreting legacy paired message-zero limits; opening settings does not write normalized values.

## Physical context inspection

`/context details` opens the ordered classified inventory separately from the ordinary usage view. Recorded compaction, current reconstruction, and the last actual-prepared request remain distinct snapshots. The detail view lists fixed prompts/tools/context/skills, summary and gap text, retained and post-compaction content, native payloads, raster frames and original images with source coverage, structural counts, controls and qualified token quantities. Manual and automatic compact summaries use concise ordinary/added/shared results and the method’s actual target.

`AgentSession.getCompactionDiagnostics("current")` materializes the reconstructed inventory only when requested; `getSourceRepresentationDetails(sourceId)` inspects current versus captured source spans without tokenizing the inventory. Neither operation runs on policy toggles or footer updates. Source quota membership and estimates are separate from disjoint physical charges: overlap does not refund quota, and a shared frame is charged once.
Original and delivered projections of the same journal entry remain separate physical occurrences when their bytes differ. Coverage joins match the projection as well as source identity; source quota membership still refers to the journal source, not an extra bill for each projection.

`getCompactionDiagnostics("recorded")` reads frozen facts from the existing atomic compaction record. It includes the initiating model, settings, fixed counts, source reasons and quotas, and actual ordinary target/calibration, complete non-user precharge and residual allocation. Legacy records without facts remain unavailable, not reconstructed with today’s settings. Settings changes and reload do not rewrite historical facts.

`getPreparedCompactionDiagnostics()` reads the last compact inventory observed at the existing post-inband `beforeModelCall` boundary on the current history owner. It works with memory disabled, never reruns preparation, retains no request Context or image bytes, and returns an independent copy to explicit viewers. The label is actual-prepared, not confirmed sent: later provider hooks, dispatch, acceptance and billing remain unobserved. Boundary-time settings and installed archive settings remain separate.

Text tokenizer quantities, generic local image estimates, actual inline-renderer estimates and unknown native/opaque costs are labeled separately. Archive identity is not a historical renderer price; absent historical pricing stays explicitly unavailable. Original-image identity takes precedence over legacy summary-role guesses, and its physical correction replaces the base estimate once. Native file references, screenshots and generated images retain a baseline image estimate even when pixel bytes are unavailable; remaining unmeasurable metadata stays unknown. None of these local quantities is a provider invoice.
