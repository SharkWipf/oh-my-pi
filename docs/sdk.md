# SDK

The SDK is the in-process integration surface for `@oh-my-pi/pi-coding-agent`.
Use it when you want direct access to agent state, event streaming, tool wiring, and session control from a Bun process.

If you need cross-language/process isolation, use RPC mode instead.

## Installation

```bash
bun add @oh-my-pi/pi-coding-agent
```

Requires Bun 1.3.14 or newer. Before the first model-backed prompt, configure
credentials for a provider or run a keyless local provider; see
[Providers](./providers.md). Session construction can succeed without an
available model, but prompting cannot.

## Entry points

The package root, `@oh-my-pi/pi-coding-agent`, is the complete embedding surface. It includes `createAgentSession` and the focused `/sdk` exports, plus lower-level session, auth, model, mode, extension, and tool APIs.

Import these core embedding APIs from the package root:

- `createAgentSession`
- `SessionManager`
- `Settings`
- `AuthStorage`
- `ModelRegistry`
- `AgentRegistry`
- `discoverAuthStorage`
- Discovery helpers (`discoverExtensions`, `discoverSkills`, `discoverContextFiles`, `discoverPromptTemplates`, `discoverSlashCommands`, `discoverCustomTSCommands`, `discoverMCPServers`)
- Tool factory surface (`createTools`, `BUILTIN_TOOLS`, tool classes)

The narrower `@oh-my-pi/pi-coding-agent/sdk` subpath exports `createAgentSession`, its option/result types, `Settings`, `AgentRegistry`, discovery and system-prompt helpers, workspace-tree helpers, selected extension/MCP/tool types, and selected tool classes/factories. It does **not** export `SessionManager`, `AuthStorage`, or `ModelRegistry`; import those three from the package root as the examples below do.

## Quick start (auto-discovery defaults)

```ts
import { createAgentSession } from "@oh-my-pi/pi-coding-agent";

const { session, modelFallbackMessage } = await createAgentSession();

if (modelFallbackMessage) {
  process.stderr.write(`${modelFallbackMessage}\n`);
}

const unsubscribe = session.subscribe((event) => {
  if (
    event.type === "message_update" &&
    event.assistantMessageEvent.type === "text_delta"
  ) {
    process.stdout.write(event.assistantMessageEvent.delta);
  }
});

await session.prompt("Summarize this repository in 3 bullets.");
unsubscribe();
await session.dispose();
```

## What `createAgentSession()` discovers by default

`createAgentSession()` follows “provide to override, omit to discover”.

If omitted, it resolves:

- `cwd`: `getProjectDir()`
- `agentDir`: `~/.omp/agent` (via `getAgentDir()`)
- `authStorage`: `discoverAuthStorage(agentDir)`
- `modelRegistry`: `new ModelRegistry(authStorage)` + background `refreshInBackground()` when the registry is not provided
- `settings`: `await Settings.init({ cwd, agentDir })`
- `sessionManager`: `SessionManager.create(cwd, SessionManager.getDefaultSessionDir(cwd, agentDir))` (file-backed)
- skills/rules/context files/prompt templates/slash commands/extensions/custom TS commands
- built-in tools via `createTools(...)`
- MCP tools (enabled by default; Exa MCP servers are folded into native Exa integration, and browser automation MCP servers are filtered when the built-in Eval browser prelude is enabled)
- LSP integration (enabled by default)
- `eventBus`: new `EventBus()` unless supplied

### Required vs optional inputs

Typically you must provide only what you want to control:

```ts
function createAgentSession(
  options?: CreateAgentSessionOptions,
): Promise<CreateAgentSessionResult>;
```

- **Must provide**: nothing for a minimal session
- **Usually provide explicitly** in embedders:
  - `sessionManager` (if you need in-memory or custom location)
  - `authStorage` + `modelRegistry` (if you own credential/model lifecycle)
  - `model` or `modelPattern` (if deterministic model selection matters)
  - `settings` (if you need isolated/test config)

For multiple concurrent top-level sessions in one process, pass a private
`AgentRegistry` to each session. The default process-global registry admits
only one `"Main"` identity per generation.

## Session manager behavior (persistent vs in-memory)

`AgentSession` always uses a `SessionManager`; behavior depends on which factory you use.

### File-backed (default)

```ts
import { createAgentSession, SessionManager } from "@oh-my-pi/pi-coding-agent";

const { session } = await createAgentSession({
  sessionManager: SessionManager.create(process.cwd()),
});

console.log(session.sessionFile); // absolute .jsonl path
```

- Persists conversation/messages/state deltas to session files.
- Supports resume/open/list/fork workflows.
- `session.sessionFile` is defined.

### In-memory

```ts
import { createAgentSession, SessionManager } from "@oh-my-pi/pi-coding-agent";

const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
});

console.log(session.sessionFile); // undefined
```

- No filesystem persistence.
- Useful for tests, ephemeral workers, request-scoped agents.
- Session methods still work, but persistence-specific behaviors (file resume/fork paths) are naturally limited.

### Resume/open/list helpers

```ts
import { SessionManager } from "@oh-my-pi/pi-coding-agent";

const recent = await SessionManager.continueRecent(process.cwd());
const listed = await SessionManager.list(process.cwd());
const opened = listed[0] ? await SessionManager.open(listed[0].path) : null;
```

## Model and auth wiring

`createAgentSession()` uses `ModelRegistry` + `AuthStorage` for model selection and API key resolution.

If both `authStorage` and `modelRegistry` are supplied,
`modelRegistry.authStorage` MUST be the same instance; session creation rejects
divergent stores.

### Explicit wiring

```ts
import {
  createAgentSession,
  discoverAuthStorage,
  ModelRegistry,
  SessionManager,
} from "@oh-my-pi/pi-coding-agent";

const authStorage = await discoverAuthStorage();
const modelRegistry = new ModelRegistry(authStorage);
await modelRegistry.refresh();

const available = modelRegistry.getAvailable();
if (available.length === 0)
  throw new Error("No authenticated models available");

const { session } = await createAgentSession({
  authStorage,
  modelRegistry,
  model: available[0],
  thinkingLevel: "medium",
  sessionManager: SessionManager.inMemory(),
});
```

### Selection order when `model` is omitted

When no explicit `model`/`modelPattern` is provided:

1. restore model from existing session (if restorable + key available)
2. settings default model role (`default`)
3. an authenticated provider-default model in availability order (falling back to the first authenticated available model when no provider default is present)

If restore fails, `modelFallbackMessage` explains fallback.

### Auth priority

`AuthStorage.getApiKey(...)` resolves in this order:

1. runtime override (`setRuntimeApiKey`, used by CLI `--api-key`)
2. config-sourced API key override (`models.yml` provider `apiKey`)
3. stored OAuth credential, including refresh when needed
4. API key persisted by a successful `/login`
5. provider environment variables
6. other stored API-key credential in `agent.db` / broker-backed storage
7. custom-provider resolver fallback

## Event subscription model

Subscribe with `session.subscribe(listener)`; it returns an unsubscribe function.

```ts
const unsubscribe = session.subscribe((event) => {
  switch (event.type) {
    case "agent_start":
    case "turn_start":
    case "tool_execution_start":
      break;
    case "message_update":
      if (event.assistantMessageEvent.type === "text_delta") {
        process.stdout.write(event.assistantMessageEvent.delta);
      }
      break;
  }
});
```

`AgentSessionEvent` includes core `AgentEvent` plus session-level events:

- `auto_compaction_start` / `auto_compaction_end`
- `auto_retry_start` / `auto_retry_end`
- `retry_fallback_applied` / `retry_fallback_succeeded`
- `model_changed`
- `thinking_level_changed`
- `ttsr_triggered`
- `todo_reminder` / `todo_auto_clear`
- `irc_message`
- `notice`
- `goal_updated`

`agent_end` includes `messages`, optional telemetry fields, and
`isTerminal?: boolean`. When `isTerminal` is `false`, maintenance or async
delivery will resume the session before its true final settle. Subscribers that
use `agent_end` as a completion signal MUST wait for `isTerminal !== false`.
Treat an absent field as terminal for compatibility with older runtimes.

## Prompt lifecycle

`session.prompt(text, options?)` is the primary entry point.

Behavior:

1. optional command/template expansion (`/` commands, custom commands, file slash commands, prompt templates)
2. if currently streaming:
   - `streamingBehavior: "steer" | "followUp"` chooses how `prompt()` queues
   - extension `sendUserMessage(content)` defaults to steer when `deliverAs` is omitted
   - queued messages are preserved instead of throwing work away
3. if idle:
   - validates model + API key
   - appends user message
   - starts agent turn

Normal input accepts optional `originalSubmission: { text, images?, imageLinks?, compactionOverride? }` metadata for hosts that transform input before delivery. `text` is the exact typed text, including a literal `/keep` or `/once` prefix when present; the override separately records its parsed meaning. Prompt, steer, follow-up and custom human input retain original image bytes and links through queue restoration. Accepted transformed originals are persisted with the ordinary journal message; an undelivered queued draft creates no source entry. Identical originals reuse the delivered content. This path does not capture, index or flush requirements memory when V2 is disabled.

`producer` is host provenance, independent of wire role and billing attribution. Normal operator prompts default to `{ type: "human" }`; generated, extension and tool input retain their explicit producers. Hosts invoking `promptCustomMessage` for human input must pass the human producer explicitly. Branch and tree-navigation results include optional rich `sourceInput` metadata for draft restoration; when an original is present its text is already exact and must not receive another preservation-command prefix.

Related APIs:

- `sendUserMessage(content, { deliverAs? })`
- `steer(text, images?)`
- `followUp(text, images?)`
- `sendCustomMessage({ customType, content, ... }, { deliverAs?, triggerTurn? })`
- `abort()`

`deliverAs: "aside"` (both APIs) delivers at the next agent step boundary without interrupting the current tool batch, instead of steering (which skips remaining tools) or waiting for the run to finish. When the session is idle both start a turn instead (in plan mode the custom message is folded into context without a turn).

## User-message classification jobs

Classification records eleven independent category facts; preservation settings interpret them separately. It never selects retention or delays `prompt()`, compaction, or disposal. Live scheduling requires both `compaction.keepUserMessages` and `compaction.keepUserMessagesLlm`; stored-category filtering and explicit actions remain independent. The configured classifier model defaults to `@tiny`, never an expensive active-model fallback.

The model returns one JSON object containing all eleven canonical category names (`longTermRule`, `longTermGoal`, `lastingSolution`, `shortTermTask`, `shortTermContext`, `venting`, `restorationGuidance`, `preventionGuidance`, `contextFreeInstruction`, `banter`, `question`) with boolean values. Key order is irrelevant; code maps names to the existing category bits. Missing, unknown, nonboolean, or legacy ordinal-string responses fail visibly without replacing valid facts. Successful facts retain the packed `{v:1,c:[sourceId,mask,...]}` storage format.

- `await session.getMessageClassificationAvailability()` returns `{ available, model?, reason? }` after model/credential resolution. Unavailable models launch no job.
- `await session.startMessageClassification(sourceId)` returns a job ID and forces selected-message classification, including a rerun of valid facts. Concurrent starts for the same current source share a job; a completed, canceled, or input-invalidated attempt may be explicitly retried immediately. Existing valid facts remain active until a new valid success.
- `await session.startMessageClassificationBackfill(workers)` returns a job ID for every missing/current-unusable real user in the captured active post-clear branch. Workers must be a positive integer; there is no fixed worker ceiling. Overlapping selected/live requests are not duplicated, do not consume owned backfill worker slots, and their outcomes are included before backfill completes. Warn users about substantial model requests, tokens, and time before launching. Presentation filters do not narrow this scope.
- `session.getMessageClassificationStatus({ includeRows: false })` reads job counts without enumerating retained row failures; `getMessageClassificationRowStatus(sourceId)` reads one runtime row. Omitting the option includes the runtime row snapshot. Persisted category facts remain the authority after successful rows settle.
- `session.subscribeMessageClassification((status, affectedIds) => ...)` returns an unsubscribe function. Subscription `status.rows` contains only affected row deltas, not the full row snapshot. Unsubscribing or closing a UI leaves jobs running.
- `session.cancelMessageClassification(jobId)` cancels only that job and its in-flight/scanner work; independent live/selected work continues. Saved facts remain intact.

Classification reads the current durable source message and its entry ID, not immutable V2 capture evidence. Source identities and inputs are revalidated before successful v1 facts are appended. Branch/reset/session changes interrupt stale work without appending to another branch; vetoed transitions resume the original scope. Failure is not an all-false classification. Restart never resumes requests automatically: explicitly launch missing-only backfill.

Cold source capture cooperates with the event loop; backfill projects its prior-user/two-assistant neighborhood in one forward pass. Before and after each request, validation reads only the captured current and auxiliary source IDs. Ordinary later appends cannot change those preceding neighbors. The existing source-rewrite callback invalidates dependent targets when content or eligibility changes, including affected rows not yet admitted by an active backfill; those rows require explicit retry while unaffected rows continue.

## Preserved source state

`await session.preparePreservedMessages()` establishes durable source IDs and the active post-reset query. `session.getPreservedMessageQuery()` returns the current query when prepared; it never starts an asynchronous rebuild. Rows and manual actions identify current source entries, not equal text or immutable V2 capture evidence. Selected image blocks use the complete atomic source interval `[0, 1)`, not an empty text interval.

Invalidated source preparation rejects with `CompactionCancelledError`. Automatic pruning treats this as cancellation and retains the initiating ownership across both pruning passes, so a stale attempt cannot continue on a new branch or session. Storage failures still propagate as errors.

Human `prompt`, `steer`, `followUp`, and explicitly human-produced `sendUserMessage` inputs recognize `/keep <message>` and `/once <message>` once at the original input boundary. The remaining body is literal, not another extension/template command. Empty directives are rejected. Explicit `compactionOverride` and restored `sourceCaptureId` inputs preserve their existing decision instead of reinterpreting the body; generated messages do not acquire user policy from directive-looking text. Queue restoration retains the manual state, source capture, image attachments, and attachment links. Initial manual state is persisted with the delivered source entry.

Maintenance compaction and handoff freeze applicable requirements with their operation. A later live requirements change does not silently replace that operation’s provider input. Handoff composes its frozen requirements segment once, before the existing physical provider transforms. Unresolved requirements deliveries remain ordinary, fully charged retained history until acknowledged; this mandatory source visibility is independent of user-preservation policy. The same active pending delivery IDs protect original content from automatic pruning and shaking.

- `session.getPreservedMessageSelection()` exposes the current policy result, including selected user sources, admitted non-user atoms, reasons, quota totals, blockers, and `unavailableLimits`. An unavailable model-relative percentage is not zero or unlimited.
- `session.subscribePreservedMessages(affectedIds => ...)` returns an unsubscribe function. An omitted ID list signals a scope/policy-wide refresh; saved classification changes publish affected source IDs.
- `session.getPreservedMessagesOwnership()` identifies the active session/branch/reset scope. Ordinary same-branch appends and policy changes do not replace this identity.
- `await session.setPreservedMessageOverride(sourceId, "auto" | "keep" | "exclude")` writes the complete manageable source atom through the journal durability boundary. Auto removes the manual override; it does not erase classifier facts or disable automatic protection.
- `await session.capturePreservedMessageOverrideReset(sourceIds?)` captures a finite, durable confirmation snapshot, including the selected IDs, their current override revisions, and source/group counts. Omitting IDs captures every non-Auto group in the active scope; UI filters must not narrow reset-all. Large captures yield cooperatively.
- `await session.resetPreservedMessageOverrides(snapshot)` returns `{ reset, skipped }`. Same-branch suffixes are allowed but excluded from the captured targets; newer manual edits are skipped instead of overwritten. Branch/reset/session changes reject stale confirmation. Reusing a completed snapshot is idempotent, and durable-write recovery reuses its committed journal transition.
- `await session.recoverCompactionPersistence()` retries durable publication of the same frozen compaction event. It never appends a duplicate or installs an old result onto a different active branch.

Append-only metadata batches preserve the existing journal bytes and serialize only their new entries. `SessionStorage` implementations must provide `appendTextAtomic(path, suffix, options?)`: publish the complete suffix as one operation against an existing file and honor the same commit guard as `writeTextAtomic`. File storage stages an asynchronous copy of the opaque prefix plus the suffix before guarded replacement; memory and indexed backends append the complete suffix without materializing the old journal in the session manager. Actual source rewrites and authoritative failure recovery still use full atomic replacement.

## `AgentSession` lifecycle and disposal

Call `await session.dispose()` when the embedder is completely done with a session. `dispose()` starts disposal itself and is idempotent: repeated or concurrent calls receive the same teardown promise, so shutdown events and owned resources are not drained twice.

`beginDispose()` is the synchronous admission barrier for wrappers that must await their own teardown before calling `dispose()`. Call it before the wrapper's first `await`; otherwise deferred work can enter the gap. It immediately marks the session disposed, cancels memory startup, title generation, and auto-learn capture, clears queued yield/asides, stops advisor runtime, detaches aside delivery, and rejects new eval executions. Deferred session work checks the disposed state and is dropped or skipped. `beginDispose()` is also idempotent, and the later `dispose()` call remains required to finish asynchronous cleanup.

```ts
import type { AgentSession } from "@oh-my-pi/pi-coding-agent";

async function closeEmbeddedSession(
  session: AgentSession,
  closeHostInputAndUi: () => Promise<void>,
): Promise<void> {
  session.beginDispose(); // no new deferred work may enter after this point
  await closeHostInputAndUi();
  await session.dispose();
}
```

During asynchronous disposal, the session records and synchronously flushes its exit diagnostic, emits `session_shutdown` once, stops extension fallback timers, aborts retries, compaction, and the active agent turn, and gives post-prompt and auto-learn work bounded time to settle. It then tears down session-owned async jobs, eval kernels, browser tabs, native computer sessions, MCP connections, advisor state, and memory state concurrently. These subsystem drains are best-effort and bounded where applicable; failures are logged rather than preventing the remaining subsystem cleanup.

Only after work capable of appending session entries has settled does disposal clean up an empty moved session, close the `SessionManager`, close provider session state, disconnect the agent, and remove listeners. A failure from the final persistence cleanup or `SessionManager.close()` rejects the shared disposal promise; individual provider-session close failures are logged.

## Tools and extension integration

### Built-ins and filtering

- Built-ins come from `createTools(...)` and `BUILTIN_TOOLS`.
- `toolNames` requests named tools and can enable tools that are disabled by
  default; by itself it is **not** an allowlist.
- Set `restrictToolNames: true` to limit the session to the names in
  `toolNames`. Restricted sessions disable ambient MCP, extensions, custom
  commands, and LSP by default.
- In a restricted session, SDK-supplied `customTools` are excluded unless
  `allowRestrictedCustomTools: true` and their names also appear in
  `toolNames`.
- Hidden tools (for example `yield`) are opt-in unless required by options.

```ts
const { session } = await createAgentSession({
  toolNames: ["read", "grep", "glob", "write"],
  restrictToolNames: true,
  requireYieldTool: true,
});
```

### Extensions

- `extensions`: inline `ExtensionFactory[]`
- `additionalExtensionPaths`: load extra extension files
- `disableExtensionDiscovery`: disable ambient scanning; explicit paths and
  inline factories still load
- `preloadedExtensions`: reuse an extension set loaded early by the same
  session-owning process. Never pass loaded extension instances from a parent
  to another session; use `preloadedExtensionPaths` so each session gets its
  own `ExtensionAPI` binding.

### Runtime tool set changes

`AgentSession` supports runtime activation updates:

- `getActiveToolNames()`
- `getAllToolNames()`
- `setActiveToolsByName(names)`
- `refreshMCPTools(mcpTools)`

System prompt is rebuilt to reflect active tool changes.

## Discovery helpers

Use these when you want partial control without recreating internal discovery logic:

- `discoverAuthStorage(agentDir?)`
- `discoverExtensions(cwd?)`
- `discoverSkills(cwd?, _agentDir?, settings?)`
- `discoverContextFiles(cwd?, _agentDir?, disabledExtensions?)`
- `discoverPromptTemplates(cwd?, agentDir?)`
- `discoverSlashCommands(cwd?)`
- `discoverCustomTSCommands(cwd?, agentDir?)`
- `discoverMCPServers(cwd?)`
- `buildSystemPrompt(options?)`

## Subagent-oriented options

For SDK consumers building orchestrators (similar to task executor flow):

- `outputSchema`: passes structured output expectation into tool context
- `outputSchemaMode`: selects permissive or strict structured-output enforcement
- `requireYieldTool`: forces `yield` tool inclusion
- `taskDepth`: recursion-depth context for nested task sessions
- `parentTaskPrefix`: artifact naming prefix for nested task outputs

These are optional for normal single-agent embedding.

## `createAgentSession()` return value

```ts
type CreateAgentSessionResult = {
  session: AgentSession;
  extensionsResult: LoadExtensionsResult;
  setToolUIContext: (uiContext: ExtensionUIContext, hasUI: boolean) => void;
  mcpManager?: MCPManager;
  modelFallbackMessage?: string;
  lspServers?: Array<{
    name: string;
    status: "connecting" | "ready" | "error" | "available";
    fileTypes: string[];
    error?: string;
  }>;
  eventBus: EventBus;
};
```

Use `setToolUIContext(...)` only if your embedder provides UI capabilities that tools/extensions should call into.

## Startup performance

`createAgentSession()` runs two background optimizations to overlap I/O with the rest of session setup:

- **Model-host preconnect.** As soon as the model is resolved, the SDK fires a best-effort `fetch.preconnect(model.baseUrl)` so DNS + TCP + TLS + HTTP/2 to the provider's host happens in parallel with extension/skill load, tool registry build, and system-prompt assembly. The first real `fetch(...)` then reuses the warm connection, saving 100–300 ms on transcontinental hops (e.g. residential IP → `api.anthropic.com`). Implementation lives in `preconnectModelHost()` in `packages/coding-agent/src/sdk.ts`. If `fetch.preconnect` is unavailable (non-Bun runtime) or the call throws, the optimization is silently skipped — never a hard dependency. Applies to every mode (interactive, print, RPC, ACP).
- **Conditional LSP warmup.** Startup LSP servers (those returned by `discoverStartupLspServers(cwd)`) are only warmed when **all** of these hold:
  - `enableLsp !== false` on the session options, **and**
  - `options.hasUI === true` (interactive TUI), **and**
  - the `lsp.lazy` setting is disabled (it defaults to `true`).

  With `lsp.lazy` enabled — the default — no language servers are launched at startup at all; each server cold-starts on first use, i.e. when the agent invokes the `lsp` tool or an edit/write touches a file whose extension matches the server's `fileTypes`. Print / script / RPC / ACP invocations (`hasUI=false`) skip the warmup regardless of the setting: they don't render the warmup status indicator and typically finish before the language servers would stabilize, so warming them just spends CPU parsing big `initialize` responses concurrently with the LLM stream consumer and jitters perceived latency. Tools that actually need an LSP server still spin one up on demand through `getOrCreateClient()` — only the _startup_ warmup is skipped. The returned `lspServers` field in `CreateAgentSessionResult` is still populated for UI sessions in lazy mode — recognized servers are discovered (no processes spawned) and reported with status `"available"` so the welcome screen and `/status` can list them; it is `undefined` only when `enableLsp === false` or `hasUI === false`.

## Minimal controlled embed example

```ts
import {
  createAgentSession,
  discoverAuthStorage,
  ModelRegistry,
  SessionManager,
  Settings,
} from "@oh-my-pi/pi-coding-agent";

const authStorage = await discoverAuthStorage();
const modelRegistry = new ModelRegistry(authStorage);
await modelRegistry.refresh();

const settings = Settings.isolated({
  "compaction.enabled": true,
  "retry.enabled": true,
});

const { session } = await createAgentSession({
  authStorage,
  modelRegistry,
  settings,
  sessionManager: SessionManager.inMemory(),
  toolNames: ["read", "grep", "glob", "edit", "write"],
  enableMCP: false,
  enableLsp: true,
});

session.subscribe((event) => {
  if (
    event.type === "message_update" &&
    event.assistantMessageEvent.type === "text_delta"
  ) {
    process.stdout.write(event.assistantMessageEvent.delta);
  }
});

await session.prompt("Find all TODO comments in this repo and propose fixes.");
await session.dispose();
```
