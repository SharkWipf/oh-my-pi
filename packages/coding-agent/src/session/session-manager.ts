import * as fs from "node:fs";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { CompactionDiagnostics } from "@oh-my-pi/pi-agent-core/compaction/diagnostics";
import { getCompactionSourceRepresentation, type SourceRewrite } from "@oh-my-pi/pi-agent-core/compaction/source";
import {
	REQUIREMENTS_OPERATOR_DECISION_ENTRY,
	requirementsHash,
	requirementsUnits,
	type ResolvedRequirementsSource,
} from "../requirements/source-capture";
import type { RequirementsObservation, RequirementsSource } from "../requirements/types";

import * as path from "node:path";
import type {
	ImageContent,
	Message,
	MessageAttribution,
	ServiceTierByFamily,
	TextContent,
	Usage,
} from "@oh-my-pi/pi-ai";
import {
	directoryIsEnterable,
	getBlobsDir,
	getProjectDir,
	getSessionsDir,
	isEnoent,
	logger,
	stringifyJson,
	toError,
} from "@oh-my-pi/pi-utils";
import type { StructuredSubagentSchemaMode } from "../task/types";
import { ArtifactManager } from "./artifacts";
import { type BlobPutOptions, type BlobPutResult, BlobStore, isBlobRef, parseBlobRef } from "./blob-store";
import type { CompactionMethod } from "./compaction-methods";
import {
	type BashExecutionMessage,
	type CustomMessage,
	type FileMentionMessage,
	type HookMessage,
	normalizeCustomMessagePayload,
	type PythonExecutionMessage,
	sanitizeRehydratedOpenAIResponsesAssistantMessage,
	stripInternalDetailsFields,
} from "./messages";
import {
	applySessionContextControlEntry,
	type BuildSessionContextOptions,
	buildSessionContext,
	buildSessionContextFromPath,
	cloneSessionContextControlState,
	createSessionContextControlState,
	getOpenAiRemoteCompactionPayload,
	type SessionContext,
	type SessionContextControlState,
	type SessionContextSourceInventory,
} from "./session-context";
import {
	type BranchSummaryEntry,
	type CompactionEntry,
	type CredentialPinEntry,
	CURRENT_SESSION_VERSION,
	type CustomEntry,
	type CustomMessageEntry,
	type FileEntry,
	type LabelEntry,
	type ModeChangeEntry,
	type ModelChangeEntry,
	type ModelUsageEntry,
	type NewSessionOptions,
	type ResetBoundaryEntry,
	type ServiceTierChangeEntry,
	type SessionEntry,
	type SessionHeader,
	type SessionInitEntry,
	type SessionMessageEntry,
	type SessionTitleSource,
	type SessionTreeNode,
	type ThinkingLevelChangeEntry,
	TITLE_CHANGE_ENTRY_TYPE,
	type TitleChangeEntry,
	type TtsrInjectionEntry,
	type UsageStatistics,
} from "./session-entries";
import { findMostRecentSession, listAllSessions, listSessions, type SessionInfo } from "./session-listing";
import {
	loadEntriesFromFile,
	loadSessionFile,
	resolveBlobRefsInEntries,
	type SessionLoadResult,
	visitEntriesFromFile,
} from "./session-loader";
import { generateId, migrateToCurrentVersion } from "./session-migrations";
import {
	computeDefaultSessionDir,
	readTerminalBreadcrumbEntry,
	resolveManagedSessionRoot,
	writeTerminalBreadcrumb,
} from "./session-paths";
import { isPersistenceTruncatedString, prepareEntryForPersistence } from "./session-persistence";
import { loadPinnedSessionIds, sortPinnedFirst } from "./session-pins";
import { rewriteSessionSources } from "./session-source-rewrite";
import { IndexedSessionStorage } from "./indexed-session-storage";
import {
	FileSessionStorage,
	MemorySessionStorage,
	type SessionStorage,
	type SessionStorageWriter,
} from "./session-storage";
import { type SessionTitleUpdate, serializeTitleSlot } from "./session-title-slot";
import {
	additionalWorkspaceDirectories,
	normalizeSessionWorkspace,
	normalizeWorkspaceDirectory,
} from "./session-workspace";
import { recordSessionTitle } from "./title-index";

const JSONL_SUFFIX_LENGTH = ".jsonl".length;
const DRAFT_ONLY_SESSION_MARKER = ".draft-only-session";
const DISCARDED_ENTRY_BRANCH_MARKER = "discarded-entry-branch";

function mintSessionId(): string {
	return Bun.randomUUIDv7();
}

function nowIso(): string {
	return new Date().toISOString();
}

function fileSafeTimestamp(iso: string): string {
	return iso.replace(/[:.]/g, "-");
}

function artifactsDirectoryFor(sessionFile: string | undefined): string | null {
	if (!sessionFile?.endsWith(".jsonl")) return null;
	return sessionFile.slice(0, -JSONL_SUFFIX_LENGTH);
}

/** Copy a session's artifact directory to another session, matching interactive `/fork`. */
export async function copySessionArtifacts(sourceSessionFile: string, destinationSessionFile: string): Promise<void> {
	const sourceArtifactsDir = artifactsDirectoryFor(sourceSessionFile);
	const destinationArtifactsDir = artifactsDirectoryFor(destinationSessionFile);
	if (!sourceArtifactsDir || !destinationArtifactsDir) return;
	if (path.resolve(sourceArtifactsDir) === path.resolve(destinationArtifactsDir)) return;

	try {
		const sourceStat = await fs.promises.stat(sourceArtifactsDir);
		if (sourceStat.isDirectory()) {
			await fs.promises.cp(sourceArtifactsDir, destinationArtifactsDir, { recursive: true });
		}
	} catch (error) {
		if (!isEnoent(error)) {
			logger.warn("Failed to copy artifacts during fork", {
				sourceArtifactsDir,
				destinationArtifactsDir,
				error: error instanceof Error ? error.message : String(error),
			});
		}
	}
}

/**
 * Resolve a breadcrumb's recorded session file to its interactive root. Subagent
 * (and other artifact) sessions live inside a parent session's artifacts dir —
 * `<parent>.jsonl` strips its suffix to `<parent>/`, and a child writes
 * `<parent>/<agentId>.jsonl`. A breadcrumb that points at such a child — a
 * pre-fix poisoned crumb left by a subagent that opened in the parent's TTY, or
 * any nested artifact — must resolve back up to the top-level session so
 * `--continue` resumes the real conversation instead of a subagent transcript.
 */
function resolveBreadcrumbToInteractiveRoot(sessionFile: string): string {
	let current = path.resolve(sessionFile);
	// Walk up while the containing dir is itself a session's artifacts dir
	// (`<dir>.jsonl` exists). Capped to defend against pathological layouts.
	for (let depth = 0; depth < 8; depth++) {
		const parentSessionFile = `${path.dirname(current)}.jsonl`;
		if (!fs.existsSync(parentSessionFile)) return current;
		current = parentSessionFile;
	}
	return current;
}

function emptyUsageStatistics(): UsageStatistics {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		orchestrationInput: 0,
		orchestrationOutput: 0,
		orchestrationCacheRead: 0,
		premiumRequests: 0,
		cost: 0,
	};
}

function taskUsageFrom(details: unknown): Usage | undefined {
	if (details === null || typeof details !== "object") return undefined;
	const maybeUsage = (details as Record<string, unknown>).usage;
	return maybeUsage !== null && typeof maybeUsage === "object" ? (maybeUsage as Usage) : undefined;
}

function entryUsage(entry: SessionEntry): Usage | undefined {
	if (entry.type === "model_usage") return entry.usage;
	if (entry.type !== "message") return undefined;
	const message = entry.message;
	if (message.role === "assistant") return message.usage;
	if (message.role === "toolResult" && message.toolName === "task") return taskUsageFrom(message.details);
	return undefined;
}

function addUsage(target: UsageStatistics, usage: Usage | undefined): void {
	if (!usage) return;
	target.input += usage.input;
	target.output += usage.output;
	target.cacheRead += usage.cacheRead;
	target.cacheWrite += usage.cacheWrite;
	target.totalTokens += usage.totalTokens;
	target.orchestrationInput += usage.orchestration?.input ?? 0;
	target.orchestrationOutput += usage.orchestration?.output ?? 0;
	target.orchestrationCacheRead += usage.orchestration?.cacheRead ?? 0;
	target.premiumRequests += usage.premiumRequests ?? 0;
	target.cost += usage.cost.total;
}

/**
 * Zero the monetary attribution on one usage record in place, leaving token
 * counts untouched. Cost, credit meters, and premium-request counts describe
 * billing; forks that must not inherit spend (see {@link SessionManager.forkFrom}
 * `resetInheritedCost`) drop them while keeping the tokens compaction relies on.
 */
function resetUsageCost(usage: Usage | undefined): void {
	if (!usage) return;
	usage.cost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
	usage.credits = undefined;
	usage.premiumRequests = undefined;
}

function isAssistantEntry(entry: SessionEntry): boolean {
	return entry.type === "message" && entry.message.role === "assistant";
}

function isDraftOnlyMetadataEntry(entry: SessionEntry): boolean {
	// Startup-recorded selector state that does not survive as user intent
	// once the draft is cleared. `mode_change` covers the `plan.defaultOnStartup`
	// path (interactive-mode.ts enters plan mode before draft restoration) and
	// `/plan` toggles that leave the session otherwise empty; entries carrying
	// real conversation state — messages, compactions, branch summaries,
	// custom/custom_message, session_init, labels, title/tool selection — never
	// reach this branch and always keep the file resumable.
	switch (entry.type) {
		case "model_change":
		case "thinking_level_change":
		case "service_tier_change":
		case "mode_change":
		case "credential_pin":
			return true;
		default:
			return false;
	}
}

function orderedByTimestamp(a: SessionTreeNode, b: SessionTreeNode): number {
	return new Date(a.entry.timestamp).getTime() - new Date(b.entry.timestamp).getTime();
}

interface SessionBranchFold {
	id: string | null;
	controls: SessionContextControlState;
	pins: Map<string, { hash: string; lastUsedAt: number }>;
	sourceContext?: { compactionId: string; path: SessionEntry[]; inventory: SessionContextSourceInventory };
}

/**
 * Maintains the derived views over a session's entry list: id lookup, the
 * parent→children adjacency, the resolved label map, the active leaf, and the
 * running usage totals. Kept in lockstep with the manager's `#entries` so reads
 * stay O(1)/O(children) instead of rescanning the whole journal.
 */
class SessionEntryIndex {
	#entriesById = new Map<string, SessionEntry>();
	#children = new Map<string | null, SessionEntry[]>();
	#labels = new Map<string, string>();
	#leaf: string | null = null;
	#usage = emptyUsageStatistics();
	#assistantUsage = emptyUsageStatistics();
	// One branch fold and one applicable boundary checkpoint, never one per message.
	// Older unrelated branches may replay ancestry; only recent siblings share this checkpoint.
	#fold = this.#emptyFold();
	#boundaryFold: SessionBranchFold | undefined;
	#rebuilding = false;

	#emptyFold(): SessionBranchFold {
		return { id: null, controls: createSessionContextControlState(), pins: new Map() };
	}

	#cloneFold(fold: SessionBranchFold): SessionBranchFold {
		const pins = new Map<string, { hash: string; lastUsedAt: number }>();
		for (const [provider, pin] of fold.pins) pins.set(provider, { ...pin });
		return { ...fold, controls: cloneSessionContextControlState(fold.controls), pins };
	}

	#foldEntry(entry: SessionEntry): void {
		applySessionContextControlEntry(this.#fold.controls, entry);
		if (entry.type === "credential_pin") {
			this.#fold.pins.set(entry.provider, { hash: entry.hash, lastUsedAt: new Date(entry.timestamp).getTime() });
		} else if (entry.type === "message" && entry.message.role === "assistant") {
			const pin = this.#fold.pins.get(entry.message.provider);
			if (pin) pin.lastUsedAt = Math.max(pin.lastUsedAt, entry.message.timestamp);
		}
		this.#fold.id = entry.id;
		if (entry.type === "reset_boundary" || entry.type === "compaction") {
			this.#boundaryFold = this.#cloneFold(this.#fold);
		}
	}

	branchFold(): SessionBranchFold {
		if (this.#fold.id === this.#leaf) return this.#fold;
		const pending: SessionEntry[] = [];
		const seen = new Set<string>();
		let cursor = this.#leaf ? this.#entriesById.get(this.#leaf) : undefined;
		while (cursor && !seen.has(cursor.id) && cursor.id !== this.#fold.id && cursor.id !== this.#boundaryFold?.id) {
			seen.add(cursor.id);
			pending.push(cursor);
			cursor = cursor.parentId ? this.#entriesById.get(cursor.parentId) : undefined;
		}
		if (cursor?.id === this.#boundaryFold?.id && this.#boundaryFold) this.#fold = this.#cloneFold(this.#boundaryFold);
		else if (!cursor || cursor.id !== this.#fold.id) this.#fold = this.#emptyFold();
		for (let i = pending.length - 1; i >= 0; i--) this.#foldEntry(pending[i]);
		return this.#fold;
	}

	assistantUsageSnapshot(): UsageStatistics {
		return { ...this.#assistantUsage };
	}

	#sourceContextPath(compaction: CompactionEntry): NonNullable<SessionBranchFold["sourceContext"]> {
		const fold = this.branchFold();
		if (fold.sourceContext?.compactionId === compaction.id) return fold.sourceContext;
		const representation = getCompactionSourceRepresentation(compaction.preserveData)!;
		const ancestry = this.pathTo(compaction.id);
		let reset = -1;
		let firstKept = -1;
		let through = -1;
		for (let i = 0; i < ancestry.length; i++) {
			const entry = ancestry[i];
			if (entry.type === "reset_boundary") reset = i;
			if (entry.id === compaction.firstKeptEntryId) firstKept = i;
			if (entry.id === representation.throughEntryId) through = i;
		}
		const referenced = new Set<string>();
		for (const part of representation.layout) if ("entryId" in part) referenced.add(part.entryId);
		for (const run of representation.coverage) referenced.add(run.entryId);
		const path: SessionEntry[] = [];
		const before = new Map<string, number>();
		const orders = new Map<string, number>();
		let total = 0;
		for (let i = Math.max(0, reset); i < ancestry.length; i++) {
			const entry = ancestry[i];
			const ordinary = representation.throughEntryId ? through >= 0 && i > through : firstKept >= 0 && i >= firstKept;
			if (i === reset || entry.id === compaction.id || i === through || ordinary || referenced.has(entry.id)) {
				path.push(entry);
				before.set(entry.id, total);
				orders.set(entry.id, i);
			}
			if (entry.type === "message" || entry.type === "custom_message") total++;
		}
		const sourceContext = { compactionId: compaction.id, path, inventory: { before, orders, total } };
		fold.sourceContext = sourceContext;
		if (this.#boundaryFold?.id === compaction.id) this.#boundaryFold.sourceContext = sourceContext;
		return sourceContext;
	}

	/** Walk only actual replay; the existing boundary fold holds cold source inventory. */
	contextPath(options?: BuildSessionContextOptions): { path: SessionEntry[]; inventory?: SessionContextSourceInventory } {
		const path: SessionEntry[] = [];
		const seen = new Set<string>();
		let cursor = this.leafEntry();
		let compaction: CompactionEntry | undefined;
		let first: string | undefined;
		while (cursor && !seen.has(cursor.id)) {
			seen.add(cursor.id);
			path.push(cursor);
			if (!compaction) {
				if (cursor.type === "reset_boundary") break;
				if (cursor.type === "compaction") {
					compaction = cursor;
					if (!options?.transcript && getOpenAiRemoteCompactionPayload(compaction)) {
						first = compaction.providerReplayThroughEntryId;
						if (!first) break;
					} else if (getCompactionSourceRepresentation(compaction.preserveData) && !getOpenAiRemoteCompactionPayload(compaction)) {
						const source = this.#sourceContextPath(compaction);
						const result = source.path.slice();
						for (let i = path.length - 2; i >= 0; i--) result.push(path[i]);
						return { path: result, inventory: source.inventory };
					} else first = compaction.firstKeptEntryId;
				}
			}
			if (compaction && cursor.id === first) break;
			cursor = cursor.parentId ? this.#entriesById.get(cursor.parentId) : undefined;
		}
		path.reverse();
		return { path };
	}

	constructor(private readonly onChange: () => void) {}

	clear(notify = true): void {
		this.#entriesById.clear();
		this.#children.clear();
		this.#labels.clear();
		this.#leaf = null;
		this.#usage = emptyUsageStatistics();
		this.#assistantUsage = emptyUsageStatistics();
		this.#fold = this.#emptyFold();
		this.#boundaryFold = undefined;
		if (notify) this.onChange();
	}

	rebuild(entries: readonly SessionEntry[]): void {
		this.clear(false);
		this.#rebuilding = true;
		for (const entry of entries) this.insert(entry, false);
		this.#rebuilding = false;
		this.branchFold();
		this.onChange();
	}

	insert(entry: SessionEntry, notify = true): void {
		this.#entriesById.set(entry.id, entry);
		this.#leaf = entry.id;

		const bucket = this.#children.get(entry.parentId);
		if (bucket) bucket.push(entry);
		else this.#children.set(entry.parentId, [entry]);

		if (entry.type === "label") {
			if (entry.label) this.#labels.set(entry.targetId, entry.label);
			else this.#labels.delete(entry.targetId);
		}

		const usage = entryUsage(entry);
		addUsage(this.#usage, usage);
		if (entry.type === "message" && entry.message.role === "assistant") addUsage(this.#assistantUsage, usage);
		if (!this.#rebuilding) {
			if (entry.parentId === this.#fold.id) this.#foldEntry(entry);
			else if (entry.type === "compaction" || entry.type === "reset_boundary") this.branchFold();
		}
		if (notify) this.onChange();
	}

	has(id: string): boolean {
		return this.#entriesById.has(id);
	}

	get(id: string): SessionEntry | undefined {
		return this.#entriesById.get(id);
	}

	/**
	 * The live id→entry map. Read-only for callers (lookups + `generateId`
	 * collision checks); never mutate it directly — go through `insert`/`rebuild`.
	 */
	entriesById(): Map<string, SessionEntry> {
		return this.#entriesById;
	}

	leafId(): string | null {
		return this.#leaf;
	}

	leafEntry(): SessionEntry | undefined {
		return this.#leaf ? this.#entriesById.get(this.#leaf) : undefined;
	}

	setLeaf(id: string | null): void {
		if (id === this.#leaf) return;
		this.#leaf = id;
		this.onChange();
	}

	childrenOf(parentId: string): SessionEntry[] {
		return [...(this.#children.get(parentId) ?? [])];
	}

	labelFor(id: string): string | undefined {
		return this.#labels.get(id);
	}

	labelsInEffect(): IterableIterator<[string, string]> {
		return this.#labels.entries();
	}

	usageSnapshot(): UsageStatistics {
		return { ...this.#usage };
	}

	pathTo(id: string | null | undefined = this.#leaf): SessionEntry[] {
		const branch: SessionEntry[] = [];
		const seen = new Set<string>();
		let cursor = id ? this.#entriesById.get(id) : undefined;

		while (cursor && !seen.has(cursor.id)) {
			seen.add(cursor.id);
			branch.push(cursor);
			cursor = cursor.parentId ? this.#entriesById.get(cursor.parentId) : undefined;
		}
		branch.reverse();
		return branch;
	}

	tree(entries: readonly SessionEntry[]): SessionTreeNode[] {
		const nodes = new Map<string, SessionTreeNode>();
		const roots: SessionTreeNode[] = [];

		for (const entry of entries) {
			nodes.set(entry.id, { entry, children: [], label: this.#labels.get(entry.id) });
		}

		for (const entry of entries) {
			const node = nodes.get(entry.id)!;
			const parentId = entry.parentId;
			if (parentId === null || parentId === entry.id) {
				roots.push(node);
				continue;
			}

			const parent = nodes.get(parentId);
			if (parent) parent.children.push(node);
			else roots.push(node);
		}

		const stack = [...roots];
		while (stack.length > 0) {
			const node = stack.pop()!;
			node.children.sort(orderedByTimestamp);
			stack.push(...node.children);
		}

		return roots;
	}
}

export type ReadonlySessionManager = Pick<
	SessionManager,
	| "getCwd"
	| "getRecordedCwd"
	| "getSessionDir"
	| "getSessionId"
	| "getSessionFile"
	| "getSessionName"
	| "getArtifactsDir"
	| "getArtifactManager"
	| "allocateArtifactPath"
	| "saveArtifact"
	| "getArtifactPath"
	| "getLeafId"
	| "getLeafEntry"
	| "getEntry"
	| "getLabel"
	| "getBranch"
	| "getHeader"
	| "getEntries"
	| "getTree"
	| "getUsageStatistics"
	| "putBlob"
	| "putBlobSync"
>;

interface SessionManagerStateSnapshot {
	cwd: string;
	sessionDir: string;
	sessionId: string;
	sessionName: string | undefined;
	titleSource: SessionTitleSource | undefined;
	sessionFile: string | undefined;
	titleUpdatedAt: string;
	hasTitleSlot: boolean;
	onDisk: boolean;
	needsRewrite: boolean;
	draftOnlySessionCleanupArmed: boolean;
	fallbackRuntimeOnly: boolean;
	header: SessionHeader;
	entries: SessionEntry[];
}

interface DiskQueueOptions {
	ignorePriorError?: boolean;
	ignoreEpoch?: boolean;
	epoch?: number;
}

interface AtomicEntryBatch {
	collecting: boolean;
	entryIds: Set<string>;
	deferredNotifications: SessionEntry[];
	preBatchLeafId: string | null;
	externalLeafChanged: boolean;
	externalLeafId: string | null;
}

/**
 * The storage may have published a write that rejected, and an authoritative
 * repair could not be proven durable. Callers must fail closed until recovery.
 */
export class SessionPersistenceIndeterminateError extends AggregateError {
	readonly operationError: Error;
	readonly recoveryErrors: readonly Error[];

	constructor(operationError: Error, recoveryErrors: readonly Error[]) {
		super(
			[operationError, ...recoveryErrors],
			`Session persistence is indeterminate after "${operationError.message}" and authoritative repair failed.`,
		);
		this.name = "SessionPersistenceIndeterminateError";
		this.operationError = operationError;
		this.recoveryErrors = [...recoveryErrors];
	}
}

/**
 * Stores and navigates an append-only conversation journal.
 *
 * A session is a JSONL file: one header line followed by entries. Entries form a
 * tree by `(id, parentId)`, and the mutable leaf pointer selects which path is
 * active for future appends and for LLM context construction.
 *
 * Durability is software-crash safe but not power-loss safe: completed entries
 * (user/assistant/toolResult messages, tool_execution_start markers, custom
 * entries) are handed to the OS synchronously in-body on append and never
 * `fsync`'d. In-flight streaming text is intentionally not durable until
 * `message_end` persists the finished message.
 *
 * While an in-place atomic rewrite is publishing, a concurrent completed append
 * supersedes that publish with a synchronous full-body rewrite so the entry is
 * software-crash durable before the append returns; the abandoned atomic's
 * `commitGuard` then refuses to clobber the fresher body.
 *
 * During {@link moveTo}, appends write a full body to the live relocation path
 * (source until rename, destination once the rename has landed) so a crash mid-
 * move still preserves completed entries without recreating a vacated source.
 * A trailing atomic rewrite still rewrites the header cwd after the path is
 * repointed.
 */
export class SessionManager {
	#cwd: string;
	/** Additional workspace directories beyond cwd (multi-root). Normalized absolute, deduped, excludes cwd. */
	#additionalDirectories: string[] = [];
	#fallbackRuntimeOnly = false;
	#sessionDir: string;
	readonly #persist: boolean;
	readonly #storage: SessionStorage;
	readonly #blobs: BlobStore;

	#sessionId = "";
	#sessionName: string | undefined;
	#titleSource: SessionTitleSource | undefined;
	#sessionFile: string | undefined;
	#header!: SessionHeader;
	#titleUpdatedAt = "";
	#hasTitleSlot = true;
	#entries: SessionEntry[] = [];
	#sourceChangeCallbacks = new Set<() => void>();
	#index = new SessionEntryIndex(() => this.#notifySourceChanged());
	#requirementsSourceRewriteVersion = 0;
	#requirementsEpochCache?: Map<string, number>;
	#requirementsDependencyJournals?: Map<string, RequirementsSource["locators"][number]>;
	#requirementsDependencyBlobs?: Set<string>;
	#requirementsActive?: { leaf: string | null; cursor: string | null; ids: Set<string> };
	#requirementsRetainedOriginals?: Map<string, Buffer>;
	/** File reflects all current entries; appends can go incrementally. */
	#fileIsCurrent = false;
	/** In-memory entries diverged from disk (load-migration/sanitize) → next persist must full-rewrite. */
	#rewriteRequired = false;
	/** Lazy gate crossed (ensureOnDisk / loaded file): every entry must persist from now on. */
	#forceFileCreation = false;
	/**
	 * Armed only when this manager observed a draft sidecar lifecycle that
	 * materialized an otherwise metadata-only session file. Explicit
	 * ensureOnDisk() callers (ACP session/new, handoff) must survive close().
	 */
	#draftOnlySessionCleanupArmed = false;

	/**
	 * Collab replication tap: invoked for every appended entry with the
	 * in-memory (pre-blob-externalization) entry, so inline images survive.
	 */
	onEntryAppended?: (entry: SessionEntry) => void;

	/**
	 * Invalidate derived source views on journal/ancestry mutation. This is not a
	 * durability or snapshot-publication event: listeners must only invalidate;
	 * read the final source state after the owning mutator returns.
	 */
	subscribeSourceChanges(callback: () => void): () => void {
		this.#sourceChangeCallbacks.add(callback);
		return () => { this.#sourceChangeCallbacks.delete(callback); };
	}

	#notifySourceChanged(): void {
		if (this.#sourceChangeCallbacks.size === 0) return;
		for (const callback of [...this.#sourceChangeCallbacks]) {
			try { callback(); }
			catch (error) { logger.warn("Session source change listener failed", { error: String(error) }); }
		}
	}

	#turnBudgetTotal: number | null = null;
	#turnBudgetHard = false;
	#turnOutputBaseline = 0;
	#turnEvalOutput = 0;

	/** The single open append writer; the manager only ever writes one file at a time. */
	#writer: SessionStorageWriter | undefined;
	/** Sealed by {@link releaseRetainedEntries}: every later append/title/rewrite is a dropped no-op. */
	#released = false;
	/** Serializes async disk work (flush/close/atomic rewrite). Appends are synchronous and bypass it. */
	#diskTail: Promise<void> = Promise.resolve();
	#diskFailure: Error | undefined;
	#diskFailureLogged = false;
	/** FIFO reservation for atomic batches and authoritative recovery. */
	#atomicPersistenceTail: Promise<void> = Promise.resolve();
	/** Observer notifications withheld until their entries are proven durable. */
	#pendingDurabilityNotifications: SessionEntry[] = [];
	/** Bumped on every sync rewrite / chain reset so stale queued tasks become no-ops. */
	#diskEpoch = 0;
	/**
	 * Epoch of the in-flight atomic rewrite, or `null` when no rewrite is running.
	 * The fence in {@link #appendToSessionFile} only applies while this matches
	 * `#diskEpoch`: once a synchronous rewrite (`flushSync` → `#rewriteSynchronously`)
	 * bumps the epoch, the pending atomic publish is guaranteed to abandon via
	 * its `commitGuard`, and appends can safely take the hot path against the
	 * freshly-published file.
	 */
	#atomicRewriteFenceEpoch: number | null = null;
	/** Set by synchronous appends that land while an atomic replacement is active. */
	#atomicRewriteDirty = false;
	/**
	 * Active {@link moveTo} relocation. Concurrent completed appends write a
	 * full body to the live path: source while it still exists, destination
	 * once rename has landed (source gone). Never recreates a vacated source.
	 * `null` outside an active relocation.
	 */
	#sessionFileRelocating: { source: string; dest: string } | null = null;
	/** Atomic entry batch currently staged for publication. */
	#atomicEntryBatch: AtomicEntryBatch | undefined;

	#artifactManager: ArtifactManager | null = null;
	#artifactManagerSessionFile: string | null = null;
	#adoptedArtifactManager: ArtifactManager | null = null;
	#inMemoryArtifacts: Map<string, string> | null = null;
	#inMemoryArtifactCounter = 0;

	#suppressBreadcrumb = false;
	/**
	 * The last breadcrumb this manager wrote marked a lazy fresh session whose
	 * JSONL is not yet on disk. Cleared (and the crumb re-stamped non-fresh) once
	 * the session materializes, so a materialized-then-deleted session still falls
	 * back to the most-recent session instead of being treated as a fresh crumb.
	 */
	#breadcrumbFresh = false;
	#sessionNameChangedCallbacks = new Set<() => void>();
	#persistenceErrorCallbacks = new Set<(error: Error) => void>();

	private constructor(cwd: string, sessionDir: string, persist: boolean, storage: SessionStorage) {
		this.#cwd = cwd;
		this.#sessionDir = sessionDir;
		this.#persist = persist;
		this.#storage = storage;
		this.#blobs = new BlobStore(getBlobsDir());

		if (persist && sessionDir) this.#storage.ensureDirSync(sessionDir);
	}

	#rememberBreadcrumb(cwd: string, sessionFile: string, fresh = false): void {
		this.#breadcrumbFresh = fresh;
		if (!this.#suppressBreadcrumb) writeTerminalBreadcrumb(cwd, sessionFile, fresh);
	}

	/**
	 * Re-stamp a fresh-session breadcrumb as non-fresh once the session has
	 * materialized on disk. A no-op unless the current breadcrumb is still fresh.
	 */
	#materializeBreadcrumb(): void {
		if (!this.#breadcrumbFresh || !this.#sessionFile) return;
		this.#rememberBreadcrumb(this.#cwd, this.#sessionFile, false);
	}

	#clearDiskError(): void {
		this.#diskFailure = undefined;
		this.#diskFailureLogged = false;
	}

	#noteDiskFailure(errorLike: unknown): Error {
		const error = toError(errorLike);
		if (!this.#diskFailure) this.#diskFailure = error;

		if (!this.#diskFailureLogged) {
			this.#diskFailureLogged = true;
			logger.error("Session persistence error.", {
				sessionFile: this.#sessionFile,
				error: error.message,
				stack: error.stack,
			});
			for (const callback of this.#persistenceErrorCallbacks) {
				try {
					callback(error);
				} catch (callbackError) {
					logger.warn("Session persistence error observer failed", {
						error: toError(callbackError).message,
					});
				}
			}
		}

		return this.#diskFailure;
	}

	#scheduleDiskWork(work: () => Promise<void>, options: DiskQueueOptions = {}): Promise<void> {
		const epoch = options.epoch ?? this.#diskEpoch;
		const scheduled = this.#diskTail
			.catch(() => undefined)
			.then(async () => {
				if (!options.ignoreEpoch && epoch !== this.#diskEpoch) return;
				if (this.#diskFailure && !options.ignorePriorError) throw this.#diskFailure;
				await work();
			});

		const reported = scheduled.catch(err => {
			throw this.#noteDiskFailure(err);
		});
		this.#diskTail = reported.catch(() => undefined);
		return reported;
	}

	async #withAtomicPersistenceLock<T>(operation: () => Promise<T>): Promise<T> {
		const predecessor = this.#atomicPersistenceTail;
		const turn = Promise.withResolvers<void>();
		this.#atomicPersistenceTail = predecessor.catch(() => undefined).then(() => turn.promise);
		await predecessor.catch(() => undefined);
		try {
			return await operation();
		} finally {
			turn.resolve();
		}
	}

	async #drainAndCloseWriter(): Promise<void> {
		try {
			await this.#scheduleDiskWork(
				async () => {
					await this.#closeWriterHandle();
				},
				{ ignorePriorError: true, ignoreEpoch: true },
			);
		} finally {
			this.#writer = undefined;
			this.#diskTail = Promise.resolve();
		}
	}

	#closeWriterEventually(): void {
		const writer = this.#writer;
		this.#writer = undefined;
		if (writer) void writer.close().catch(() => undefined);
	}

	async #closeWriterHandle(): Promise<void> {
		const writer = this.#writer;
		if (!writer) return;
		this.#writer = undefined;
		await writer.close();
	}

	#latchIndeterminate(operationError: Error, recoveryErrors: readonly Error[]): SessionPersistenceIndeterminateError {
		const error = new SessionPersistenceIndeterminateError(operationError, recoveryErrors);
		this.#diskFailure = error;
		if (!this.#diskFailureLogged) {
			this.#diskFailureLogged = true;
			logger.error("Session persistence became indeterminate.", {
				sessionFile: this.#sessionFile,
				error: error.message,
			});
		}
		return error;
	}

	#notifyDurableEntries(entries: readonly SessionEntry[] = []): void {
		const notifications = [...this.#pendingDurabilityNotifications, ...entries];
		this.#pendingDurabilityNotifications = [];
		const seen = new Set<string>();
		for (const entry of notifications) {
			if (seen.has(entry.id)) continue;
			seen.add(entry.id);
			this.#notifyEntryAppended(entry);
		}
	}

	async #authoritativelyRewriteCurrentStateLocked(operationError: Error): Promise<void> {
		if (this.#released) {
			// Terminal seal: repair would reset the disk tail (escaping the
			// close() serialization) and atomically publish #fileBody() — after
			// release that truncates, and a revival may already own the file.
			// The original operation error still propagates to the caller.
			logger.warn("Skipped authoritative session repair after terminal release", {
				error: String(operationError),
			});
			return;
		}
		if (!this.#persist || !this.#sessionFile) return;
		const previousDiskTail = this.#diskTail;
		const writer = this.#writer;
		this.#diskEpoch++;
		const epoch = this.#diskEpoch;
		this.#writer = undefined;
		this.#diskTail = Promise.resolve();
		this.#forceFileCreation = true;
		this.#fileIsCurrent = false;
		this.#rewriteRequired = true;
		this.#atomicRewriteFenceEpoch = epoch;
		if (!this.#diskFailure) this.#diskFailure = operationError;
		try {
			await previousDiskTail.catch(() => undefined);
			let closeError: Error | undefined;
			if (writer) {
				try {
					await writer.close();
				} catch (error) {
					closeError = toError(error);
				}
			}
			let drainError: Error | undefined;
			try {
				await this.#storage.drain();
			} catch (error) {
				drainError = toError(error);
			}
			if (writer?.isOpen()) {
				throw this.#latchIndeterminate(operationError, [
					closeError ?? new Error("Failed to close session writer before authoritative repair."),
					...(drainError ? [drainError] : []),
				]);
			}

			do {
				this.#atomicRewriteDirty = false;
				const sessionFile = this.#sessionFile;
				if (!sessionFile) {
					throw this.#latchIndeterminate(operationError, [
						new Error("Session file disappeared during authoritative repair."),
					]);
				}
				const body = this.#fileBody();
				try {
					await this.#storage.writeTextAtomic(sessionFile, body, {
						commitGuard: () => !this.#released && this.#diskEpoch === epoch,
					});
				} catch (error) {
					const recoveryErrors = [toError(error)];
					try {
						await this.#storage.drain();
					} catch (drainFailure) {
						recoveryErrors.push(toError(drainFailure));
					}
					let actual: string;
					try {
						actual = await this.#storage.readText(sessionFile);
					} catch (readFailure) {
						recoveryErrors.push(toError(readFailure));
						throw this.#latchIndeterminate(operationError, recoveryErrors);
					}
					if (actual !== body) {
						recoveryErrors.push(new Error("Authoritative session repair did not match durable storage."));
						throw this.#latchIndeterminate(operationError, recoveryErrors);
					}
				}
				if (this.#diskEpoch !== epoch) {
					throw this.#latchIndeterminate(operationError, [
						new Error("Authoritative session repair was superseded before verification."),
					]);
				}
			} while (this.#atomicRewriteDirty);

			this.#fileIsCurrent = true;
			this.#rewriteRequired = false;
			this.#hasTitleSlot = true;
			this.#clearDiskError();
		} catch (error) {
			if (error instanceof SessionPersistenceIndeterminateError) throw error;
			throw this.#latchIndeterminate(operationError, [toError(error)]);
		} finally {
			if (this.#atomicRewriteFenceEpoch === epoch) this.#atomicRewriteFenceEpoch = null;
		}
	}

	#appendWriter(): SessionStorageWriter {
		if (!this.#sessionFile) throw new Error("Cannot open a session writer before a session file exists");

		if (this.#writer?.isOpen()) return this.#writer;

		this.#writer = this.#storage.openWriter(this.#sessionFile, {
			flags: "a",
			onError: err => this.#noteDiskFailure(err),
		});
		return this.#writer;
	}

	#lineFor(entry: FileEntry): string {
		return `${stringifyJson(prepareEntryForPersistence(entry, this.#blobs)) ?? "null"}\n`;
	}

	#titleSlotLine(): string {
		return serializeTitleSlot({
			title: this.#sessionName,
			source: this.#titleSource,
			updatedAt: this.#titleUpdatedAt || this.#header.timestamp,
		});
	}

	#fileBody(): string {
		let body = this.#titleSlotLine();
		body += this.#lineFor(this.#header);
		for (const entry of this.#entries) body += this.#lineFor(entry);
		return body;
	}

	#historyContainsAssistantMessage(): boolean {
		return this.#entries.some(isAssistantEntry);
	}

	#shouldHaveSessionFile(): boolean {
		return this.#forceFileCreation || this.#fileIsCurrent || this.#historyContainsAssistantMessage();
	}

	/**
	 * Live path for concurrent completed appends during {@link moveTo}.
	 * Prefers destination once rename has landed (source gone); otherwise
	 * source. Never invents a path that does not already exist.
	 */
	#liveRelocationWritePath(): string | null {
		const relocating = this.#sessionFileRelocating;
		if (!relocating) return null;
		if (this.#storage.existsSync(relocating.dest)) return relocating.dest;
		if (this.#storage.existsSync(relocating.source)) return relocating.source;
		// Rename in flight with neither path visible (rare cross-device edge):
		// fall back to destination so we do not recreate a vacated source.
		return relocating.dest;
	}

	/**
	 * Synchronously rewrite the whole file (header + entries) and keep no open
	 * writer; the next append re-opens one. `writeTextSync` returns with the
	 * bytes in the kernel page cache, so the file is software-crash durable.
	 *
	 * During {@link moveTo}, writes to the live relocation path (source pre-
	 * rename, destination post-rename) rather than always `#sessionFile`, so
	 * concurrent completed entries are durable without recreating a vacated source.
	 */
	#rewriteSynchronously(): void {
		if (this.#released) return;
		if (!this.#persist || !this.#shouldHaveSessionFile()) return;
		const targetPath = this.#liveRelocationWritePath() ?? this.#sessionFile;
		if (!targetPath) return;

		try {
			const body = this.#fileBody();
			this.#diskEpoch++;
			this.#diskTail = Promise.resolve();
			this.#closeWriterEventually();
			this.#storage.writeTextSync(targetPath, body);
			this.#clearDiskError();
			// Only mark the manager current when writing the active session path.
			// Mid-move writes update the live relocation path; `#sessionFile` is
			// still the pre-repoint source until moveTo repoints it.
			if (!this.#sessionFileRelocating || targetPath === this.#sessionFile) {
				this.#fileIsCurrent = true;
				this.#materializeBreadcrumb();
				this.#rewriteRequired = false;
				this.#hasTitleSlot = true;
			} else {
				// Destination body is current on disk; in-memory still needs a
				// header-cwd rewrite after repoint, but entries are durable.
				this.#fileIsCurrent = false;
				this.#rewriteRequired = true;
				this.#hasTitleSlot = true;
			}
		} catch (err) {
			this.#noteDiskFailure(err);
		}
	}

	/**
	 * Publish atomically (temp-write + rename, EPERM-safe) on the disk chain.
	 * An append-only batch supplies its durable prefix length; other callers
	 * rewrite the whole file. Serialization happens after the writer is closed.
	 * The fence is enabled BEFORE `#closeWriterHandle()` and stays active until the last
	 * atomic publish returns, so a sync append landing in the close-yield window
	 * cannot open a fresh writer that the pending replacement would then detach
	 * from the current JSONL path. A `commitGuard` also prevents a superseding
	 * synchronous rewrite from being overwritten by the stale body serialized
	 * before it ran.
	 */
	async #rewriteAtomically(appendFrom?: number): Promise<void> {
		if (!this.#persist || !this.#sessionFile) return;
		if (this.#released) return;

		const startEpoch = this.#diskEpoch;
		await this.#scheduleDiskWork(
			async () => {
				if (await this.#runFencedAtomicRewrite(startEpoch, appendFrom)) {
					this.#fileIsCurrent = true;
					this.#materializeBreadcrumb();
					this.#rewriteRequired = false;
					this.#hasTitleSlot = true;
				}
			},
			{ epoch: startEpoch },
		);
	}

	/**
	 * Shared fenced atomic-rewrite loop used by `#rewriteAtomically` and the
	 * `#persistTitleChangeEntry` fallback. Holds the epoch fence across writer
	 * close and publication, and loops on `#atomicRewriteDirty` so a fenced
	 * append landing during publication is captured before the task resolves.
	 * Returns `false` when the disk epoch
	 * moved (a superseding synchronous rewrite has taken over) so callers skip
	 * their post-publish state updates.
	 */
	async #runFencedAtomicRewrite(epoch: number, appendFrom?: number): Promise<boolean> {
		if (this.#released) return false;
		this.#atomicRewriteFenceEpoch = epoch;
		try {
			do {
				this.#atomicRewriteDirty = false;
				await this.#closeWriterHandle();
				const sessionFile = this.#sessionFile;
				if (!sessionFile) return false;
				if (this.#diskEpoch !== epoch) return false;
				const options = { commitGuard: () => !this.#released && this.#diskEpoch === epoch };
				if (appendFrom === undefined) {
					await this.#storage.writeTextAtomic(sessionFile, this.#fileBody(), options);
				} else {
					// Snapshot only the unpublished tail. Entries arriving during storage
					// publication dirty the fence and belong to the next suffix, not this one.
					const end = this.#entries.length;
					if (appendFrom < end) {
						let suffix = "";
						for (let i = appendFrom; i < end; i++) suffix += this.#lineFor(this.#entries[i]);
						await this.#storage.appendTextAtomic(sessionFile, suffix, options);
						appendFrom = end;
					}
				}
				if (this.#diskEpoch !== epoch) return false;
			} while (this.#atomicRewriteDirty);
			return true;
		} finally {
			// Only relinquish the fence if we still own it. A superseding
			// synchronous rewrite (`flushSync` → `#rewriteSynchronously`) may
			// have reset `#diskTail`, scheduled a fresh atomic task at the new
			// epoch, and that task may have taken ownership of the fence while
			// this stale rewrite was still awaiting storage. Clearing it here
			// unconditionally would strand appends during the newer publish.
			if (this.#atomicRewriteFenceEpoch === epoch) this.#atomicRewriteFenceEpoch = null;
		}
	}

	#appendToSessionFile(entry: SessionEntry): void {
		if (this.#released || !this.#persist || !this.#sessionFile) return;
		if (this.#atomicEntryBatch) {
			this.#fileIsCurrent = false;
			this.#rewriteRequired = true;
			this.#atomicRewriteDirty = true;
			return;
		}
		if (this.#diskFailure) {
			// The failed entry and any later entries remain in memory. A full
			// replacement is the writability probe and restores all of them once
			// transient storage pressure clears.
			this.#fileIsCurrent = false;
			this.#rewriteRequired = true;
		}

		// Lazy gate: a brand-new session is not written until it has an assistant
		// message (or someone forced creation), so sessions that never produce
		// output never create a file.
		if (!this.#shouldHaveSessionFile()) {
			this.#fileIsCurrent = false;
			return;
		}

		// Atomic replacement / move window: do not open a fresh append writer that
		// a Windows EPERM replace could detach from the current JSONL path.
		// - moveTo: write a full body to the live relocation path (source pre-
		//   rename, destination post-rename) so completed entries are durable
		//   without recreating a vacated source.
		// - in-place atomic fence: supersede the pending publish with a
		//   synchronous full-body rewrite; bumping `#diskEpoch` abandons the
		//   in-flight atomic via its `commitGuard`.
		if (this.#sessionFileRelocating) {
			this.#rewriteSynchronously();
			return;
		}
		if (this.#atomicRewriteFenceEpoch !== null && this.#atomicRewriteFenceEpoch === this.#diskEpoch) {
			this.#atomicRewriteDirty = true;
			this.#rewriteSynchronously();
			return;
		}
		// Cold/divergent: not on disk yet, or in-memory entries diverged from the
		// file → rewrite the whole file synchronously and keep going.
		if (!this.#fileIsCurrent || this.#rewriteRequired) {
			this.#rewriteSynchronously();
			return;
		}

		// Hot path: write the entry directly on the writer, outside the async disk
		// chain. Prefer appendSync so write failures latch `#diskFailure` before
		// this call returns (not via a discarded rejected Promise after a later
		// microtask). Callers stay non-throwing here — the core turn loop invokes
		// appendMessage/appendCustomEntry without try/catch. A later entry retries
		// all in-memory state through a full rewrite. File writers apply each line
		// to the OS page cache before return.
		// A mid-close writer leaves `#writer` undefined, so `#appendWriter` simply
		// opens a fresh append handle and the entry still lands.
		try {
			const writer = this.#appendWriter();
			const line = this.#lineFor(entry);
			if (writer.appendSync) {
				writer.appendSync(line);
			} else {
				void writer.append(line).catch(err => {
					this.#fileIsCurrent = false;
					this.#rewriteRequired = true;
					this.#noteDiskFailure(err);
				});
			}
		} catch (err) {
			this.#fileIsCurrent = false;
			this.#rewriteRequired = true;
			this.#noteDiskFailure(err);
		}
	}

	async #persistTitleChangeEntry(entry: TitleChangeEntry, update: SessionTitleUpdate): Promise<void> {
		if (!this.#persist || !this.#sessionFile) return;
		if (this.#diskFailure) {
			this.#fileIsCurrent = false;
			this.#rewriteRequired = true;
			this.#rewriteSynchronously();
			if (this.#diskFailure) throw this.#diskFailure;
			return;
		}

		if (!this.#shouldHaveSessionFile()) {
			this.#fileIsCurrent = false;
			return;
		}

		// Title changes use their own asynchronous append path rather than
		// #appendToSessionFile. During move, write the full body (including the
		// title entry) to the live relocation path so a crash mid-move still
		// keeps the title change; the trailing rewrite still updates header cwd.
		if (this.#sessionFileRelocating) {
			this.#rewriteSynchronously();
			return;
		}

		if (
			!this.#fileIsCurrent ||
			this.#rewriteRequired ||
			!this.#hasTitleSlot ||
			!this.#storage.existsSync(this.#sessionFile)
		) {
			await this.#rewriteAtomically();
			return;
		}

		const epoch = this.#diskEpoch;
		const line = this.#lineFor(entry);
		await this.#scheduleDiskWork(
			async () => {
				if (this.#released) return;
				const sessionFile = this.#sessionFile;
				if (!sessionFile) return;
				try {
					await this.#appendWriter().append(line);
					await this.#storage.updateSessionTitle(sessionFile, update);
					if (this.#diskEpoch === epoch) this.#fileIsCurrent = true;
				} catch {
					if (!(await this.#runFencedAtomicRewrite(epoch))) return;
					this.#clearDiskError();
					this.#fileIsCurrent = true;
					this.#rewriteRequired = false;
					this.#hasTitleSlot = true;
				}
			},
			{ epoch },
		);
	}

	#notifyEntryAppended(entry: SessionEntry): void {
		const callback = this.onEntryAppended;
		if (callback) {
			try {
				callback(entry);
			} catch (err) {
				logger.warn("collab entry hook failed", { error: String(err) });
			}
		}
	}

	#resetToNewSession(options?: NewSessionOptions, forcedSessionFile?: string): string | undefined {
		this.#diskTail = Promise.resolve();
		this.#clearDiskError();
		this.#reconcileSessionDirForFallback();
		this.#sessionId = mintSessionId();
		this.#sessionName = undefined;
		this.#titleSource = undefined;
		this.#titleUpdatedAt = "";
		this.#hasTitleSlot = true;

		const timestamp = nowIso();
		this.#header = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: this.#sessionId,
			timestamp,
			cwd: this.#cwd,
			parentSession: options?.parentSession,
			providerPromptCacheKey: options?.providerPromptCacheKey,
		};
		const workspace = normalizeSessionWorkspace({
			cwd: this.#cwd,
			directories: options?.additionalDirectories ?? [],
		});
		this.#additionalDirectories = additionalWorkspaceDirectories(workspace);
		if (this.#additionalDirectories.length > 0) {
			this.#header.additionalDirectories = [...this.#additionalDirectories];
		}
		this.#titleUpdatedAt = timestamp;

		this.#entries = [];
		this.#index.clear();
		this.#requirementsActive = undefined;
		this.#requirementsEpochCache = undefined;
		this.#requirementsDependencyJournals = undefined;
		this.#requirementsDependencyBlobs = undefined;
		this.#fileIsCurrent = false;
		this.#rewriteRequired = false;
		this.#forceFileCreation = false;
		this.#draftOnlySessionCleanupArmed = false;
		this.#turnBudgetTotal = null;
		this.#turnBudgetHard = false;
		this.#turnOutputBaseline = 0;
		this.#turnEvalOutput = 0;
		this.#artifactManager = null;
		this.#artifactManagerSessionFile = null;
		this.#adoptedArtifactManager = null;
		this.#inMemoryArtifacts = null;
		this.#inMemoryArtifactCounter = 0;

		if (this.#persist) {
			this.#sessionFile =
				forcedSessionFile ??
				path.join(this.#sessionDir, `${fileSafeTimestamp(timestamp)}_${this.#sessionId}.jsonl`);
			this.#rememberBreadcrumb(this.#cwd, this.#sessionFile, true);
		} else {
			this.#sessionFile = undefined;
		}

		return this.#sessionFile;
	}

	#applyEntries(header: SessionHeader, entries: SessionEntry[]): void {
		this.#requirementsActive = undefined;
		this.#requirementsEpochCache = undefined;
		this.#requirementsDependencyJournals = undefined;
		this.#requirementsDependencyBlobs = undefined;
		this.#requirementsSourceRewriteVersion++;
		this.#header = header;
		this.#entries = entries;
		this.#sessionId = header.id;
		this.#sessionName = header.title;
		this.#titleSource = header.titleSource;
		this.#titleUpdatedAt = header.timestamp;
		this.#index.rebuild(entries);
	}

	#freshEntryFields(): { id: string; parentId: string | null; timestamp: string } {
		return {
			id: generateId(this.#index),
			parentId: this.#index.leafId(),
			timestamp: nowIso(),
		};
	}

	#setLeaf(id: string | null): void {
		if (this.#index.leafId() !== id) this.#requirementsSourceRewriteVersion++;
		this.#index.setLeaf(id);
		const batch = this.#atomicEntryBatch;
		if (batch && !batch.collecting) {
			batch.externalLeafChanged = true;
			batch.externalLeafId = id;
		}
	}

	#recordEntry(entry: SessionEntry): void {
		if (this.#released) {
			logger.warn("Dropped session entry appended after terminal release", { type: entry.type });
			return;
		}
		if (entry.type === "reset_boundary" || entry.type === "message" || entry.type === "custom_message" ||
			(entry.type === "custom" && entry.customType === REQUIREMENTS_OPERATOR_DECISION_ENTRY))
			this.#requirementsSourceRewriteVersion++;
		this.#entries.push(entry);
		this.#index.insert(entry);
		const batch = this.#atomicEntryBatch;
		if (batch?.collecting) batch.entryIds.add(entry.id);
		if (batch && !batch.collecting) {
			batch.externalLeafChanged = true;
			batch.externalLeafId = entry.id;
		}
		this.#appendToSessionFile(entry);
		if (batch) batch.deferredNotifications.push(entry);
		else this.#notifyEntryAppended(entry);
	}

	#rollbackAtomicEntryBatch(batch: AtomicEntryBatch): void {
		const retainedAncestor = (id: string | null): string | null => {
			const seen = new Set<string>();
			while (id && batch.entryIds.has(id) && !seen.has(id)) {
				seen.add(id);
				id = this.#index.get(id)?.parentId ?? null;
			}
			return id;
		};
		const retained = this.#entries.filter(entry => !batch.entryIds.has(entry.id));
		for (const entry of retained) entry.parentId = retainedAncestor(entry.parentId);
		const restoredLeaf = retainedAncestor(batch.externalLeafChanged ? batch.externalLeafId : batch.preBatchLeafId);
		this.#entries = retained;
		this.#index.rebuild(retained);
		this.#index.setLeaf(restoredLeaf && this.#index.has(restoredLeaf) ? restoredLeaf : null);
	}

	#draftPath(): string | null {
		const artifactsDir = this.getArtifactsDir();
		return artifactsDir ? path.join(artifactsDir, "draft.txt") : null;
	}

	#draftOnlySessionMarkerPath(): string | null {
		const artifactsDir = this.getArtifactsDir();
		return artifactsDir ? path.join(artifactsDir, DRAFT_ONLY_SESSION_MARKER) : null;
	}

	#hasDraftOnlySessionMarker(): boolean {
		const markerPath = this.#draftOnlySessionMarkerPath();
		return markerPath !== null && this.#storage.existsSync(markerPath);
	}

	async #writeDraftOnlySessionMarker(): Promise<void> {
		const markerPath = this.#draftOnlySessionMarkerPath();
		if (!markerPath) return;
		await this.#storage.writeText(markerPath, "");
	}

	async #clearDraftOnlySessionMarker(): Promise<void> {
		const markerPath = this.#draftOnlySessionMarkerPath();
		if (!markerPath) return;
		try {
			await this.#storage.unlink(markerPath);
		} catch (err) {
			if (!isEnoent(err)) throw err;
		}
	}

	#artifactManagerForSession(): ArtifactManager | null {
		if (this.#adoptedArtifactManager) return this.#adoptedArtifactManager;

		const sessionFile = this.#sessionFile;
		if (!sessionFile) {
			this.#artifactManager = null;
			this.#artifactManagerSessionFile = null;
			return null;
		}

		if (this.#artifactManager && this.#artifactManagerSessionFile === sessionFile) return this.#artifactManager;

		this.#artifactManager = new ArtifactManager(sessionFile.slice(0, -JSONL_SUFFIX_LENGTH));
		this.#artifactManagerSessionFile = sessionFile;
		return this.#artifactManager;
	}

	#notifySessionNameListeners(): void {
		for (const callback of Array.from(this.#sessionNameChangedCallbacks)) {
			try {
				callback();
			} catch (err) {
				logger.warn("SessionManager: session name change hook failed", { error: String(err) });
			}
		}
	}

	static #cleanTitle(raw: string): string {
		return raw
			.replace(/[\u0000-\u001f\u007f-\u009f]/g, " ")
			.replace(/ +/g, " ")
			.trim();
	}

	/** Puts a binary blob into the blob store and returns the blob reference. */
	async putBlob(data: Buffer, options?: BlobPutOptions): Promise<BlobPutResult> {
		return this.#blobs.put(data, options);
	}

	/** Synchronous variant of {@link putBlob} for rebuild-only render paths. */
	putBlobSync(data: Buffer, options?: BlobPutOptions): BlobPutResult {
		return this.#blobs.putSync(data, options);
	}

	captureState(): SessionManagerStateSnapshot {
		return {
			cwd: this.#cwd,
			sessionDir: this.#sessionDir,
			sessionId: this.#sessionId,
			sessionName: this.#sessionName,
			titleSource: this.#titleSource,
			titleUpdatedAt: this.#titleUpdatedAt,
			hasTitleSlot: this.#hasTitleSlot,
			sessionFile: this.#sessionFile,
			onDisk: this.#fileIsCurrent,
			needsRewrite: this.#rewriteRequired,
			draftOnlySessionCleanupArmed: this.#draftOnlySessionCleanupArmed,
			fallbackRuntimeOnly: this.#fallbackRuntimeOnly,
			// Entries are snapshotted by reference (switch/reload replaces the
			// array wholesale). The header is cloned: moveTo mutates it in place
			// (cwd, additionalDirectories), so a by-reference capture would let
			// a rollback observe the move it is undoing.
			header: structuredClone(this.#header),
			entries: [...this.#entries],
		};
	}

	/**
	 * Create an independent manager for the current logical session and branch.
	 * The clone shares the storage backend but owns its entry index and writer, so
	 * callers can finish session-owned work after this manager switches elsewhere.
	 * Set `persist` false when the original session is intentionally being dropped.
	 */
	cloneCurrentSession(options?: { persist?: boolean }): SessionManager {
		const persist = options?.persist ?? this.#persist;
		const clone = new SessionManager(this.#cwd, this.#sessionDir, persist, this.#storage);
		clone.#suppressBreadcrumb = true;
		clone.restoreState(this.captureState());
		if (!persist) {
			clone.#sessionFile = undefined;
			clone.#fileIsCurrent = false;
			clone.#rewriteRequired = false;
			clone.#forceFileCreation = false;
		}
		return clone;
	}

	restoreState(snapshot: SessionManagerStateSnapshot): void {
		this.#closeWriterEventually();
		this.#diskTail = Promise.resolve();
		this.#clearDiskError();

		this.#cwd = snapshot.cwd;
		this.#sessionDir = snapshot.sessionDir;
		this.#sessionFile = snapshot.sessionFile;
		this.#fileIsCurrent = snapshot.onDisk;
		this.#rewriteRequired = snapshot.needsRewrite;
		this.#forceFileCreation = snapshot.onDisk;
		this.#draftOnlySessionCleanupArmed = snapshot.draftOnlySessionCleanupArmed;
		this.#fallbackRuntimeOnly = snapshot.fallbackRuntimeOnly;
		this.#applyEntries(snapshot.header, [...snapshot.entries]);
		this.#additionalDirectories = snapshot.header.additionalDirectories ?? [];
		this.#sessionName = snapshot.sessionName;

		this.#titleSource = snapshot.titleSource;
		this.#titleUpdatedAt = snapshot.titleUpdatedAt;
		this.#hasTitleSlot = snapshot.hasTitleSlot;
		this.#artifactManager = null;
		this.#artifactManagerSessionFile = null;
		this.#adoptedArtifactManager = null;

		if (this.#sessionFile) this.#rememberBreadcrumb(this.#cwd, this.#sessionFile);
	}

	/**
	 * Undo a {@link moveTo} using a {@link captureState} snapshot: rename the
	 * session and artifacts back into the captured bucket, then restore the
	 * captured metadata (cwd, header, additionalDirectories). The captured
	 * header is persisted after relocation so a fresh open of the source
	 * session sees the pre-move metadata, including workspace roots the move
	 * filtered out. Rollbacks must not re-enter forward-move hooks, so this
	 * bypasses AgentSession entirely. If the rename-back itself fails, the
	 * manager stays pointed at the actual moved file (restoring the snapshot
	 * would split the transcript across a recreated source and the stranded
	 * target) and the error names where the session file actually lives.
	 */
	async rollbackMove(snapshot: SessionManagerStateSnapshot): Promise<void> {
		try {
			const targetSessionDir = snapshot.sessionFile ? path.dirname(snapshot.sessionFile) : snapshot.sessionDir;
			await this.moveTo(snapshot.cwd, targetSessionDir);
		} catch (error) {
			const movedFile = this.getSessionFile();
			throw new Error(
				`could not relocate the session back to ${snapshot.sessionDir} (${error instanceof Error ? error.message : String(error)}); the session file remains at ${movedFile}`,
			);
		}
		this.restoreState(snapshot);
		// The inverse moveTo already rewrote the source file with the
		// target-filtered header. Persist the captured one so disk and memory
		// agree after a fresh open.
		if (this.#persist && this.#sessionFile) {
			this.#forceFileCreation = true;
			this.#rewriteRequired = true;
			await this.#rewriteAtomically();
		}
	}
	/** Switch to a different session file (resume / branch). */
	async setSessionFile(sessionFile: string): Promise<void> {
		await this.#setSessionFile(sessionFile);
	}

	async #setSessionFile(sessionFile: string, loadedSession?: SessionLoadResult): Promise<void> {
		await this.#drainAndCloseWriter();
		this.#clearDiskError();
		this.#draftOnlySessionCleanupArmed = false;

		const resolvedSessionFile = path.resolve(sessionFile);
		const loaded = loadedSession ?? (await loadSessionFile(resolvedSessionFile, this.#storage));
		if (loaded.invalidHeader) {
			throw new Error(
				`Cannot resume session "${resolvedSessionFile}": the session header is missing or malformed. The file was not modified.`,
			);
		}

		this.#sessionFile = resolvedSessionFile;
		this.#rememberBreadcrumb(this.#cwd, resolvedSessionFile);

		const { entries: fileEntries, titleSlot } = loaded;
		if (fileEntries.length === 0) {
			// Explicit but empty/missing path (e.g. --session flag): start fresh but
			// keep the requested path and materialize the header immediately.
			this.#resetToNewSession(undefined, resolvedSessionFile);
			this.#forceFileCreation = true;
			await this.#rewriteAtomically();
			this.#fileIsCurrent = true;
			return;
		}

		const migrated = migrateToCurrentVersion(fileEntries);
		await resolveBlobRefsInEntries(fileEntries, this.#blobs);
		// loadEntriesFromFile guarantees entries[0] is a valid session header.
		const header = fileEntries[0] as SessionHeader;

		// Adopt the loaded session's working directory only when it is verifiably
		// accessible. Sessions live in a dir keyed by their cwd, so resuming a
		// session from another project must re-point cwd/sessionDir at that
		// project — but a deleted OR permission-blocked directory (macOS TCC
		// denial) must not be adopted: callers without a cwd-change callback
		// (extension UI, RPC) would otherwise track a directory the process
		// cannot enter. Keep the current cwd so the session stays where the
		// user already is.
		const headerCwd = header.cwd ? path.resolve(header.cwd) : undefined;
		if (headerCwd && headerCwd !== path.resolve(this.#cwd) && (await directoryIsEnterable(headerCwd))) {
			this.#cwd = headerCwd;
			this.#sessionDir = path.dirname(resolvedSessionFile);
			this.#fallbackRuntimeOnly = false;
			this.#rememberBreadcrumb(this.#cwd, resolvedSessionFile);
		} else if (headerCwd && headerCwd !== path.resolve(this.#cwd)) {
			// Header cwd not enterable: keep runtime cwd but mark fallback
			// so workspace changes stay runtime-only until the transcript
			// is relocated.
			this.#fallbackRuntimeOnly = true;
		} else {
			this.#fallbackRuntimeOnly = false;
		}

		this.#applyEntries(header, fileEntries.slice(1) as SessionEntry[]);
		this.#additionalDirectories = header.additionalDirectories ?? [];
		this.#titleUpdatedAt = titleSlot?.updatedAt ?? header.timestamp;
		this.#hasTitleSlot = titleSlot !== undefined;
		this.#fileIsCurrent = true;
		this.#rewriteRequired = migrated || loaded.malformedRecords > 0;
		this.#forceFileCreation = true;
		this.#artifactManager = null;
		this.#artifactManagerSessionFile = null;

		if (this.sanitizeLoadedOpenAIResponsesReplayMetadata()) this.#rewriteRequired = true;
	}

	/**
	 * Start a new session and persist its header before returning.
	 *
	 * The durable empty boundary prevents a later process on another terminal
	 * from selecting the previous conversation as the most recent session.
	 */
	async newSession(options?: NewSessionOptions): Promise<string | undefined> {
		await this.#drainAndCloseWriter();
		const sessionFile = this.#resetToNewSession(options);
		await this.ensureOnDisk();
		return sessionFile;
	}

	/** Delete a session file and its artifact directory. ENOENT is treated as success. */
	async dropSession(sessionPath: string): Promise<void> {
		await this.#drainAndCloseWriter();
		try {
			await this.#storage.deleteSessionWithArtifacts(sessionPath);
		} catch (err) {
			if (!isEnoent(err)) throw err;
		}
	}

	/**
	 * Fork the current session into a new file with the same entries.
	 * @returns the old and new session file paths, or undefined when not persisting.
	 */
	async fork(): Promise<{ oldSessionFile: string; newSessionFile: string } | undefined> {
		if (!this.#persist || !this.#sessionFile) return undefined;

		const oldSessionFile = this.#sessionFile;
		const parentSessionId = this.#sessionId;
		SessionManager.#preserveSourceOrigins(this.#entries, parentSessionId);
		await this.#drainAndCloseWriter();
		this.#clearDiskError();
		this.#reconcileSessionDirForFallback();

		const timestamp = nowIso();
		this.#sessionId = mintSessionId();
		this.#sessionFile = path.join(this.#sessionDir, `${fileSafeTimestamp(timestamp)}_${this.#sessionId}.jsonl`);
		this.#header = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: this.#sessionId,
			title: this.#header.title ?? this.#sessionName,
			titleSource: this.#header.titleSource ?? this.#titleSource,
			timestamp,
			cwd: this.#cwd,
			additionalDirectories: this.#additionalDirectories.length > 0 ? [...this.#additionalDirectories] : undefined,
			parentSession: parentSessionId,
			providerPromptCacheKey: this.#header.providerPromptCacheKey ?? parentSessionId,
		};
		this.#sessionName = this.#header.title;
		this.#titleSource = this.#header.titleSource;
		this.#titleUpdatedAt = timestamp;
		this.#hasTitleSlot = true;
		this.#fileIsCurrent = false;
		this.#rewriteRequired = false;
		this.#forceFileCreation = true;
		this.#draftOnlySessionCleanupArmed = false;
		this.#artifactManager = null;
		this.#artifactManagerSessionFile = null;
		this.#rememberBreadcrumb(this.#cwd, this.#sessionFile);

		await this.#rewriteAtomically();
		return { oldSessionFile, newSessionFile: this.#sessionFile };
	}

	/** Move the session to a new working directory. */
	async moveTo(newCwd: string, targetSessionDir?: string): Promise<void> {
		const resolvedCwd = path.resolve(newCwd);
		const resolvedTargetDir = targetSessionDir ? path.resolve(targetSessionDir) : undefined;
		const managedRoot = resolveManagedSessionRoot(this.#sessionDir, this.#cwd);
		const nextSessionDir =
			resolvedTargetDir ??
			(managedRoot
				? computeDefaultSessionDir(resolvedCwd, this.#storage, managedRoot)
				: computeDefaultSessionDir(resolvedCwd, this.#storage));
		const expectedSessionFile = this.#sessionFile
			? path.join(nextSessionDir, path.basename(this.#sessionFile))
			: undefined;
		if (
			resolvedCwd === path.resolve(this.#cwd) &&
			!this.#fallbackRuntimeOnly &&
			(!resolvedTargetDir || resolvedTargetDir === path.resolve(this.#sessionDir)) &&
			(!expectedSessionFile || path.resolve(this.#sessionFile!) === path.resolve(expectedSessionFile))
		) {
			return;
		}

		let sessionFileExisted = false;
		// Track source+dest for concurrent completed appends during relocation
		// (see `#sessionFileRelocating`). Existence of either path decides the
		// live write target — not a `#diskEpoch` bump, which would cancel any
		// disk task already queued at the current epoch (e.g. a header-only
		// `ensureOnDisk()` materializing rewrite) before the drain below runs it.
		if (this.#persist && this.#sessionFile) {
			const source = this.#sessionFile;
			const dest = path.join(nextSessionDir, path.basename(source));
			this.#sessionFileRelocating = { source, dest };
		}

		try {
			if (this.#persist && this.#sessionFile) {
				this.#storage.ensureDirSync(nextSessionDir);
				await this.#drainAndCloseWriter();
				this.#clearDiskError();

				const oldSessionFile = this.#sessionFile;
				const newSessionFile = path.join(nextSessionDir, path.basename(oldSessionFile));
				const oldArtifactsDir = artifactsDirectoryFor(oldSessionFile);
				const newArtifactsDir = artifactsDirectoryFor(newSessionFile);
				const sessionPathChanged = path.resolve(oldSessionFile) !== path.resolve(newSessionFile);
				const artifactPathChanged =
					oldArtifactsDir !== null &&
					newArtifactsDir !== null &&
					path.resolve(oldArtifactsDir) !== path.resolve(newArtifactsDir);
				sessionFileExisted = this.#storage.existsSync(oldSessionFile);

				let sessionMoved = false;
				let artifactsMoved = false;

				try {
					if (sessionFileExisted && sessionPathChanged) {
						await fs.promises.rename(oldSessionFile, newSessionFile);
						sessionMoved = true;
					}

					if (artifactPathChanged) {
						try {
							const artifactStat = await fs.promises.stat(oldArtifactsDir);
							if (artifactStat.isDirectory()) {
								await fs.promises.rename(oldArtifactsDir, newArtifactsDir);
								artifactsMoved = true;
							}
						} catch (err) {
							if (!isEnoent(err)) throw err;
						}
					}
				} catch (err) {
					if (artifactsMoved && oldArtifactsDir && newArtifactsDir) {
						try {
							await fs.promises.rename(newArtifactsDir, oldArtifactsDir);
						} catch (rollbackErr) {
							throw new Error(
								`Failed to move artifacts and rollback: ${rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)}`,
							);
						}
					}

					if (sessionMoved) {
						try {
							await fs.promises.rename(newSessionFile, oldSessionFile);
						} catch (rollbackErr) {
							throw new Error(
								`Failed to move session file and rollback: ${rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)}`,
							);
						}
					}

					throw err;
				}

				if (sessionFileExisted && sessionPathChanged) {
					this.#header.previousSessionFiles = [
						...new Set([...(this.#header.previousSessionFiles ?? []), path.resolve(oldSessionFile)]),
					];
				}

				this.#sessionFile = newSessionFile;
				this.#artifactManager = null;
				this.#artifactManagerSessionFile = null;
				// Path is repointed; hot-path appends may use `#sessionFile` again.
				this.#sessionFileRelocating = null;
			}

			this.#cwd = resolvedCwd;
			this.#sessionDir = nextSessionDir;
			this.#header.cwd = resolvedCwd;
			// Clear only after the rename has landed. If the move threw,
			// keep the flag so the next relocation retries.
			this.#fallbackRuntimeOnly = false;
			if (this.#additionalDirectories.length === 0) {
				this.#header.additionalDirectories = undefined;
			} else {
				// Re-filter additional roots: the new cwd may have been an
				// additional root, or it may now contain one.
				this.#additionalDirectories = this.#additionalDirectories.filter(d => d !== resolvedCwd);
				this.#header.additionalDirectories =
					this.#additionalDirectories.length > 0 ? this.#additionalDirectories : undefined;
			}

			// Rewrite at the new location when the file already existed (update cwd) or
			// there is in-memory output worth materializing; otherwise stay lazy.
			const hasAssistant = this.#historyContainsAssistantMessage();
			if (this.#persist && this.#sessionFile && (sessionFileExisted || hasAssistant)) {
				this.#forceFileCreation = true;
				await this.#rewriteAtomically();
			}

			if (this.#sessionFile) this.#rememberBreadcrumb(resolvedCwd, this.#sessionFile);
		} finally {
			this.#sessionFileRelocating = null;
		}
	}

	/**
	 * Force the session onto disk even with no assistant message yet (ACP
	 * session/new must create a discoverable file immediately).
	 */
	async ensureOnDisk(): Promise<void> {
		if (!this.#persist || !this.#sessionFile) return;
		this.#forceFileCreation = true;
		if (this.#fileIsCurrent && !this.#rewriteRequired) return;
		await this.#rewriteAtomically();
	}

	/** Persist this session's transcript as a newly identified OMP session. */
	async persistCopy(
		options?: { sessionDir?: string; suppressBreadcrumb?: boolean },
		storage: SessionStorage = new FileSessionStorage(),
	): Promise<SessionManager> {
		const sessionDir = options?.sessionDir ?? SessionManager.getDefaultSessionDir(this.#cwd, undefined, storage);
		const manager = new SessionManager(this.#cwd, sessionDir, true, storage);
		manager.#suppressBreadcrumb = options?.suppressBreadcrumb === true;
		manager.#resetToNewSession();
		manager.#sessionName = this.#sessionName;
		manager.#titleSource = this.#titleSource;
		manager.#titleUpdatedAt = this.#titleUpdatedAt;
		manager.#header.title = this.#sessionName;
		manager.#header.titleSource = this.#titleSource;
		manager.#additionalDirectories = [...this.#additionalDirectories];
		manager.#header.additionalDirectories =
			manager.#additionalDirectories.length > 0 ? [...manager.#additionalDirectories] : undefined;
		manager.#entries = structuredClone(this.#entries);
		SessionManager.#preserveSourceOrigins(manager.#entries, this.#sessionId);
		manager.#index.rebuild(manager.#entries);
		manager.#forceFileCreation = true;
		await manager.#rewriteAtomically();
		return manager;
	}

	/**
	 * Stage a synchronous group of entry appends and publish the whole batch as
	 * one atomic suffix without reserializing durable history. A failed publish
	 * removes only the staged entries, preserves/reparents concurrent appends,
	 * restores the prior durable file view, and clears the failed writer latch
	 * for retry.
	 * The callback MUST be synchronous.
	 */
	appendEntriesAtomically<T>(append: () => T): Promise<T> {
		return this.#withAtomicPersistenceLock(() => this.#appendEntriesAtomicallyLocked(append));
	}

	async #appendEntriesAtomicallyLocked<T>(append: () => T): Promise<T> {
		if (!this.#persist || !this.#sessionFile) return append();
		if (this.#atomicEntryBatch) throw new Error("Atomic persistence lock ownership was violated.");
		try {
			await this.ensureOnDisk();
			await this.flush();
		} catch (error) {
			const operationError = toError(error);
			await this.#authoritativelyRewriteCurrentStateLocked(operationError);
			this.#notifyDurableEntries();
			throw error;
		}

		// All entries through this cursor are durable, including appends during flush.
		const appendFrom = this.#entries.length;
		const batch: AtomicEntryBatch = {
			collecting: true,
			entryIds: new Set(),
			deferredNotifications: [],
			preBatchLeafId: this.#index.leafId(),
			externalLeafChanged: false,
			externalLeafId: null,
		};
		this.#atomicEntryBatch = batch;
		let result!: T;
		try {
			try {
				result = append();
			} finally {
				batch.collecting = false;
			}
			await this.#rewriteAtomically(appendFrom);
			if (!this.#fileIsCurrent || this.#rewriteRequired) {
				throw new Error("Atomic session batch was superseded before commit.");
			}
			this.#atomicEntryBatch = undefined;
			this.#notifyDurableEntries(batch.deferredNotifications);
			return result;
		} catch (error) {
			batch.collecting = false;
			const operationError = toError(error);
			this.#rollbackAtomicEntryBatch(batch);
			try {
				await this.#authoritativelyRewriteCurrentStateLocked(operationError);
			} catch (repairError) {
				const retainedNotifications = batch.deferredNotifications.filter(entry => !batch.entryIds.has(entry.id));
				this.#pendingDurabilityNotifications.push(...retainedNotifications);
				this.#atomicEntryBatch = undefined;
				this.#fileIsCurrent = false;
				this.#rewriteRequired = true;
				if (repairError instanceof SessionPersistenceIndeterminateError) throw repairError;
				throw this.#latchIndeterminate(operationError, [toError(repairError)]);
			}
			const retainedNotifications = batch.deferredNotifications.filter(entry => !batch.entryIds.has(entry.id));
			this.#atomicEntryBatch = undefined;
			this.#notifyDurableEntries(retainedNotifications);
			throw error;
		}
	}

	/**
	 * Replace an uncertain append tail with the authoritative in-memory journal.
	 * Callers must only use this for monotonic recovery where every retained
	 * entry remains intended (for example, an explicit terminal tombstone).
	 */
	recoverPersistenceFromCurrentState(): Promise<void> {
		return this.#withAtomicPersistenceLock(async () => {
			if (!this.#persist || !this.#sessionFile) return;
			if (this.#atomicEntryBatch) throw new Error("Atomic persistence lock ownership was violated.");
			const operationError =
				this.#diskFailure ?? new Error("Authoritative session persistence recovery was requested.");
			await this.#authoritativelyRewriteCurrentStateLocked(operationError);
			this.#notifyDurableEntries();
		});
	}

	/** Flush pending writes. Call before switching sessions or on shutdown. */
	async flush(): Promise<void> {
		if (!this.#persist || !this.#sessionFile) return;
		await this.#scheduleDiskWork(async () => {
			if (this.#writer?.isOpen()) await this.#writer.flush();
		});
		// Drain any fire-and-forget backing writes (e.g. `writeTextSync` queued
		// on IndexedSessionStorage during `flushSync`) so callers relying on
		// flush() see the write durably visible to readers.
		await this.#storage.drain();
		if (this.#diskFailure) throw this.#diskFailure;
	}

	/**
	 * Synchronously makes the current append-only session durable. Avoid rewriting
	 * an already-current file: large restored sessions can contain GiB of compacted
	 * history, and Ctrl+C must not rebuild the whole JSONL string just to flush.
	 */
	flushSync(): void {
		if (!this.#persist || !this.#sessionFile) return;
		if (this.#atomicEntryBatch) throw new Error("Cannot synchronously flush during an atomic session batch.");
		if (this.#diskFailure) throw this.#diskFailure;
		if (this.#fileIsCurrent && !this.#rewriteRequired) {
			this.#writer?.flushSync?.();
			const writerError = this.#writer?.getError();
			if (writerError) throw writerError;
			return;
		}
		this.#rewriteSynchronously();
		if (this.#diskFailure) throw this.#diskFailure;
	}

	/**
	 * Drop only session files that this manager saw materialized for a draft and
	 * that still contain no durable conversation or extension state. Explicit
	 * ensureOnDisk() records (ACP session/new, handoff) stay resumable.
	 */
	async #dropIfEmptyAndNoDraft(): Promise<void> {
		if (!this.#draftOnlySessionCleanupArmed) return;
		const sessionFile = this.#sessionFile;
		if (!sessionFile || !this.#storage.existsSync(sessionFile)) {
			this.#draftOnlySessionCleanupArmed = false;
			return;
		}
		const draftPath = this.#draftPath();
		if (draftPath && this.#storage.existsSync(draftPath)) return;
		if (!this.#entries.every(isDraftOnlyMetadataEntry)) {
			await this.#clearDraftOnlySessionMarker();
			this.#draftOnlySessionCleanupArmed = false;
			return;
		}
		try {
			await this.#storage.deleteSessionWithArtifacts(sessionFile);
			this.#fileIsCurrent = false;
			this.#forceFileCreation = false;
			this.#hasTitleSlot = false;
			this.#draftOnlySessionCleanupArmed = false;
		} catch (err) {
			if (!isEnoent(err)) {
				logger.warn("Failed to drop empty session on close", { sessionFile, error: String(err) });
			}
		}
	}

	/** Flush, then close the append writer. */
	async close(): Promise<void> {
		if (!this.#persist) return;
		await this.#scheduleDiskWork(async () => {
			const hadWriter = this.#writer !== undefined;
			await this.#closeWriterHandle();
			if (hadWriter || (this.#sessionFile && this.#storage.existsSync(this.#sessionFile)))
				this.#fileIsCurrent = true;
		});
		await this.#dropIfEmptyAndNoDraft();
		// Wait for any queued backing writes (IndexedSessionStorage per-path
		// tail) to become durable so a graceful shutdown does not exit while
		// a fire-and-forget publish is still on the wire.
		await this.#storage.drain();
		if (this.#diskFailure) throw this.#diskFailure;
	}

	/**
	 * Raise the terminal write barrier ahead of the final {@link close}. Once
	 * sealed:
	 * - every later append, title change, and rewrite is a dropped no-op —
	 *   including work an event handler tries to enqueue while dispose is
	 *   awaiting `close()` on the disk tail;
	 * - the disk epoch is bumped, so queued-but-unexecuted tail work is
	 *   superseded and an ALREADY-RUNNING fenced/repair rewrite (awaiting the
	 *   tail, drain, writer close, or the atomic stage) fails its commit guard
	 *   at the rename fence instead of publishing over a revived file.
	 * The final `close()` itself is scheduled after the bump and still runs;
	 * pre-seal hot-path appends are already in the page cache. Idempotent;
	 * terminal.
	 */
	seal(): void {
		if (this.#released) return;
		this.#released = true;
		this.#diskEpoch++;
	}

	/**
	 * Terminal release: drop the in-memory transcript and complete the
	 * {@link seal}. The entry journal and its index mirror the agent's message
	 * array (tool results, file contents, base64 frame images); on a disposed
	 * session — e.g. a parked subagent still referenced by the lifecycle
	 * adoption record — they would otherwise stay pinned for the process
	 * lifetime.
	 *
	 * Closes the append writer; with the seal up, nothing can reopen it. A
	 * revival may reopen the same JSONL through a NEW manager the moment
	 * dispose returns; a late event handler resuming on THIS manager must
	 * never race that writer — and a post-release rewrite would persist the
	 * now-empty entry list, truncating the transcript. Reads after this point
	 * reopen from disk (revival, `history://`). Only call from session
	 * dispose, after the final `close()`; idempotent.
	 */
	releaseRetainedEntries(): void {
		this.seal();
		this.#entries = [];
		this.#index.clear();
		this.#closeWriterEventually();
	}

	getCwd(): string {
		return this.#cwd;
	}

	/** Recorded cwd from the session header (original project), may differ from runtime {@link getCwd} when fallback retained launch cwd. */
	getRecordedCwd(): string | undefined {
		return this.#header?.cwd;
	}

	setCwdWithoutRelocation(newCwd: string): void {
		const resolvedCwd = path.resolve(newCwd);
		if (resolvedCwd === path.resolve(this.#cwd)) {
			this.#fallbackRuntimeOnly = true;
			return;
		}
		this.#cwd = resolvedCwd;
		this.#fallbackRuntimeOnly = true;
		if (this.#sessionFile) {
			this.#rememberBreadcrumb(resolvedCwd, this.#sessionFile);
		}
	}
	adoptRecordedCwd(): void {
		const recordedCwd = this.#header.cwd;
		if (!recordedCwd) return;
		this.#cwd = path.resolve(recordedCwd);
		if (this.#sessionFile) this.#sessionDir = path.dirname(this.#sessionFile);
		this.#fallbackRuntimeOnly = false;
		if (this.#sessionFile) this.#rememberBreadcrumb(this.#cwd, this.#sessionFile);
	}

	/**
	 * Re-anchor the session bucket to the runtime cwd after a fallback.
	 * The fallback flag keeps the transcript at its recorded path (stale
	 * bucket) while runtime cwd is the launch dir; only a true relocation
	 * should recompute sessionDir. Workspace-dir mutations must not clear
	 * it early.
	 */
	#reconcileSessionDirForFallback(): void {
		if (this.#fallbackRuntimeOnly) {
			this.#sessionDir = computeDefaultSessionDir(this.#cwd, this.#storage);
			this.#fallbackRuntimeOnly = false;
		}
	}

	/** Additional workspace directories beyond cwd (multi-root), absolute and normalized. */
	getAdditionalDirectories(): string[] {
		return [...this.#additionalDirectories];
	}

	/**
	 * Persist a workspace-directory change to the session header. Respects the
	 * lazy-persistence gate: a session with no durable output yet keeps the
	 * change in memory (the header lands with the first real write), so seeding
	 * roots at launch never materializes an empty resumable session file.
	 */
	async #persistWorkspaceDirectoriesChange(): Promise<void> {
		if (!this.#persist || !this.#sessionFile || !this.#shouldHaveSessionFile()) return;
		this.#rewriteRequired = true;
		await this.#rewriteAtomically();
	}

	/**
	 * Add a workspace directory. Normalizes (relative to cwd), dedupes, rejects
	 * the cwd itself, persists to the session header, and triggers an atomic
	 * rewrite so the change survives a crash. Returns the resolved absolute
	 * path or `null` when the directory was already present (no-op).
	 */
	async addWorkspaceDirectory(directory: string): Promise<string | null> {
		const resolved = normalizeWorkspaceDirectory(directory, this.#cwd);
		if (resolved === path.resolve(this.#cwd)) {
			throw new Error("The current working directory is already the primary workspace root.");
		}
		if (this.#additionalDirectories.includes(resolved)) return null;
		this.#additionalDirectories = [...this.#additionalDirectories, resolved];
		// In fallback the transcript is still in the stale bucket; keep
		// workspace edits runtime-only until relocation.
		if (this.#fallbackRuntimeOnly) {
			return resolved;
		}
		this.#header.additionalDirectories = this.#additionalDirectories;
		await this.#persistWorkspaceDirectoriesChange();
		return resolved;
	}

	/**
	 * Remove a workspace directory by absolute or cwd-relative path. Persists
	 * the trimmed header. Returns the resolved path that was removed, or
	 * `null` when the directory was not an additional root (no-op).
	 */
	async removeWorkspaceDirectory(directory: string): Promise<string | null> {
		const resolved = normalizeWorkspaceDirectory(directory, this.#cwd);
		const idx = this.#additionalDirectories.findIndex(p => path.resolve(p) === resolved);
		if (idx === -1) return null;
		this.#additionalDirectories = this.#additionalDirectories.filter((_, i) => i !== idx);
		// In fallback keep edits runtime-only until relocation.
		if (this.#fallbackRuntimeOnly) {
			return resolved;
		}
		if (this.#additionalDirectories.length === 0) {
			this.#header.additionalDirectories = undefined;
		} else {
			this.#header.additionalDirectories = this.#additionalDirectories;
		}
		await this.#persistWorkspaceDirectoriesChange();
		return resolved;
	}

	/** Seed additional directories from settings or a passed list. Also called on resumed sessions with --add-dir; persists the updated header when the session file is already durable. No-op when the normalized list is unchanged (avoids rewriting large session files on every startup). */
	async setAdditionalDirectories(directories: string[]): Promise<void> {
		const workspace = normalizeSessionWorkspace({ cwd: this.#cwd, directories });
		const next = additionalWorkspaceDirectories(workspace);
		// In fallback keep edits runtime-only until relocation.
		if (this.#fallbackRuntimeOnly) {
			this.#additionalDirectories = next;
			return;
		}
		if (
			next.length === this.#additionalDirectories.length &&
			next.every((d, i) => d === this.#additionalDirectories[i])
		) {
			return;
		}
		this.#additionalDirectories = next;
		if (this.#additionalDirectories.length > 0) {
			this.#header.additionalDirectories = this.#additionalDirectories;
		} else {
			this.#header.additionalDirectories = undefined;
		}
		await this.#persistWorkspaceDirectoriesChange();
	}

	getUsageStatistics(): UsageStatistics {
		return this.#index.usageSnapshot();
	}

	/** Cumulative top-level assistant spend, excluding task and background model usage. */
	getAssistantUsageStatistics(): UsageStatistics {
		return this.#index.assistantUsageSnapshot();
	}

	/**
	 * Open a new per-turn budget window: snapshot the cumulative output baseline,
	 * reset the eval-subagent counter, and set the (optional) ceiling.
	 */
	beginTurnBudget(total: number | null, hard: boolean): void {
		this.#turnBudgetTotal = total;
		this.#turnBudgetHard = hard;
		this.#turnOutputBaseline = this.#index.usageSnapshot().output;
		this.#turnEvalOutput = 0;
	}

	recordEvalSubagentOutput(output: number): void {
		if (Number.isFinite(output) && output > 0) this.#turnEvalOutput += output;
	}

	getTurnBudget(): { total: number | null; spent: number; hard: boolean } {
		const mainOutput = Math.max(0, this.#index.usageSnapshot().output - this.#turnOutputBaseline);
		return { total: this.#turnBudgetTotal, spent: mainOutput + this.#turnEvalOutput, hard: this.#turnBudgetHard };
	}

	getSessionDir(): string {
		return this.#sessionDir;
	}

	getSessionId(): string {
		return this.#sessionId;
	}

	getSessionFile(): string | undefined {
		return this.#sessionFile;
	}

	/**
	 * Whether the current session has actually been materialized to durable
	 * storage (the JSONL exists on disk / in the active storage backend).
	 *
	 * Session persistence is lazy: the file is only written once the history
	 * contains an assistant message (or an explicit {@link ensureOnDisk}
	 * caller forces it). Until then {@link getSessionFile} returns an allocated
	 * path that leads nowhere, so a `--resume <id>` hint built from it would
	 * always fail. Consumers that advertise a resume command must gate on this
	 * (issue #8860).
	 */
	isSessionOnDisk(): boolean {
		return !!this.#sessionFile && this.#storage.existsSync(this.#sessionFile);
	}

	getArtifactsDir(): string | null {
		if (this.#adoptedArtifactManager) return this.#adoptedArtifactManager.dir;
		return artifactsDirectoryFor(this.#sessionFile);
	}

	adoptArtifactManager(manager: ArtifactManager): void {
		this.#adoptedArtifactManager = manager;
	}

	getArtifactManager(): ArtifactManager | null {
		return this.#artifactManagerForSession();
	}

	async allocateArtifactPath(toolType: string): Promise<{ id?: string; path?: string }> {
		return (await this.#artifactManagerForSession()?.allocatePath(toolType)) ?? {};
	}

	async saveArtifact(content: string, toolType: string): Promise<string | undefined> {
		const manager = this.#artifactManagerForSession();
		if (manager) return manager.save(content, toolType);

		// Non-persistent session: keep an in-memory copy so spill truncation works.
		this.#inMemoryArtifacts ??= new Map();
		const id = String(this.#inMemoryArtifactCounter++);
		this.#inMemoryArtifacts.set(id, content);
		return id;
	}

	async getArtifactPath(id: string): Promise<string | null> {
		return (await this.#artifactManagerForSession()?.getPath(id)) ?? null;
	}

	async saveDraft(text: string): Promise<void> {
		const draftPath = this.#draftPath();
		if (!draftPath || !this.#persist) return;

		if (text.length === 0) {
			try {
				await this.#storage.unlink(draftPath);
			} catch (err) {
				if (!isEnoent(err)) throw err;
			}
			return;
		}

		const sessionFile = this.#sessionFile;
		const draftWillMaterializeMetadataOnlyFile =
			sessionFile !== undefined &&
			!this.#storage.existsSync(sessionFile) &&
			this.#entries.every(isDraftOnlyMetadataEntry);
		// Force the header onto disk so resume can find the file this draft attaches to.
		await this.ensureOnDisk();
		if (draftWillMaterializeMetadataOnlyFile) {
			await this.#writeDraftOnlySessionMarker();
			this.#draftOnlySessionCleanupArmed = true;
		}
		await this.#storage.writeText(draftPath, text);
	}

	async consumeDraft(): Promise<string | null> {
		const draftPath = this.#draftPath();
		if (!draftPath) return null;

		let draft: string;
		try {
			draft = await this.#storage.readText(draftPath);
		} catch (err) {
			if (isEnoent(err)) return null;
			throw err;
		}

		try {
			await this.#storage.unlink(draftPath);
		} catch (err) {
			if (!isEnoent(err)) throw err;
		}
		if (this.#entries.every(isDraftOnlyMetadataEntry) && this.#hasDraftOnlySessionMarker())
			this.#draftOnlySessionCleanupArmed = true;

		return draft;
	}

	/** The source that set the session name: "user" (manual/RPC) or "auto" (generated title). */
	get titleSource(): SessionTitleSource | undefined {
		return this.#titleSource;
	}

	getSessionName(): string | undefined {
		return this.#sessionName;
	}

	onSessionNameChanged(cb: () => void): () => void {
		this.#sessionNameChangedCallbacks.add(cb);
		return () => {
			this.#sessionNameChangedCallbacks.delete(cb);
		};
	}

	/** Subscribe to persistence failures so hosts can surface lost-durability state. */
	onPersistenceError(cb: (error: Error) => void): () => void {
		this.#persistenceErrorCallbacks.add(cb);
		return () => {
			this.#persistenceErrorCallbacks.delete(cb);
		};
	}

	/**
	 * Set the session display name.
	 * @param source "user" for explicit renames; "auto" for generated titles.
	 *   Auto titles are ignored once the user has set a name.
	 */
	async setSessionName(name: string, source: SessionTitleSource = "auto", trigger?: string): Promise<boolean> {
		if (this.#released) return false;
		if (this.#titleSource === "user" && source === "auto") return false;

		const title = SessionManager.#cleanTitle(name);
		if (!title) return false;

		const previousTitle = this.#sessionName;
		const timestamp = nowIso();
		this.#sessionName = title;
		this.#titleSource = source;
		this.#titleUpdatedAt = timestamp;
		this.#header.title = title;
		this.#header.titleSource = source;

		const entry: TitleChangeEntry = {
			type: TITLE_CHANGE_ENTRY_TYPE,
			...this.#freshEntryFields(),
			timestamp,
			title,
			source,
		};
		if (previousTitle) entry.previousTitle = previousTitle;
		if (trigger) entry.trigger = trigger;
		this.#entries.push(entry);
		this.#index.insert(entry);
		this.#notifyEntryAppended(entry);
		await this.#persistTitleChangeEntry(entry, { title, source, updatedAt: timestamp });
		// Keep the recent-sessions title index current so welcome-screen lookups
		// never have to content-scan this session's file.
		if (this.#persist && this.#storage instanceof FileSessionStorage) {
			recordSessionTitle(this.#sessionId, title);
		}

		this.#notifySessionNameListeners();
		return true;
	}

	/**
	 * Append a foreign (host-authored) entry verbatim, preserving its
	 * `id`/`parentId`. Used by collab guests to mirror the host session.
	 */
	ingestReplicatedEntry(entry: SessionEntry): void {
		this.#recordEntry(entry);
	}

	/**
	 * Snapshot the session for collab replication: the live header plus a deep
	 * copy of every entry (the host mutates entries in place on rewrite paths, so
	 * guests must not share references).
	 */
	snapshotForReplication(): { header: SessionHeader; entries: SessionEntry[] } {
		return { header: structuredClone(this.#header), entries: structuredClone(this.#entries) as SessionEntry[] };
	}

	/**
	 * Append a message as a child of the current leaf, then advance the leaf.
	 * CompactionSummaryMessage / BranchSummaryMessage are rejected here — they are
	 * top-level entries via appendCompaction()/branchWithSummary().
	 */
	appendMessage(
		message:
			| Message
			| CustomMessage
			| HookMessage
			| BashExecutionMessage
			| PythonExecutionMessage
			| FileMentionMessage,
		options?: { compactionOverride?: "keep" | "exclude"; sourceOrigin?: SessionMessageEntry["sourceOrigin"] },
	): string {
		if ((message.role === "user" || message.role === "custom") && message.originalSubmission) {
			const original = message.originalSubmission;
			const content = message.content;
			const images = original.images;
			const sameContent = typeof content === "string"
				? content === original.text && !images?.length
				: content.length === 1 + (images?.length ?? 0) && content[0]?.type === "text" && content[0].text === original.text &&
					(!images || images.every((image, index) => {
						const delivered = content[index + 1];
						return delivered?.type === "image" && delivered.data === image.data && delivered.mimeType === image.mimeType;
					}));
			const deliveredLinks = message.imageLinks;
			const sameLinks = original.imageLinks === deliveredLinks ||
				(original.imageLinks?.length === deliveredLinks?.length &&
					original.imageLinks?.every((link, index) => link === deliveredLinks?.[index]));
			if (sameContent && sameLinks && original.compactionOverride === message.compactionOverride) {
				const { originalSubmission: _original, ...delivered } = message;
				message = delivered;
			}
		}
		const entry: SessionMessageEntry = { type: "message", ...this.#freshEntryFields(), message };
		entry.sourceOrigin = options?.sourceOrigin ? { ...options.sourceOrigin } : { journalId: this.#sessionId, entryId: entry.id };
		if ((message.role === "user" || message.role === "custom") && options?.compactionOverride) {
			entry.compactionOverride = options.compactionOverride;
		}
		this.#recordEntry(entry);
		return entry.id;
	}

	static #preserveSourceOrigins(entries: SessionEntry[], journalId: string): void {
		for (const entry of entries) {
			if (entry.type === "message" || entry.type === "custom_message")
				entry.sourceOrigin ??= { journalId, entryId: entry.id };
		}
	}

	/** Explicit authored deletion invalidates the exact entry, never remaps frozen evidence. */
	invalidateRequirementsSources(entryIds: readonly string[]): void {
		if (!entryIds.length) return;
		this.#requirementsSourceRewriteVersion++;
		this.#rewriteRequired = true;
		this.#fileIsCurrent = false;
		for (const id of entryIds) {
			const entry = this.getEntry(id);
			if (entry?.type === "message" || entry?.type === "custom_message") entry.requirementsInvalidated = true;
		}
	}

	getRequirementsSourceVersion(): string {
		const versions = [...(this.#requirementsDependencyJournals?.values() ?? [])].map(locator => this.#requirementsJournalVersion(locator));
		const blobs = [...(this.#requirementsDependencyBlobs ?? [])].map(hash => this.#blobs.getVersion(hash));
		return JSON.stringify([this.#sessionId, this.#requirementsSourceRewriteVersion, versions, blobs]);
	}

	#requirementsJournalVersion(locator: RequirementsSource["locators"][number]): string | null {
		if (!locator.journalPath) return locator.sessionId === this.#sessionId ? String(this.#requirementsSourceRewriteVersion) : null;
		try {
			const storage = this.#storage.existsSync(locator.journalPath) ? this.#storage : new FileSessionStorage();
			const stat = storage.statSync(locator.journalPath) as { size: number; mtimeMs: number; ctimeMs?: number; ino?: number; dev?: number };
			return JSON.stringify([stat.dev, stat.ino, stat.size, stat.mtimeMs, stat.ctimeMs]);
		} catch (error) { if (isEnoent(error)) return null; throw error; }
	}

	/** Lazy ancestry metadata: no requirements index is allocated while the owner is disabled. */
	getRequirementsEpoch(entryId = this.getLeafId()): number {
		const cache = this.#requirementsEpochCache ??= new Map();
		const path: SessionEntry[] = [];
		const seen = new Set<string>();
		let cursor = entryId ? this.getEntry(entryId) : undefined;
		while (cursor && !cache.has(cursor.id) && !seen.has(cursor.id)) {
			seen.add(cursor.id); path.push(cursor);
			cursor = cursor.parentId ? this.getEntry(cursor.parentId) : undefined;
		}
		let epoch = cursor ? cache.get(cursor.id) ?? 0 : 0;
		for (let index = path.length - 1; index >= 0; index--) {
			if (path[index].type === "reset_boundary") epoch++;
			cache.set(path[index].id, epoch);
		}
		return epoch;
	}

	#requirementsIdentity(entry: SessionMessageEntry | CustomMessageEntry) {
		return entry.sourceOrigin ?? { journalId: this.#sessionId, entryId: entry.id };
	}

	#requirementsMessage(entry: SessionMessageEntry | CustomMessageEntry): AgentMessage {
		return entry.type === "message" ? entry.message : {
			role: "custom", customType: entry.customType, content: entry.content, display: entry.display,
			details: entry.details, attribution: entry.attribution, timestamp: Date.parse(entry.timestamp),
		};
	}

	#requirementsDescriptor(entry: SessionMessageEntry | CustomMessageEntry, epoch: number, parentKey: string | null): RequirementsSource {
		const message = this.#requirementsMessage(entry);
		const producer = "producer" in message ? message.producer : undefined;
		const kind = producer?.type === "generated" ? "unknown" : producer?.type
			?? (message.role === "assistant" ? "assistant" : message.role === "toolResult" ? "tool" : "unknown");
		const authoritative = kind === "human";
		const original = this.#requirementsIdentity(entry);
		return {
			key: JSON.stringify([original.journalId, original.entryId]), original,
			locators: [{ sessionId: this.#sessionId, journalPath: this.#sessionFile, entryId: entry.id }],
			// A catalog token is not a claim that the body has been resolved or verified.
			integrity: requirementsHash(JSON.stringify([original, entry.timestamp, !!entry.requirementsInvalidated])),
			integrityAvailable: false, parentKey, ownerSessionId: this.#sessionId,
			branchId: this.getLeafId() ?? "", epoch,
			origin: { kind, producerId: producer && "toolCallId" in producer ? producer.toolCallId : undefined },
			units: [], durable: this.#persist,
			...(message.role !== "user" && !authoritative ? { referenceOnly: true as const } : {}),
			state: entry.requirementsInvalidated ? "orphaned" : authoritative ? "pending" : "unsupported",
			reason: entry.requirementsInvalidated ? "Original source was explicitly invalidated"
				: authoritative ? undefined : "Original producer is not attested human or SDK input",
		};
	}

	/** Metadata only: accepted delivery needs no capture, original-body read, or journal flush. */
	getRequirementsSource(entryId: string): RequirementsSource | undefined {
		const entry = this.getEntry(entryId);
		if (entry?.type !== "message" && entry?.type !== "custom_message") return undefined;
		const epoch = this.getRequirementsEpoch(entry.id);
		let parentKey: string | null = null;
		let cursor = entry.parentId ? this.getEntry(entry.parentId) : undefined;
		while (cursor && cursor.type !== "reset_boundary") {
			if (cursor.type === "message" && cursor.message.role === "user") {
				const origin = this.#requirementsIdentity(cursor);
				parentKey = JSON.stringify([origin.journalId, origin.entryId]);
				break;
			}
			cursor = cursor.parentId ? this.getEntry(cursor.parentId) : undefined;
		}
		return this.#requirementsDescriptor(entry, epoch, parentKey);
	}

	/** Current branch ancestry by journal index; neither authority inference nor evidence-body reads. */
	isRequirementsSourceApplicable(source: RequirementsSource): boolean {
		const local = source.locators.filter(locator => locator.sessionId === this.#sessionId);
		if (!local.length) return true;
		const leaf = this.getLeafId();
		let active = this.#requirementsActive;
		if (active?.leaf !== leaf) {
			const entry = leaf ? this.getEntry(leaf) : undefined;
			if (active && entry?.parentId === active.leaf && entry.type !== "reset_boundary") {
				active.leaf = leaf; active.ids.add(entry.id);
			} else this.#requirementsActive = active = { leaf, cursor: leaf, ids: new Set() };
		}
		if (!active) return false;
		while (active.cursor && !local.some(locator => active.ids.has(locator.entryId))) {
			const entry = this.getEntry(active.cursor);
			if (!entry || entry.type === "reset_boundary" || active.ids.has(entry.id)) { active.cursor = null; break; }
			active.ids.add(entry.id); active.cursor = entry.parentId;
		}
		return local.some(locator => {
			const entry = this.getEntry(locator.entryId);
			if (!active.ids.has(locator.entryId) || (entry?.type !== "message" && entry?.type !== "custom_message") || entry.requirementsInvalidated) return false;
			const original = this.#requirementsIdentity(entry);
			return JSON.stringify([original.journalId, original.entryId]) === source.key;
		});
	}

	/** Enabled owner only. Traverse ancestry cooperatively; do not touch original bodies. */
	async *iterateRequirementsSources(): AsyncGenerator<{ source: RequirementsSource }, number> {
		const leaf = this.getLeafId();
		const branch: SessionEntry[] = [];
		const seen = new Set<string>();
		let cursor = this.getLeafEntry();
		let epoch = 0;
		let activeEnd: number | undefined;
		while (cursor && !seen.has(cursor.id)) {
			seen.add(cursor.id);
			if (cursor.type === "reset_boundary") { activeEnd ??= branch.length; epoch++; }
			branch.push(cursor);
			cursor = cursor.parentId ? this.getEntry(cursor.parentId) : undefined;
			if ((branch.length & 255) === 0) await Bun.sleep(0);
		}
		let parentKey: string | null = null;
		const epochs = this.#requirementsEpochCache ??= new Map();
		let entryEpoch = 0;
		const activeIds = new Set<string>();
		for (let index = branch.length - 1; index >= 0; index--) {
			const entry = branch[index];
			if (entry.type === "reset_boundary") entryEpoch++;
			epochs.set(entry.id, entryEpoch);
			if (index < (activeEnd ?? branch.length)) {
				activeIds.add(entry.id);
				if (entry.type === "message" || entry.type === "custom_message") {
					const source = this.#requirementsDescriptor(entry, epoch, parentKey);
					source.branchId = leaf ?? "";
					if (!source.referenceOnly) { parentKey = source.key; yield { source }; }
				}
			}
			if ((index & 255) === 0) await Bun.sleep(0);
		}
		if (leaf === this.getLeafId()) this.#requirementsActive = { leaf, cursor: null, ids: activeIds };
		return epoch;
	}

	async getRequirementsSources(): Promise<{ sources: RequirementsSource[]; context: AgentMessage[]; observations: RequirementsObservation[] }> {
		const sources: RequirementsSource[] = [];
		for await (const { source } of this.iterateRequirementsSources()) sources.push(source);
		return { sources, context: [], observations: [] };
	}

	/** Read only the addressed existing journal entry; never open a writable foreign session. */
	async #requirementsEntry(locator: RequirementsSource["locators"][number]): Promise<SessionMessageEntry | CustomMessageEntry | undefined> {
		let entry: SessionEntry | undefined;
		if (!locator.journalPath) {
			if (locator.sessionId === this.#sessionId && !this.#persist) entry = this.getEntry(locator.entryId);
		} else {
			let journalId: string | undefined;
			try {
				await visitEntriesFromFile(locator.journalPath, candidate => {
					if (candidate.type === "session") { journalId = candidate.id; return; }
					if (candidate.id !== locator.entryId) return;
					if (journalId === locator.sessionId) entry = candidate;
					return false;
				}, this.#storage.existsSync(locator.journalPath) ? this.#storage : new FileSessionStorage());
			} catch (error) { if (!isEnoent(error)) throw error; }
		}
		return entry?.type === "message" || entry?.type === "custom_message" ? entry : undefined;
	}

	#requirementsOperatorTargets(entryId: string): string[] | undefined {
		if (!this.#index.childrenOf(entryId).some(entry => entry.type === "custom" && entry.customType === REQUIREMENTS_OPERATOR_DECISION_ENTRY)) return undefined;
		let cursor = this.getLeafEntry();
		while (cursor && cursor.id !== entryId && cursor.type !== "reset_boundary") {
			if (cursor.type === "custom" && cursor.customType === REQUIREMENTS_OPERATOR_DECISION_ENTRY) {
				const marker = cursor.data as { sourceEntryId?: string; targetRevisionIds?: unknown };
				if (marker.sourceEntryId === entryId && Array.isArray(marker.targetRevisionIds) && marker.targetRevisionIds.every(id => typeof id === "string")) return [...marker.targetRevisionIds];
			}
			cursor = cursor.parentId ? this.getEntry(cursor.parentId) : undefined;
		}
		return undefined;
	}

	/** Explicit processing materializes the source chronology; catalog and currentness checks never do. */
	async resolveRequirementsEvidence(key: string, descriptor?: RequirementsSource, options?: { context?: boolean }): Promise<ResolvedRequirementsSource | undefined> {
		if (options?.context === false) return this.#resolveRequirementsUnits(key, descriptor);
		const generation = this.#requirementsSourceRewriteVersion;
		const journalLocator = { sessionId: this.#sessionId, journalPath: this.#sessionFile, entryId: "" };
		const journalVersion = this.#requirementsJournalVersion(journalLocator);
		const resolved = await this.#resolveRequirementsUnits(key, descriptor);
		if (!resolved || resolved.source.referenceOnly) return resolved;
		const locator = resolved.source.locators.find(locator => locator.sessionId === this.#sessionId);
		const entry = locator ? this.getEntry(locator.entryId) : undefined;
		const predecessors: (SessionMessageEntry | CustomMessageEntry)[] = [];
		const visited = new Set<string>();
		let cursor = entry?.parentId ? this.getEntry(entry.parentId) : undefined;
		while (cursor && cursor.type !== "reset_boundary" && !visited.has(cursor.id)) {
			visited.add(cursor.id);
			if (cursor.type === "message" || cursor.type === "custom_message") predecessors.push(cursor);
			cursor = cursor.parentId ? this.getEntry(cursor.parentId) : undefined;
			if ((visited.size & 255) === 0) await Bun.sleep(0);
		}
		let journalEntries: Map<string, SessionMessageEntry | CustomMessageEntry> | undefined;
		if (predecessors.length && journalLocator.journalPath) {
			journalEntries = new Map();
			let journalId: string | undefined;
			try {
				await visitEntriesFromFile(journalLocator.journalPath, candidate => {
					if (candidate.type === "session") { journalId = candidate.id; return; }
					if (journalId === journalLocator.sessionId && visited.has(candidate.id) && (candidate.type === "message" || candidate.type === "custom_message")) journalEntries!.set(candidate.id, candidate);
					if (journalEntries!.size === predecessors.length) return false;
				}, this.#storage.existsSync(journalLocator.journalPath) ? this.#storage : new FileSessionStorage());
			} catch (error) { if (!isEnoent(error)) throw error; }
		}
		const preceding: ResolvedRequirementsSource[] = [];
		const unavailableContext: RequirementsSource[] = [];
		for (let index = predecessors.length - 1; index >= 0; index--) {
			const reference = this.#requirementsDescriptor(predecessors[index], resolved.source.epoch, null);
			const original = await this.#resolveRequirementsUnits(reference.key, reference, journalEntries);
			if (original) preceding.push(original);
			else unavailableContext.push({ ...reference, state: "orphaned", integrityAvailable: false, reason: "Original contextual evidence is unavailable" });
			if ((index & 255) === 0) await Bun.sleep(0);
		}
		if (generation !== this.#requirementsSourceRewriteVersion || journalVersion !== this.#requirementsJournalVersion(journalLocator)) return undefined;
		const context = [...preceding.flatMap(reference => reference.context), ...resolved.context];
		for (const [index, reference] of preceding.entries()) { reference.context = context; reference.contextIndex = index; }
		return { ...resolved, context, contextIndex: context.length - 1, referents: preceding, unavailableContext };
	}

	/** Materialize only the addressed original whole units, using its existing image depot. */
	async #resolveRequirementsUnits(key: string, descriptor?: RequirementsSource,
		journalEntries?: ReadonlyMap<string, SessionMessageEntry | CustomMessageEntry>,
		dependencies?: { journals: Map<string, RequirementsSource["locators"][number]>; blobs: Set<string> }): Promise<ResolvedRequirementsSource | undefined> {
		if (!descriptor) {
			let identity: unknown;
			try { identity = JSON.parse(key); } catch { return undefined; }
			if (!Array.isArray(identity) || identity.length !== 2 || typeof identity[1] !== "string") return undefined;
			descriptor = this.getRequirementsSource(identity[1]);
			if (descriptor?.key !== key) return undefined;
		}
		for (const locator of descriptor.locators) {
			dependencies?.journals.set(JSON.stringify([locator.sessionId, locator.journalPath]), locator);
			const before = this.#requirementsJournalVersion(locator);
			const entry = journalEntries ? journalEntries.get(locator.entryId) : await this.#requirementsEntry(locator);
			if (!entry && before === null && locator.retainedBlobHash) {
				const local = locator.sessionId === this.#sessionId ? this.getEntry(locator.entryId) : undefined;
				if ((local?.type === "message" || local?.type === "custom_message") && local.requirementsInvalidated) continue;
				const hash = locator.retainedBlobHash;
				if (!/^[a-f0-9]{64}$/.test(hash)) continue;
				dependencies?.blobs.add(hash);
				const blobVersion = this.#blobs.getVersion(hash);
				const bytes = this.#requirementsRetainedOriginals?.get(hash) ?? await this.#blobs.get(hash);
				if (!bytes || requirementsHash(bytes) !== hash || blobVersion !== this.#blobs.getVersion(hash)) continue;
				const retained = JSON.parse(bytes.toString()) as { version: number; source: RequirementsSource; units: ResolvedRequirementsSource["units"]; operatorTargetRevisionIds?: string[] };
				if (retained.version !== 1 || retained.source.key !== key || retained.source.integrity !== descriptor.integrity) continue;
				if (before !== this.#requirementsJournalVersion(locator)) continue;
				const content = retained.units.map(unit => unit.image ?? { type: "text" as const, text: unit.text ?? "" });
				return { source: { ...retained.source, locators: descriptor.locators, ownerSessionId: descriptor.ownerSessionId, branchId: descriptor.branchId, epoch: descriptor.epoch },
					units: retained.units, operatorTargetRevisionIds: retained.operatorTargetRevisionIds,
					context: [{ role: "custom", customType: "requirements-retained-original", content, display: false, timestamp: 0 }], contextIndex: 0, referents: [] };
			}
			if (!entry || entry.requirementsInvalidated) continue;
			const original = entry.sourceOrigin ?? { journalId: locator.sessionId, entryId: entry.id };
			if (JSON.stringify([original.journalId, original.entryId]) !== key) continue;
			const frozen = journalEntries ? entry : structuredClone(entry);
			const message = this.#requirementsMessage(frozen);
			if (!("content" in message)) continue;
			const submission = "originalSubmission" in message ? message.originalSubmission : undefined;
			const content = submission ? [{ type: "text" as const, text: submission.text }, ...(submission.images ?? [])]
				: typeof message.content === "string" ? [{ type: "text" as const, text: message.content }] : message.content;
			const unitIds: string[] = [];
			const available = content.filter((part, index): part is TextContent | ImageContent => {
				if (part.type !== "text" && part.type !== "image") return false;
				unitIds.push(String(index)); return true;
			});
			let missing = false;
			for (const part of available) {
				if (part.type !== "image" || !isBlobRef(part.data)) continue;
				const hash = parseBlobRef(part.data);
				if (!hash) { missing = true; break; }
				dependencies?.blobs.add(hash);
				const version = this.#blobs.getVersion(hash);
				const bytes = await this.#blobs.get(hash);
				if (!bytes || requirementsHash(bytes) !== hash || version !== this.#blobs.getVersion(hash)) { missing = true; break; }
				part.data = bytes.toString("base64");
			}
			if (missing) continue;
			if (available.some(part => part.type === "text" ? isPersistenceTruncatedString(part.text) : !part.data || isBlobRef(part.data))) continue;
			const manifest = requirementsUnits(available);
			for (const [index, unit] of manifest.entries()) unit.id = unitIds[index];
			const operatorTargetRevisionIds = locator.sessionId === this.#sessionId ? this.#requirementsOperatorTargets(entry.id) : undefined;
			const source = { ...this.#requirementsDescriptor(frozen, descriptor.epoch, descriptor.parentKey),
				original, key, locators: descriptor.locators, ownerSessionId: descriptor.ownerSessionId,
				branchId: descriptor.branchId, durable: descriptor.durable, integrityAvailable: true,
				integrity: requirementsHash(JSON.stringify([manifest, available.map(part => part.type === "image" ? part.mimeType : null),
					submission?.imageLinks ?? ("imageLinks" in message ? message.imageLinks : undefined),
					submission?.compactionOverride ?? ("compactionOverride" in message ? message.compactionOverride : undefined),
					"producer" in message ? message.producer : undefined, operatorTargetRevisionIds])),
				units: manifest,
			};
			if (!manifest.length || (!source.referenceOnly && content.length !== available.length)) { source.state = "unsupported"; source.reason = "Original contains unsupported content blocks"; }
			const units = available.map((part, index) => part.type === "text" ? { id: unitIds[index], text: part.text } : { id: unitIds[index], image: part });
			if (before !== this.#requirementsJournalVersion(locator)) continue;
			return { source, units, operatorTargetRevisionIds, context: [submission ? { ...message, content: available } as AgentMessage : message], contextIndex: 0, referents: [] };
		}
		return undefined;
	}

	/** Validate actual current dependencies, not a blanket catalog or frozen historical authority. */
	async observeRequirementsEvidence(sources: readonly RequirementsSource[]): Promise<RequirementsObservation[]> {
		const dependencies = { journals: new Map<string, RequirementsSource["locators"][number]>(), blobs: new Set<string>() };
		const version = `${this.#sessionId}:${this.#requirementsSourceRewriteVersion}`;
		const observations: RequirementsObservation[] = [];
		for (const source of sources) {
			const resolved = this.isRequirementsSourceApplicable(source)
				? await this.#resolveRequirementsUnits(source.key, source, undefined, dependencies) : undefined;
			observations.push({ key: source.key, integrity: resolved?.source.integrity ?? null,
				units: resolved?.source.units, locators: resolved?.source.locators });
		}
		const current = version === `${this.#sessionId}:${this.#requirementsSourceRewriteVersion}`;
		if (current) { this.#requirementsDependencyJournals = dependencies.journals; this.#requirementsDependencyBlobs = dependencies.blobs; }
		return current ? observations : sources.map(source => ({ key: source.key, integrity: null }));
	}

	async readRequirementsSource(key: string): Promise<(TextContent | ImageContent)[] | undefined> {
		const resolved = await this.#resolveRequirementsUnits(key);
		return resolved?.units.map(unit => unit.image ?? { type: "text", text: unit.text ?? "" });
	}

	/** Explicit operator retention before journal deletion: freeze only the selected original. */
	async retainRequirementsEvidence(source: RequirementsSource): Promise<RequirementsSource> {
		const versions = source.locators.map(locator => this.#requirementsJournalVersion(locator));
		const resolved = await this.#resolveRequirementsUnits(source.key, source);
		if (!resolved || resolved.source.integrity !== source.integrity)
			throw new Error("Original requirements evidence changed or is unavailable");
		const original = { ...resolved.source, locators: resolved.source.locators.map(({ retainedBlobHash: _retained, ...locator }) => locator) };
		const bytes = Buffer.from(JSON.stringify({ version: 1, source: original, units: resolved.units, operatorTargetRevisionIds: resolved.operatorTargetRevisionIds }));
		const hash = requirementsHash(bytes);
		if (this.#persist) await this.#blobs.put(bytes);
		else (this.#requirementsRetainedOriginals ??= new Map()).set(hash, bytes);
		if (versions.some((version, index) => version !== this.#requirementsJournalVersion(source.locators[index])))
			throw new Error("Original journal changed while retaining requirements evidence");
		return { ...source, locators: source.locators.map(locator => ({ ...locator, retainedBlobHash: hash })) };
	}

	/** An explicit replay is newly accepted input; forks preserve origin on their journal entries. */
	async retainRequirementsSource(key: string): Promise<import("@oh-my-pi/pi-ai").UserMessage> {
		const resolved = await this.#resolveRequirementsUnits(key);
		if (!resolved || (resolved.source.origin.kind !== "human" && resolved.source.origin.kind !== "sdk"))
			throw new Error("Original authoritative requirements source is unavailable");
		return { role: "user", content: resolved.units.map(unit => unit.image ?? { type: "text", text: unit.text ?? "" }),
			producer: { type: "human" }, timestamp: Date.now() };
	}

	/**
	 * Append to a non-active branch without changing the current leaf.
	 * Used by work that retains ownership of a branch across tree navigation.
	 */
	appendMessageToBranch(
		message:
			| Message
			| CustomMessage
			| HookMessage
			| BashExecutionMessage
			| PythonExecutionMessage
			| FileMentionMessage,
		parentId: string | null,
	): string {
		if (parentId !== null && !this.#index.has(parentId)) throw new Error(`Entry ${parentId} not found`);
		const activeLeafId = this.#index.leafId();
		const entry: SessionMessageEntry = {
			type: "message",
			id: generateId(this.#index),
			parentId,
			timestamp: nowIso(),
			message,
		};
		entry.sourceOrigin = { journalId: this.#sessionId, entryId: entry.id };
		this.#recordEntry(entry);
		this.#index.setLeaf(activeLeafId);
		return entry.id;
	}

	/** Record usage on its initiating branch without moving a successor branch or session. */
	appendModelUsage(
		usage: Pick<
			ModelUsageEntry,
			"purpose" | "role" | "api" | "provider" | "model" | "usage" | "stopReason" | "errorMessage"
		>,
		owner: { sessionId: string; parentId: string | null },
	): string | undefined {
		if (this.#sessionId !== owner.sessionId || (owner.parentId !== null && !this.#index.has(owner.parentId))) {
			return undefined;
		}
		const activeLeafId = this.#index.leafId();
		const entry: ModelUsageEntry = {
			type: "model_usage",
			id: generateId(this.#index),
			parentId: owner.parentId,
			timestamp: nowIso(),
			...usage,
		};
		this.#recordEntry(entry);
		if (activeLeafId !== owner.parentId) this.#index.setLeaf(activeLeafId);
		return entry.id;
	}

	/** Append a thinking level change as child of current leaf, then advance leaf. Returns entry id. */
	appendThinkingLevelChange(thinkingLevel?: string, configured?: string): string {
		const entry: ThinkingLevelChangeEntry = {
			type: "thinking_level_change",
			...this.#freshEntryFields(),
			thinkingLevel: thinkingLevel ?? null,
			configured: configured ?? null,
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	appendServiceTierChange(serviceTier: ServiceTierByFamily | null): string {
		const entry: ServiceTierChangeEntry = { type: "service_tier_change", ...this.#freshEntryFields(), serviceTier };
		this.#recordEntry(entry);
		return entry.id;
	}

	appendModeChange(mode: string, data?: Record<string, unknown>): string {
		const entry: ModeChangeEntry = { type: "mode_change", ...this.#freshEntryFields(), mode, data };
		this.#recordEntry(entry);
		return entry.id;
	}

	/**
	 * Append a model change as a child of the current leaf, then advance the leaf.
	 * @param model Model in "provider/modelId" format
	 * @param role Optional role (default: "default")
	 * @param resolvedModelIsFallback Whether this transition selected a retry-fallback model
	 */
	appendModelChange(model: string, role?: string, resolvedModelIsFallback = false): string {
		const entry: ModelChangeEntry = {
			type: "model_change",
			...this.#freshEntryFields(),
			model,
			role,
			resolvedModelIsFallback,
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	appendSessionInit(init: {
		systemPrompt: string;
		task: string;
		tools: string[];
		agent?: string;
		modelRole?: string;
		resolvedModel?: string;
		readOnly?: boolean;
		outputSchema?: unknown;
		outputSchemaMode?: StructuredSubagentSchemaMode;
		restrictToolNames?: boolean;
		spawns?: string;
		readSummarize?: boolean;
		advisor?: string;
	}): string {
		const entry: SessionInitEntry = { type: "session_init", ...this.#freshEntryFields(), ...init };
		this.#recordEntry(entry);
		return entry.id;
	}

	appendCompaction<T = unknown>(
		summary: string,
		shortSummary: string | undefined,
		firstKeptEntryId: string,
		tokensBefore: number,
		options: {
			details?: T;
			fromExtension?: boolean;
			preserveData?: Record<string, unknown>;
			method?: CompactionMethod;
			providerReplayThroughEntryId?: string;
			tokensAfter?: number;
			diagnostics?: CompactionDiagnostics;
		} = {},
	): string {
		const entry: CompactionEntry<T> = {
			type: "compaction",
			...this.#freshEntryFields(),
			summary,
			shortSummary,
			firstKeptEntryId,
			tokensBefore,
			tokensAfter: options.tokensAfter,
			method: options.method,
			providerReplayThroughEntryId: options.providerReplayThroughEntryId,
			details: options.details,
			fromExtension: options.fromExtension,
			preserveData: options.preserveData,
			diagnostics: options.diagnostics,
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	/**
	 * Append the durable conversation boundary recorded by `/clear`. The
	 * collapsed live transcript and the model-context rebuild start after the
	 * latest one, while the full history stays on disk (the plain
	 * `transcript:true` export walks it unchanged).
	 */
	appendResetBoundary(): string {
		const entry: ResetBoundaryEntry = { type: "reset_boundary", ...this.#freshEntryFields() };
		this.#recordEntry(entry);
		return entry.id;
	}

	appendCustomEntry(customType: string, data?: unknown): string {
		const entry: CustomEntry = { type: "custom", customType, data, ...this.#freshEntryFields() };
		this.#recordEntry(entry);
		return entry.id;
	}

	/** Publish source bytes, artifact coverage and classifier validity in one journal rewrite. */
	async rewriteEntries(
		rewrites: readonly SourceRewrite[] = [],
		applySourceRewrites?: () => void,
		onAffected?: (entryIds: readonly string[]) => void,
	): Promise<void> {
		if (this.#released) return;
		if (rewrites.length > 0 && !applySourceRewrites) {
			throw new Error("Source rewrite maps require a synchronous source mutation callback");
		}
		const affected = applySourceRewrites
			? rewriteSessionSources(this.#entries, rewrites, applySourceRewrites, id => this.#index.get(id), id => this.#index.childrenOf(id))
			: [];
		const leaf = this.#index.leafId();
		this.#index.rebuild(this.#entries);
		this.#index.setLeaf(leaf);
		onAffected?.(affected);
		if (!this.#persist || !this.#sessionFile) return;
		await this.#rewriteAtomically();
	}

	/**
	 * Append a custom message entry (for extensions) that participates in LLM context.
	 * @param customType Hook identifier for filtering on reload
	 * @param content Message content (string or TextContent/ImageContent array)
	 * @param display Whether to show in TUI (true = styled display, false = hidden)
	 * @param details Optional extension-specific metadata (not sent to LLM)
	 * @param attribution Who initiated this message for billing/attribution semantics
	 */
	appendCustomMessageEntry<T = unknown>(
		customType: string | undefined,
		content: string | (TextContent | ImageContent)[] | undefined,
		display: boolean | undefined,
		details?: T,
		attribution: MessageAttribution | undefined = "agent",
		timestamp?: number,
	): string {
		const normalized = normalizeCustomMessagePayload<T>({ customType, content, display, details, attribution });
		const fresh = this.#freshEntryFields();
		const entry: CustomMessageEntry<T> = {
			type: "custom_message",
			customType: normalized.customType,
			content: normalized.content,
			display: normalized.display,
			// Drop AgentSession-internal transient fields before disk persistence.
			details: stripInternalDetailsFields(normalized.details),
			attribution: normalized.attribution,
			...fresh,
			// Prefer the initiating message's own timestamp: without it the entry
			// records the emission time, which on rebuild excludes provider
			// preparation / hook time from the prompt→yield anchor.
			timestamp: timestamp !== undefined ? new Date(timestamp).toISOString() : fresh.timestamp,
		};
		entry.sourceOrigin = { journalId: this.#sessionId, entryId: entry.id };
		this.#recordEntry(entry);
		return entry.id;
	}

	/** Append a TTSR injection entry recording which rules were injected. */
	appendTtsrInjection(ruleNames: string[]): string {
		const entry: TtsrInjectionEntry = {
			type: "ttsr_injection",
			...this.#freshEntryFields(),
			injectedRules: [...ruleNames],
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	/** All unique TTSR rule names injected on the current branch (root → leaf). */
	getInjectedTtsrRules(): string[] {
		const names = new Set<string>();
		for (const entry of this.getBranch()) {
			if (entry.type !== "ttsr_injection") continue;
			for (const name of entry.injectedRules) names.add(name);
		}
		return [...names];
	}

	/** Append a credential pin recording which OAuth account served `provider`. */
	appendCredentialPin(provider: string, hash: string): string {
		const entry: CredentialPinEntry = {
			type: "credential_pin",
			...this.#freshEntryFields(),
			provider,
			hash,
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	/**
	 * Latest credential pin per provider on the current branch (root → leaf),
	 * with the effective last-use time of the pinned account.
	 *
	 * Pins are appended only when the serving account *changes*, so a long
	 * session on one account carries a single old pin entry. Any assistant turn
	 * for the same provider after that pin was necessarily served by the pinned
	 * account, so its timestamp advances `lastUsedAt` — a resume seconds after
	 * the last turn seeds a warm sticky instead of a stale one.
	 */
	getCredentialPins(): Map<string, { hash: string; lastUsedAt: number }> {
		const pins = new Map<string, { hash: string; lastUsedAt: number }>();
		for (const [provider, pin] of this.#index.branchFold().pins) pins.set(provider, { ...pin });
		return pins;
	}

	getLeafId(): string | null {
		return this.#index.leafId();
	}

	getLeafEntry(): SessionEntry | undefined {
		return this.#index.leafEntry();
	}

	/**
	 * The most recent model role on the current branch, or undefined when no
	 * model change has been recorded.
	 */
	getLastModelChangeRole(): string | undefined {
		const branch = this.getBranch();
		for (let index = branch.length - 1; index >= 0; index--) {
			const entry = branch[index];
			if (entry.type === "model_change") return entry.role ?? "default";
		}
		return undefined;
	}

	getEntry(id: string): SessionEntry | undefined {
		return this.#index.get(id);
	}

	/** All direct children of an entry. */
	getChildren(parentId: string): SessionEntry[] {
		return this.#index.childrenOf(parentId);
	}

	getLabel(id: string): string | undefined {
		return this.#index.labelFor(id);
	}

	/**
	 * Set or clear a label on an entry. Pass undefined/empty to clear.
	 */
	appendLabelChange(targetId: string, label: string | undefined): string {
		if (!this.#index.has(targetId)) throw new Error(`Entry ${targetId} not found`);

		const entry: LabelEntry = { type: "label", ...this.#freshEntryFields(), targetId, label };
		this.#recordEntry(entry);
		return entry.id;
	}

	/**
	 * Walk from an entry to root, returning entries in path order. Includes all
	 * entry types; use buildSessionContext() for the resolved LLM messages.
	 */
	getBranch(fromId?: string): SessionEntry[] {
		return this.#index.pathTo(fromId ?? this.#index.leafId());
	}

	/**
	 * Build the session context (LLM messages), or — with `{ transcript: true }` —
	 * the full-history display transcript, from the current leaf path.
	 */
	buildSessionContext(options?: BuildSessionContextOptions): SessionContext {
		if (options?.transcript && !options.collapseCompactedHistory) {
			return buildSessionContext(this.#entries, this.#index.leafId(), this.#index.entriesById(), options);
		}
		const context = this.#index.contextPath(options);
		return buildSessionContextFromPath(context.path, options, this.#index.branchFold().controls, context.inventory);
	}

	/** Strip stale OpenAI Responses assistant replay metadata from loaded entries. */
	sanitizeLoadedOpenAIResponsesReplayMetadata(): boolean {
		let changed = false;
		for (const entry of this.#entries) {
			if (entry.type !== "message" || entry.message.role !== "assistant") continue;

			const sanitized = sanitizeRehydratedOpenAIResponsesAssistantMessage(entry.message);
			if (sanitized === entry.message) continue;

			entry.message = sanitized;
			changed = true;
		}

		if (changed) this.#notifySourceChanged();
		return changed;
	}

	getHeader(): SessionHeader | null {
		return this.#header;
	}

	/** All session entries (excludes header). Returns a shallow copy. */
	getEntries(): SessionEntry[] {
		return [...this.#entries];
	}

	/**
	 * The session as a tree. A well-formed session has exactly one root; orphaned
	 * entries (broken parent chain) are returned as roots too.
	 */
	getTree(): SessionTreeNode[] {
		return this.#index.tree(this.#entries);
	}

	/**
	 * Move the leaf to an earlier entry so the next append forms a new branch.
	 * Existing entries are never modified or deleted.
	 */
	branch(branchFromId: string): void {
		if (!this.#index.has(branchFromId)) throw new Error(`Entry ${branchFromId} not found`);
		this.#setLeaf(branchFromId);
	}

	/** Reset the leaf to null so the next append creates a new root entry. */
	resetLeaf(): void {
		this.#setLeaf(null);
	}

	/**
	 * Durably move the active branch past a discarded entry.
	 *
	 * The loader reconstructs the active branch from the last physical journal
	 * entry, so changing the in-memory leaf alone is lost on reload. Known
	 * metadata children are chained onto the discarded entry's parent before the
	 * entry is removed. If any child may carry content, the subtree is preserved
	 * off-branch instead. Both paths append a metadata-only branch marker and
	 * rewrite the journal, making the selected path durable.
	 */
	async discardEntryDurably(entryId: string): Promise<void> {
		const entry = this.#index.get(entryId);
		if (!entry) return;
		const children = this.#index.childrenOf(entryId);
		const canReparentChildren = children.every(child => child.type === "service_tier_change");
		let leafId = entry.parentId;
		if (canReparentChildren) {
			for (const child of children) {
				child.parentId = leafId;
				leafId = child.id;
			}
			this.#entries = this.#entries.filter(candidate => candidate.id !== entryId);
			this.#index.rebuild(this.#entries);
		}
		this.branchWithSummary(leafId, "", {
			kind: DISCARDED_ENTRY_BRANCH_MARKER,
			discardedEntryId: entryId,
		});
		await this.rewriteEntries();
	}

	/** Like branch(), but also records a branch_summary of the abandoned path. */
	branchWithSummary(branchFromId: string | null, summary: string, details?: unknown, fromExtension?: boolean): string {
		if (branchFromId !== null && !this.#index.has(branchFromId)) throw new Error(`Entry ${branchFromId} not found`);

		this.#setLeaf(branchFromId);
		const entry: BranchSummaryEntry = {
			type: "branch_summary",
			id: generateId(this.#index),
			parentId: branchFromId,
			timestamp: nowIso(),
			fromId: branchFromId ?? "root",
			summary,
			details,
			fromExtension,
		};
		this.#recordEntry(entry);
		return entry.id;
	}

	/**
	 * Create a new session file containing only the path from root to `leafId`.
	 * Returns the new file path, or undefined when not persisting.
	 */
	createBranchedSession(leafId: string): string | undefined {
		const sourceSessionFile = this.#sessionFile;
		const branchPath = this.getBranch(leafId);
		if (branchPath.length === 0) throw new Error(`Entry ${leafId} not found`);

		// Drop label entries from the path; recreate them fresh from the resolved map.
		const entriesToKeep = branchPath.filter(entry => entry.type !== "label");
		SessionManager.#preserveSourceOrigins(entriesToKeep, this.#sessionId);
		const keptIds = new Set(entriesToKeep.map(entry => entry.id));
		const labelsToCarry: Array<{ targetId: string; label: string }> = [];
		for (const [targetId, label] of this.#index.labelsInEffect()) {
			if (keptIds.has(targetId)) labelsToCarry.push({ targetId, label });
		}

		const timestamp = nowIso();
		const newSessionId = mintSessionId();
		this.#reconcileSessionDirForFallback();
		const newSessionFile = path.join(this.#sessionDir, `${fileSafeTimestamp(timestamp)}_${newSessionId}.jsonl`);
		const header: SessionHeader = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: newSessionId,
			timestamp,
			cwd: this.#cwd,
			title: this.#sessionName,
			titleSource: this.#titleSource,
			parentSession: this.#persist ? sourceSessionFile : undefined,
			additionalDirectories: this.#additionalDirectories.length > 0 ? [...this.#additionalDirectories] : undefined,
		};

		const labels: LabelEntry[] = [];
		let parentId = entriesToKeep[entriesToKeep.length - 1]?.id ?? null;
		for (const carried of labelsToCarry) {
			const labelEntry: LabelEntry = {
				type: "label",
				id: generateId(new Set([...keptIds, ...labels.map(entry => entry.id)])),
				parentId,
				timestamp: nowIso(),
				targetId: carried.targetId,
				label: carried.label,
			};
			labels.push(labelEntry);
			parentId = labelEntry.id;
		}

		this.#header = header;
		this.#entries = [...entriesToKeep, ...labels];
		this.#sessionId = newSessionId;
		this.#sessionName = header.title;
		this.#titleSource = header.titleSource;
		this.#titleUpdatedAt = timestamp;
		this.#hasTitleSlot = true;
		this.#index.rebuild(this.#entries);
		this.#artifactManager = null;
		this.#artifactManagerSessionFile = null;
		this.#forceFileCreation = this.#persist;

		if (!this.#persist) {
			this.#sessionFile = undefined;
			this.#fileIsCurrent = false;
			this.#rewriteRequired = false;
			return undefined;
		}

		this.#sessionFile = newSessionFile;
		this.#rewriteSynchronously();
		this.#rememberBreadcrumb(this.#cwd, newSessionFile);
		return newSessionFile;
	}

	/** Resolve the canonical default session directory for a cwd. */
	static getDefaultSessionDir(
		cwd: string,
		agentDir?: string,
		storage: SessionStorage = new FileSessionStorage(),
	): string {
		return computeDefaultSessionDir(cwd, storage, getSessionsDir(agentDir));
	}

	/**
	 * Create a new session.
	 * @param cwd Working directory (stored in the session header)
	 * @param sessionDir Optional session directory; defaults to the cwd-derived dir.
	 */
	static create(cwd: string, sessionDir?: string, storage: SessionStorage = new FileSessionStorage()): SessionManager {
		const dir = sessionDir ?? SessionManager.getDefaultSessionDir(cwd, undefined, storage);
		const manager = new SessionManager(cwd, dir, true, storage);
		manager.#resetToNewSession();
		return manager;
	}

	/**
	 * Create a fresh empty session file in the default session directory for
	 * `cwd`, writing only the session header. The returned path can be passed to
	 * `setSessionFile` / `AgentSession.switchSession` when a caller explicitly
	 * needs a brand-new persisted session at a cwd-derived path.
	 */
	static createEmptySessionFile(cwd: string, storage: SessionStorage = new FileSessionStorage()): string {
		const sessionDir = SessionManager.getDefaultSessionDir(cwd, undefined, storage);
		const id = mintSessionId();
		const timestamp = nowIso();
		const header: SessionHeader = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id,
			timestamp,
			cwd: path.resolve(cwd),
		};
		const file = path.join(sessionDir, `${fileSafeTimestamp(timestamp)}_${id}.jsonl`);
		storage.writeTextSync(file, `${serializeTitleSlot({ updatedAt: timestamp })}${JSON.stringify(header)}\n`);
		return file;
	}

	/**
	 * Fork a session into the current project directory: copy history from another
	 * session file while creating a fresh session file in this sessionDir.
	 *
	 * `options.sessionFile` pins the new session's file path (default: an
	 * auto-named `<timestamp>_<id>.jsonl` in `sessionDir`). Artifacts are copied
	 * recursively by default; nested agents that deliberately share their parent's
	 * artifact root may disable this with `copyArtifacts: false`.
	 */
	static async forkFrom(
		sourcePath: string,
		cwd: string,
		sessionDir?: string,
		storage: SessionStorage = new FileSessionStorage(),
		options?: {
			copyArtifacts?: boolean;
			suppressBreadcrumb?: boolean;
			sessionFile?: string;
			resetInheritedCost?: boolean;
		},
	): Promise<SessionManager> {
		const dir = sessionDir ?? SessionManager.getDefaultSessionDir(cwd, undefined, storage);
		const manager = new SessionManager(cwd, dir, true, storage);
		manager.#suppressBreadcrumb = options?.suppressBreadcrumb === true;

		const sourceEntries = structuredClone(await loadEntriesFromFile(sourcePath, storage)) as FileEntry[];
		migrateToCurrentVersion(sourceEntries);
		await resolveBlobRefsInEntries(sourceEntries, manager.#blobs);

		const sourceHeader = sourceEntries.find(entry => entry.type === "session") as SessionHeader | undefined;
		const history = sourceEntries.filter(entry => entry.type !== "session") as SessionEntry[];
		if (sourceHeader) SessionManager.#preserveSourceOrigins(history, sourceHeader.id);
		if (options?.resetInheritedCost) SessionManager.#resetInheritedUsageCost(history);
		manager.#resetToNewSession(
			{
				parentSession: sourceHeader?.id,
				providerPromptCacheKey: sourceHeader?.providerPromptCacheKey ?? sourceHeader?.id,
			},
			options?.sessionFile,
		);
		manager.#header.title = sourceHeader?.title;
		manager.#header.titleSource = sourceHeader?.titleSource;
		manager.#additionalDirectories = (sourceHeader?.additionalDirectories ?? []).filter(d => d !== path.resolve(cwd));
		manager.#header.additionalDirectories =
			manager.#additionalDirectories.length > 0 ? manager.#additionalDirectories : undefined;
		manager.#sessionName = manager.#header.title;
		manager.#titleSource = manager.#header.titleSource;
		manager.#titleUpdatedAt = nowIso();
		manager.#hasTitleSlot = true;
		manager.#entries = history;
		manager.#index.rebuild(history);
		manager.sanitizeLoadedOpenAIResponsesReplayMetadata();
		manager.#forceFileCreation = true;
		await manager.#rewriteAtomically();
		if (options?.copyArtifacts !== false) {
			await copySessionArtifacts(sourcePath, manager.#sessionFile!);
		}
		return manager;
	}

	/**
	 * Zero the monetary attribution (cost, credits, premium requests) on the
	 * forked history's assistant turns and completed `task` results, in place.
	 *
	 * A tan fork is a fresh agent that inherits the parent's transcript purely
	 * for context; its spend must reflect only its own work. Session cost is
	 * derived by summing `usage.cost` over the transcript, so without this the
	 * clone's Agent Hub row would open at the parent's entire accumulated cost.
	 * Token counts are left intact — compaction anchors and context math depend
	 * on them — since only billing attribution is inherited, not context size.
	 */
	static #resetInheritedUsageCost(history: SessionEntry[]): void {
		for (const entry of history) resetUsageCost(entryUsage(entry));
	}

	/**
	 * Open a specific session file.
	 * @param sessionDir Optional dir for /new or /branch; defaults to the file's parent.
	 * @param options.initialCwd Cwd to use when the file is empty or missing.
	 */
	static async open(
		filePath: string,
		sessionDir?: string,
		storage: SessionStorage = new FileSessionStorage(),
		options?: { initialCwd?: string; suppressBreadcrumb?: boolean },
	): Promise<SessionManager> {
		const loaded = await loadSessionFile(filePath, storage);
		const header = loaded.entries.find(entry => entry.type === "session") as SessionHeader | undefined;
		// Resume into the session's recorded cwd only when it is verifiably
		// accessible. A deleted or permission-blocked (macOS TCC denial) project
		// dir would make the constructor's #cwd — and the `setProjectDir` chdir
		// interactive mode runs next — fail, so fall back to the launch cwd and
		// anchor /new and /branch there too, keeping the resumed session where
		// the user already is.
		const recordedCwd = header?.cwd;
		const recordedCwdUsable = !!recordedCwd && (await directoryIsEnterable(recordedCwd));
		const cwd = recordedCwdUsable ? recordedCwd : (options?.initialCwd ?? getProjectDir());
		const dir =
			sessionDir ??
			(recordedCwd && !recordedCwdUsable
				? SessionManager.getDefaultSessionDir(cwd, undefined, storage)
				: path.dirname(path.resolve(filePath)));
		const manager = new SessionManager(cwd, dir, true, storage);
		manager.#suppressBreadcrumb = options?.suppressBreadcrumb === true;
		await manager.#setSessionFile(filePath, loaded);
		return manager;
	}

	/**
	 * Lock-free peek for cold subagent revival: returns the recorded working
	 * directory (session header) and the latest `session_init` contract (system
	 * prompt / tools / output schema) WITHOUT taking the single-writer lock that
	 * {@link open} acquires — the caller re-opens for the actual revive. Returns
	 * null when the file can't be read; `init` is null for files written before
	 * `session_init` was recorded (no faithful contract to rebuild from).
	 */
	static async peekSessionInit(
		filePath: string,
		storage: SessionStorage = new FileSessionStorage(),
	): Promise<{
		cwd: string;
		init: {
			systemPrompt: string;
			task: string;
			tools: string[];
			agent?: string;
			modelRole?: string;
			resolvedModel?: string;
			readOnly?: boolean;
			outputSchema?: unknown;
			outputSchemaMode?: StructuredSubagentSchemaMode;
			restrictToolNames?: boolean;
			spawns?: string;
			readSummarize?: boolean;
			advisor?: string;
		} | null;
	} | null> {
		let header: SessionHeader | undefined;
		let init: {
			systemPrompt: string;
			task: string;
			tools: string[];
			agent?: string;
			modelRole?: string;
			resolvedModel?: string;
			readOnly?: boolean;
			outputSchema?: unknown;
			outputSchemaMode?: StructuredSubagentSchemaMode;
			restrictToolNames?: boolean;
			spawns?: string;
			readSummarize?: boolean;
			advisor?: string;
		} | null = null;
		const visit = (entry: FileEntry): void => {
			if (entry.type === "session") {
				header ??= entry;
				return;
			}
			if (entry.type === "session_init") {
				init = {
					systemPrompt: entry.systemPrompt,
					task: entry.task,
					tools: entry.tools,
					agent: entry.agent,
					modelRole: entry.modelRole,
					resolvedModel: entry.resolvedModel,
					readOnly: entry.readOnly,
					outputSchema: entry.outputSchema,
					outputSchemaMode: entry.outputSchemaMode,
					restrictToolNames: entry.restrictToolNames,
					readSummarize: entry.readSummarize,
					spawns: entry.spawns,
					advisor: entry.advisor,
				};
			}
		};

		try {
			await visitEntriesFromFile(filePath, visit, storage);
		} catch {
			return null;
		}
		// A missing, empty, or invalid file has no usable session.
		if (!header) return null;
		return { cwd: header.cwd ?? getProjectDir(), init };
	}

	/** Continue the most recent session, or create a new one if none exists. */
	static async continueRecent(
		cwd: string,
		sessionDir?: string,
		storage: SessionStorage = new FileSessionStorage(),
	): Promise<SessionManager> {
		const dir = sessionDir ?? SessionManager.getDefaultSessionDir(cwd, undefined, storage);
		const resolvedCwd = path.resolve(cwd);
		const breadcrumb = await readTerminalBreadcrumbEntry();
		let chosenSession: string | null | undefined;

		if (breadcrumb) {
			// A lazy fresh-session boundary whose JSONL was never materialized
			// (for example, initial creation followed by exit before any assistant
			// output). Honor the boundary rather than falling back to
			// findMostRecentSession(), which would resurrect an older transcript.
			// Explicit newSession() boundaries are materialized before it returns so
			// this remains correct even when the relaunch has a different terminal id.
			if (breadcrumb.fresh && !breadcrumb.exists) {
				const manager = new SessionManager(cwd, dir, true, storage);
				manager.#resetToNewSession();
				return manager;
			}

			// Recover stale crumbs: a subagent open (pre-fix) may have pointed this
			// terminal's breadcrumb at an artifact child; resume the parent instead.
			breadcrumb.sessionFile = resolveBreadcrumbToInteractiveRoot(breadcrumb.sessionFile);
			const breadcrumbCwd = path.resolve(breadcrumb.cwd);
			if (breadcrumbCwd === resolvedCwd) {
				chosenSession = breadcrumb.sessionFile;
			} else {
				// The terminal's last session started in a different cwd. If that cwd is
				// gone (worktree move/rename) and this location has no sessions of its
				// own, re-root the moved session here instead of starting fresh. When an
				// explicit sessionDir is reused across the move, the stale breadcrumb file
				// may be the newest entry there; prefer a genuine current-cwd session.
				let newestInTargetDir = await findMostRecentSession(dir, storage);
				const breadcrumbFile = path.resolve(breadcrumb.sessionFile);
				const breadcrumbCwdMissing = !fs.existsSync(breadcrumbCwd);
				const newestIsBreadcrumb = newestInTargetDir ? path.resolve(newestInTargetDir) === breadcrumbFile : false;
				let currentProjectAlreadyHasSession = false;

				if (breadcrumbCwdMissing && newestIsBreadcrumb) {
					const localSession = (await SessionManager.list(cwd, dir, storage)).find(
						session =>
							path.resolve(session.path) !== breadcrumbFile &&
							session.cwd &&
							path.resolve(session.cwd) === resolvedCwd,
					);
					if (localSession) {
						newestInTargetDir = localSession.path;
						currentProjectAlreadyHasSession = true;
					}
				}

				const looksLikeMovedProject =
					breadcrumbCwdMissing &&
					(newestInTargetDir === null || (newestIsBreadcrumb && !currentProjectAlreadyHasSession));
				if (looksLikeMovedProject) {
					logger.info("Re-rooting moved session", { from: breadcrumbCwd, to: resolvedCwd });
					// Anchor at the gone breadcrumb cwd so the moveTo below relocates the
					// session: open() now falls back to the launch cwd for a missing
					// recorded cwd, which would no-op moveTo when it equals `cwd`.
					const manager = await SessionManager.open(breadcrumb.sessionFile, undefined, storage, {
						initialCwd: breadcrumbCwd,
					});
					await manager.moveTo(cwd, sessionDir);
					return manager;
				}

				chosenSession = newestInTargetDir;
			}
		}

		if (chosenSession === undefined) chosenSession = await findMostRecentSession(dir, storage);

		const manager = new SessionManager(cwd, dir, true, storage);
		if (chosenSession) await manager.setSessionFile(chosenSession);
		else manager.#resetToNewSession();
		return manager;
	}

	/** Create an in-memory session (no file persistence). */
	static inMemory(
		cwd: string = getProjectDir(),
		storage: SessionStorage = new MemorySessionStorage(),
	): SessionManager {
		const manager = new SessionManager(cwd, "", false, storage);
		manager.#resetToNewSession();
		return manager;
	}

	/**
	 * List sessions for a project directory.
	 * @param sessionDir Optional dir; defaults to the cwd-derived dir.
	 */
	static async list(
		cwd: string,
		sessionDir?: string,
		storage: SessionStorage = new FileSessionStorage(),
	): Promise<SessionInfo[]> {
		const dir = sessionDir ?? SessionManager.getDefaultSessionDir(cwd, undefined, storage);
		const sessions = await listSessions(dir, storage);
		return sortPinnedFirst(sessions, await loadPinnedSessionIds());
	}

	/** List all sessions across all project directories, pinned sessions first. */
	static async listAll(storage: SessionStorage = new FileSessionStorage()): Promise<SessionInfo[]> {
		const sessions = await listAllSessions(storage);
		return sortPinnedFirst(sessions, await loadPinnedSessionIds());
	}
}

/**
 * If the current session was created by `/move` and contains no real
 * user/assistant messages, delete it so empty move sessions don't accumulate.
 */
export async function cleanupEmptyMoveSession(
	sessionManager: SessionManager,
	movedFromEmptySessionFile: string | undefined,
): Promise<void> {
	const sessionFile = sessionManager.getSessionFile();
	if (!sessionFile || !movedFromEmptySessionFile) return;
	if (path.resolve(sessionFile) !== path.resolve(movedFromEmptySessionFile)) return;
	const entries = sessionManager.getEntries();
	const hasRealMessages = entries.some(
		e => e.type === "message" && (e.message.role === "user" || e.message.role === "assistant"),
	);
	if (hasRealMessages) return;
	try {
		await sessionManager.dropSession(sessionFile);
	} catch (err) {
		logger.warn("Failed to clean up empty move session", { sessionFile, error: String(err) });
	}
}
