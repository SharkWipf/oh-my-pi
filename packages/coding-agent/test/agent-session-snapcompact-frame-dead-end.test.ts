import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { Agent, Tokenizer } from "@oh-my-pi/pi-agent-core";
import * as compactionModule from "@oh-my-pi/pi-agent-core/compaction";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { loadExtensions } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/loader";
import { ExtensionRunner } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/runner";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { buildSessionContext } from "@oh-my-pi/pi-coding-agent/session/session-context";
import type { CompactionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { getProjectAgentDir, TempDir } from "@oh-my-pi/pi-utils";
import * as snapcompact from "@oh-my-pi/snapcompact";

/**
 * Regression test for the snapcompact frame dead-end.
 *
 * A branch whose LAST entry is a snapcompact CompactionEntry billed past the
 * maintenance threshold (FRAME_TOKEN_ESTIMATE × frames) dead-ends every pass:
 * prepareCompaction returns undefined (nothing after the entry to summarize),
 * and the elide/image rescue tiers only inspect "message"/"custom_message"
 * entries, so the `type: "compaction"` tail escapes both and the no-progress
 * warning re-fires on every resume — the shape issue #4786's rescue does not
 * cover.
 *
 * Rescue reframes the retained original-source archive locally. It must reduce
 * actual emitted context and survive reload without duplicating originals.
 */
describe("AgentSession snapcompact frame dead-end rescue", () => {
	let tempDir: TempDir;
	let session: AgentSession;
	let sessionManager: SessionManager;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;

	beforeAll(async () => {
		authStorage = await AuthStorage.create(":memory:");
		authStorage.keys.setRuntime("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage);
	});

	const NOTICE_SOURCE = "compaction";
	const SEEDED_FRAME_COUNT = 16;
	const ORIGINAL_TEXT = `HEAD sentinel. ${"Archived history line. ".repeat(250)}TAIL sentinel.`;
	const tokenizer = new Tokenizer();

	async function createSession(options: {
		frameCount: number;
		visionModel?: boolean;
		/** Seed no compaction entry; instead a hook supplies one carrying this
		 *  many frames — exercising the POST-PASS dead-end (a completed pass
		 *  whose just-written archive is itself the over-budget cost). */
		hookArchiveFrames?: number;
		/** Seed a kept-recent entry between firstKeptEntryId and the archive —
		 *  re-emitted by buildSessionContext, so the rescue budget must charge it. */
		preArchiveKeptText?: string;
	}): Promise<void> {
		tempDir = TempDir.createSync("@pi-snapcompact-frame-dead-end-");
		sessionManager = SessionManager.create(tempDir.path(), tempDir.path());
		const original = { role: "user" as const, content: ORIGINAL_TEXT, timestamp: 1 };
		const originalId = sessionManager.appendMessage(original);
		const userEntryId = sessionManager.appendMessage({ role: "user", content: "hello", timestamp: 2 });
		const source = { entryId: originalId, order: 0, message: original };
		// Small real frames model an archive written with an older geometry.
		const archive = await snapcompact.compact(
			{
				firstKeptEntryId: userEntryId,
				tokensBefore: 150_000,
				fileOps: snapcompact.createFileOps(),
				messagesToSummarize: [original],
				turnPrefixMessages: [],
				sourcesToSummarize: [source],
				turnPrefixSources: [],
				recentSources: [],
				selectedSources: [source],
			},
			{
				shape: {
					font: "8x8",
					cellWidth: 8,
					cellHeight: 8,
					lineRepeat: 1,
					variant: "bw",
					frameSize: 128,
					frameTokenEstimate: 100,
				},
				maxFrames: options.hookArchiveFrames ?? options.frameCount,
			},
		);

		let extensionRunner: ExtensionRunner | undefined;
		if (options.hookArchiveFrames !== undefined) {
			// Short-circuit the summarization LLM call with a hook-supplied
			// compaction whose archive carries the oversized frame payload —
			// mirrors agent-session-auto-compaction-progress-guard.test.ts.
			const extensionsDir = path.join(getProjectAgentDir(tempDir.path()), "extensions");
			fs.mkdirSync(extensionsDir, { recursive: true });
			const extensionPath = path.join(extensionsDir, "compaction-short-circuit.ts");
			fs.writeFileSync(
				extensionPath,
				[
					"export default function(pi) {",
					'\tpi.on("session_before_compact", async (event) => {',
					"\t\treturn {",
					"\t\t\tcompaction: {",
					'\t\t\t\tsummary: "compacted",',
					"\t\t\t\tshortSummary: undefined,",
					`\t\t\t\tfirstKeptEntryId: ${JSON.stringify(userEntryId)},`,
					"\t\t\t\ttokensBefore: event.preparation.tokensBefore,",
					"\t\t\t\tdetails: {},",
					`\t\t\t\tpreserveData: ${JSON.stringify(archive.preserveData)},`,
					"\t\t\t},",
					"\t\t};",
					"\t});",
					"}",
				].join("\n"),
			);
			const extensionsResult = await loadExtensions([extensionPath], tempDir.path());
			extensionRunner = new ExtensionRunner(
				extensionsResult.extensions,
				extensionsResult.runtime,
				tempDir.path(),
				sessionManager,
				modelRegistry,
			);
		}

		const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!bundled) {
			throw new Error("Expected built-in anthropic model to exist");
		}
		// Pin the window: threshold/band math below is tuned to 200k.
		const model = {
			...bundled,
			contextWindow: 200_000,
			maxTokens: 64_000,
			...(options.visionModel === false ? { input: ["text" as const] } : {}),
		};

		// Seed the poisoned shape: one user turn, then (unless the hook supplies
		// the archive) a trailing snapcompact CompactionEntry as the LAST branch
		// entry — the real prepareCompaction must hit its
		// last-entry-is-compaction guard organically.
		if (options.preArchiveKeptText !== undefined && options.hookArchiveFrames === undefined) {
			// A kept-recent entry BETWEEN firstKeptEntryId and the archive:
			// buildSessionContext re-emits it before the compaction entry, so
			// the rescue budget must charge it too.
			sessionManager.appendMessage({
				role: "toolResult",
				toolCallId: "call-kept",
				toolName: "bash",
				content: [{ type: "text", text: options.preArchiveKeptText }],
				isError: false,
				timestamp: Date.now(),
			});
		}
		if (options.hookArchiveFrames === undefined) {
			sessionManager.appendCompaction(
				"Archived history onto stale snapcompact frames.",
				"stale snapcompact archive",
				userEntryId,
				150_000,
				{
					details: { readFiles: ["src/a.ts"], modifiedFiles: ["src/b.ts"] },
					preserveData: archive.preserveData,
				},
			);
		}

		const agent = new Agent({
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [],
				messages: sessionManager.buildSessionContext().messages,
			},
		});

		session = new AgentSession({
			agent,
			sessionManager,
			settings: Settings.isolated({
				"compaction.autoContinue": true,
				"compaction.methodOrder": ["snapcompact", "soft"],
				// Fixed trigger so the rescue's threshold-derived frame budget is
				// deterministic: band 0.8 × 60k = 48k minus base/edge reserves
				// yields well under 16 frames — the rebuild must shrink.
				"compaction.thresholdTokens": 60_000,
			}),
			modelRegistry,
			extensionRunner,
		});
	}

	afterEach(async () => {
		try {
			await session?.dispose();
			await sessionManager?.close();
		} finally {
			await tempDir?.remove();
			vi.restoreAllMocks();
		}
	});

	afterAll(() => {
		authStorage.close();
	});

	function collectNotices() {
		const notices: { level: string; message: string; source?: string }[] = [];
		session.subscribe(event => {
			if (event.type === "notice") {
				notices.push({ level: event.level, message: event.message, source: event.source });
			}
		});
		return notices;
	}

	/** Threshold-tripping assistant turn against the 60k trigger. */
	function highUsageAssistant() {
		return {
			role: "assistant" as const,
			content: [{ type: "text" as const, text: "Done." }],
			api: "anthropic-messages" as const,
			provider: "anthropic" as const,
			model: "claude-sonnet-4-5",
			stopReason: "stop" as const,
			usage: {
				input: 190000,
				output: 1000,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 191000,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		};
	}

	async function triggerMaintenance(options: { appendAssistant?: boolean } = {}): Promise<void> {
		const { promise: compactionDone, resolve: onCompactionDone } = Promise.withResolvers<void>();
		session.subscribe(event => {
			if (event.type === "auto_compaction_end") onCompactionDone();
		});
		const assistantMsg = highUsageAssistant();
		// Resume maintenance without extending the archived branch. Appending an
		// assistant would make the retained user turn legitimately compactable.
		if (options.appendAssistant) session.agent.emitExternalEvent({ type: "message_end", message: assistantMsg });
		session.agent.emitExternalEvent({ type: "agent_end", messages: [assistantMsg] });
		await compactionDone;
		await session.waitForIdle();
	}

	function compactions(): CompactionEntry[] {
		return sessionManager.getBranch().filter((entry): entry is CompactionEntry => entry.type === "compaction");
	}

	function useLocalContextUsage(): void {
		vi.spyOn(session, "getContextUsage").mockImplementation(() => {
			const tokens = tokenizer.countMessages(session.agent.state.messages);
			return { tokens, contextWindow: 200_000, percent: tokens / 2000 };
		});
	}

	async function expectDurableRescue(): Promise<CompactionEntry> {
		const entries = compactions();
		expect(entries).toHaveLength(2);
		const [stale, rebuilt] = entries;
		const oldArchive = snapcompact.getPreservedArchive(stale.preserveData)!;
		const archive = snapcompact.getPreservedArchive(rebuilt.preserveData)!;
		expect(oldArchive.frames).toHaveLength(SEEDED_FRAME_COUNT);
		expect(archive.frames.length).toBeLessThan(oldArchive.frames.length);
		expect(archive.text).toBe(oldArchive.text);
		expect(archive.text).toContain(ORIGINAL_TEXT);
		const context = sessionManager.buildSessionContext().messages;
		const previousContext = buildSessionContext(sessionManager.getEntries(), stale.id).messages;
		expect(tokenizer.countMessages(context)).toBeLessThan(tokenizer.countMessages(previousContext));
		// The archived original is not replayed a second time as a live user turn.
		expect(context.filter(message => message.role === "user").map(message => message.content)).toEqual(["hello"]);
		const original = sessionManager.getBranch().find(entry => entry.type === "message");
		if (original?.type !== "message" || original.message.role !== "user")
			throw new Error("Expected retained original");
		expect(original.message.content).toBe(ORIGINAL_TEXT);
		await sessionManager.ensureOnDisk();
		await sessionManager.flush();
		const reloaded = await SessionManager.open(sessionManager.getSessionFile()!);
		try {
			expect(
				reloaded.buildSessionContext().messages.map(message => ({
					role: message.role,
					content: "content" in message ? message.content : undefined,
				})),
			).toEqual(
				context.map(message => ({
					role: message.role,
					content: "content" in message ? message.content : undefined,
				})),
			);
			expect(reloaded.getEntry(original.id)).toEqual(original);
			for (const entry of entries) {
				const persisted = reloaded.getEntry(entry.id);
				if (persisted?.type !== "compaction") throw new Error("Expected durable archive");
				expect(snapcompact.getPreservedArchive(persisted.preserveData)?.text).toBe(oldArchive.text);
			}
		} finally {
			await reloaded.close();
		}
		return rebuilt;
	}

	it("reframes a stale archive into real headroom without losing or duplicating originals", async () => {
		await createSession({ frameCount: SEEDED_FRAME_COUNT });
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		useLocalContextUsage();
		const before = tokenizer.countMessages(session.agent.state.messages);
		const notices = collectNotices();
		const ends: { result?: compactionModule.CompactionResult; skipped?: boolean }[] = [];
		session.subscribe(event => {
			if (event.type === "auto_compaction_end") ends.push({ result: event.result, skipped: event.skipped });
		});
		await triggerMaintenance();
		const rebuilt = await expectDurableRescue();
		const after = tokenizer.countMessages(session.agent.state.messages);
		expect(after).toBeLessThan(before);
		expect(after).toBeLessThan(48_000);
		// Consumers must receive the installed result rather than a skipped pass.
		expect(ends).toHaveLength(1);
		expect(ends[0].result?.summary).toBe(rebuilt.summary);
		expect(ends[0].skipped).not.toBe(true);
		expect(notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning")).toEqual([]);
	});

	it("reframes a just-written archive without losing its durable original source", async () => {
		await createSession({ frameCount: 0, hookArchiveFrames: SEEDED_FRAME_COUNT });
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		useLocalContextUsage();
		const notices = collectNotices();
		await triggerMaintenance({ appendAssistant: true });
		await expectDurableRescue();
		expect(tokenizer.countMessages(session.agent.state.messages)).toBeLessThan(48_000);
		expect(notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning")).toEqual([]);
	});

	it("pauses continuation and marks the active archive when reframing still leaves usage over budget", async () => {
		await createSession({ frameCount: SEEDED_FRAME_COUNT });
		const prompt = vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		const continuation = vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		const before = tokenizer.countMessages(session.agent.state.messages);
		const notices = collectNotices();
		await triggerMaintenance();
		const active = await expectDurableRescue();
		expect(tokenizer.countMessages(session.agent.state.messages)).toBeLessThan(before);
		const warnings = notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning");
		expect(warnings).toHaveLength(1);
		expect(active.warning).toBe(warnings[0].message);
		expect(prompt).not.toHaveBeenCalled();
		expect(continuation).not.toHaveBeenCalled();
	});

	it("bails when the kept tail plus fixed context leaves no frame budget", async () => {
		// Codex review on #6362 (round 5): a tail just under the recovery band
		// still cannot coexist with the fixed context + a minimum rebuilt
		// archive. The budget now charges the kept tail like
		// #compactionCreatedHeadroom does, so the rescue must bail instead of
		// appending a rebuild that can never create headroom.
		await createSession({ frameCount: SEEDED_FRAME_COUNT });
		// ~40k estimated tokens: under the 48k band, but over band − edges/template.
		sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: "call-mid",
			toolName: "bash",
			content: [{ type: "text", text: "y".repeat(160_000) }],
			isError: false,
			timestamp: Date.now(),
		});
		vi.spyOn(compactionModule, "prepareCompaction").mockReturnValue(undefined);
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		vi.spyOn(session, "shake").mockResolvedValue({
			mode: "elide",
			toolResultsDropped: 0,
			blocksDropped: 0,
			tokensFreed: 0,
		});

		await triggerMaintenance();

		expect(compactions()).toHaveLength(1);
		expect(snapcompact.getPreservedArchive(compactions()[0].preserveData)?.frames).toHaveLength(SEEDED_FRAME_COUNT);
		expect(sessionManager.getBranch().at(-1)?.type).not.toBe("compaction");
	});

	it("bails when the kept region BEFORE the archive leaves no frame budget", async () => {
		// Codex review on #6362 (round 6): the rebuilt compaction preserves
		// firstKeptEntryId, and buildSessionContext re-emits the kept messages
		// that sit BEFORE the compaction entry — so a large pre-archive kept
		// region costs the rebuilt prompt exactly like a post-archive tail.
		// The budget must charge it, or the rescue appends a still-over-band
		// archive and the dead-end persists.
		await createSession({ frameCount: SEEDED_FRAME_COUNT, preArchiveKeptText: "y".repeat(160_000) });
		vi.spyOn(compactionModule, "prepareCompaction").mockReturnValue(undefined);
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		vi.spyOn(session, "shake").mockResolvedValue({
			mode: "elide",
			toolResultsDropped: 0,
			blocksDropped: 0,
			tokensFreed: 0,
		});

		await triggerMaintenance();

		expect(compactions()).toHaveLength(1);
		expect(snapcompact.getPreservedArchive(compactions()[0].preserveData)?.frames).toHaveLength(SEEDED_FRAME_COUNT);
	});

	it("leaves an oversized non-archive tail to the elide tiers instead of rescuing the archive", async () => {
		// Codex review on #6362 (round 4): with […, archive, HUGE kept tool
		// result], rebuilding the archive would append the replacement at the
		// leaf — making the branch tail a compaction entry that
		// prepareCompaction's last-entry guard can never summarize past, even
		// after elide shrinks the real culprit. The rescue must bail when the
		// post-archive tail alone exceeds the recovery band.
		await createSession({ frameCount: SEEDED_FRAME_COUNT });
		// Seed a kept tool-result tail far above the 0.8 × 60k band.
		sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: "call-huge",
			toolName: "bash",
			content: [{ type: "text", text: "x".repeat(400_000) }],
			isError: false,
			timestamp: Date.now(),
		});
		// Force the no-preparation dead-end (as in the #4786 guard tests): the
		// oversized turn leaves nothing summarizable, which is the shape where
		// a premature archive rebuild would wedge prepareCompaction.
		vi.spyOn(compactionModule, "prepareCompaction").mockReturnValue(undefined);
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		vi.spyOn(session, "shake").mockResolvedValue({
			mode: "elide",
			toolResultsDropped: 0,
			blocksDropped: 0,
			tokensFreed: 0,
		});

		const notices = collectNotices();
		await triggerMaintenance();

		expect(compactions()).toHaveLength(1);
		expect(snapcompact.getPreservedArchive(compactions()[0].preserveData)?.frames).toHaveLength(SEEDED_FRAME_COUNT);
		const lastEntry = sessionManager.getBranch().at(-1);
		expect(lastEntry?.type).not.toBe("compaction");
		const noProgress = notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning");
		expect(noProgress.length).toBe(1);
	});

	it("still warns once when the trailing archive is already at the minimum frame count", async () => {
		await createSession({ frameCount: 1 });
		const promptSpy = vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		vi.spyOn(session, "shake").mockResolvedValue({
			mode: "elide",
			toolResultsDropped: 0,
			blocksDropped: 0,
			tokensFreed: 0,
		});

		const notices = collectNotices();
		await triggerMaintenance();

		expect(compactions()).toHaveLength(1);
		expect(snapcompact.getPreservedArchive(compactions()[0].preserveData)?.frames).toHaveLength(1);
		expect(promptSpy).not.toHaveBeenCalled();
		expect(continueSpy).not.toHaveBeenCalled();
		const noProgress = notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning");
		expect(noProgress).toHaveLength(1);
	});

	it("skips the frame rescue when the active model is not vision-capable", async () => {
		await createSession({ frameCount: SEEDED_FRAME_COUNT, visionModel: false });
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined as never);
		vi.spyOn(session.agent, "continue").mockResolvedValue();
		vi.spyOn(session, "getContextUsage").mockReturnValue({ tokens: 190000, contextWindow: 200000, percent: 95 });
		vi.spyOn(session, "shake").mockResolvedValue({
			mode: "elide",
			toolResultsDropped: 0,
			blocksDropped: 0,
			tokensFreed: 0,
		});

		const notices = collectNotices();
		await triggerMaintenance();

		// Text-only model: no frame re-render; existing tiers still run and the
		// existing dead-end warning is preserved.
		expect(compactions()).toHaveLength(1);
		expect(snapcompact.getPreservedArchive(compactions()[0].preserveData)?.frames).toHaveLength(SEEDED_FRAME_COUNT);
		const noProgress = notices.filter(n => n.source === NOTICE_SOURCE && n.level === "warning");
		expect(noProgress.length).toBe(1);
	});
});
