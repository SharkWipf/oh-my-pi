import { afterEach, describe, expect, it, vi } from "bun:test";
import { Agent, Tokenizer } from "@oh-my-pi/pi-agent-core";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { MemorySessionStorage } from "@oh-my-pi/pi-coding-agent/session/session-storage";
import * as snapcompact from "@oh-my-pi/snapcompact";

// Unknown tokenizer keeps the repeated-archive comparison in the local byte
// estimate domain rather than provider billing or serializer-specific constants.
const bundled = getBundledModel("openai", "gpt-5.5");
if (!bundled) throw new Error("Expected bundled gpt-5.5 model");
const model = { ...bundled, tokenizer: undefined, contextWindow: 36_000, maxTokens: 8192 };
const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

let session: AgentSession | undefined;
let authStorage: AuthStorage | undefined;
afterEach(async () => {
	await session?.dispose();
	authStorage?.close();
	vi.restoreAllMocks();
});

describe("repeated snapcompact reduction baseline", () => {
	it("counts the committed archive once before reducing it, including after journal reload", async () => {
		const first = await snapcompact.compact(
			{
				firstKeptEntryId: "tail",
				messagesToSummarize: [user("A".repeat(50_000))],
				turnPrefixMessages: [],
				tokensBefore: 0,
				fileOps: snapcompact.createFileOps(),
			},
			{ model, shape: snapcompact.resolveShape(model), maxFrames: 3 },
		);
		const storage = new MemorySessionStorage();
		const file = "/snapcompact-baseline/session.jsonl";
		storage.writeTextSync(file, "");
		let manager = await SessionManager.open(file, undefined, storage, { suppressBreadcrumb: true });
		manager.appendMessage(user("A".repeat(50_000)));
		const tail = manager.appendMessage(user(""));
		manager.appendCompaction(first.summary, first.shortSummary, tail, first.tokensBefore, {
			preserveData: first.preserveData,
			method: "snapcompact",
		});
		await manager.ensureOnDisk();
		await manager.close();
		manager = await SessionManager.open(file, undefined, storage, { suppressBreadcrumb: true });
		manager.appendMessage(user("B".repeat(40_000)));
		manager.appendMessage(user("tail"));

		authStorage = await AuthStorage.create(":memory:");
		const settings = Settings.isolated({
			"compaction.methodOrder": ["snapcompact"],
			"compaction.autoContinue": false,
			"compaction.asyncEnabled": false,
			"compaction.keepRecentTokens": 1,
			"compaction.reserveTokens": 8192,
		});
		const agent = new Agent({
			initialState: { model, systemPrompt: [], tools: [], messages: manager.buildSessionContext().messages },
			streamFn: () => {
				throw new Error("Snapcompact must not call a provider");
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: manager,
			settings,
			modelRegistry: new ModelRegistry(authStorage),
		});
		const tokenizer = new Tokenizer();
		const before = tokenizer.countMessages(agent.state.messages, { excludeEncryptedReasoning: true });
		await session.compact(undefined, { mode: "snapcompact" });
		const after = tokenizer.countMessages(agent.state.messages, { excludeEncryptedReasoning: true });
		expect(after).toBeLessThan(before);
		// Comparing only newly archived input would reject this actual reduction.
		expect(after).toBeGreaterThan(tokenizer.countMessage(user("B".repeat(40_000))));
		const committed = manager.getBranch().findLast(entry => entry.type === "compaction");
		expect(snapcompact.getPreservedArchive(committed?.preserveData)?.frames).toHaveLength(3);
		await manager.flush();
		const reloaded = await SessionManager.open(file, undefined, storage, { suppressBreadcrumb: true });
		expect(tokenizer.countMessages(reloaded.buildSessionContext().messages)).toBe(after);
		await reloaded.close();
		// Isolate equality from the previous pass's nonempty retained tail:
		// correctly archiving that tail would otherwise yield positive savings.
		// An unchanged archive plus an empty newly archived slice must not commit.
		if (!committed) throw new Error("Expected committed archive");
		await session.dispose();
		manager = SessionManager.inMemory();
		const emptyTail = manager.appendMessage(user(""));
		manager.appendCompaction(committed.summary, committed.shortSummary, emptyTail, 0, {
			preserveData: committed.preserveData,
			method: "snapcompact",
		});
		manager.appendMessage(user(""));
		const lastTail = manager.appendMessage(user("tail"));
		session = new AgentSession({
			agent: new Agent({
				initialState: { model, systemPrompt: [], tools: [], messages: manager.buildSessionContext().messages },
				streamFn: () => {
					throw new Error("Snapcompact must not call a provider");
				},
			}),
			sessionManager: manager,
			settings,
			modelRegistry: new ModelRegistry(authStorage),
		});
		vi.spyOn(snapcompact, "compact").mockResolvedValue({
			summary: committed.summary,
			shortSummary: committed.shortSummary,
			firstKeptEntryId: lastTail,
			tokensBefore: 0,
			details: { readFiles: [], modifiedFiles: [] },
			preserveData: committed.preserveData,
		});
		const leafBeforeRejectedPass = manager.getLeafId();
		await expect(session.compact(undefined, { mode: "snapcompact" })).rejects.toThrow(
			"snapcompact would not reduce context locally.",
		);
		expect(manager.getLeafId()).toBe(leafBeforeRejectedPass);
	});
});
