import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { Agent, type AgentEvent } from "@oh-my-pi/pi-agent-core";
import { USELESS_NOTICE } from "@oh-my-pi/pi-agent-core/compaction/pruning";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

/**
 * Regression: the per-turn supersede/useless prune pass rewrote the LIVE agent
 * context but never persisted the rewrite, so the session file kept the
 * original (un-pruned) history. Anything that rebuilds from the file — `/tan`
 * and `/fork` clones, session resume — then produced a divergent, larger
 * prefix and cold-missed the provider prompt cache the parent had populated.
 *
 * Contract: after the prune fires, rebuilding the session from disk yields the
 * same message content as the live agent state.
 */
describe("AgentSession per-turn prune persistence", () => {
	let tempDir: TempDir;
	let session: AgentSession;
	let sessionManager: SessionManager;
	let authStorage: AuthStorage;
	let dispatch: (event: AgentEvent) => Promise<void>;

	const BIG_CALL_ID = "call-big-useless";

	beforeEach(async () => {
		tempDir = TempDir.createSync("@pi-prune-persistence-");
		authStorage = await AuthStorage.create(path.join(tempDir.path(), "testauth.db"));
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		const modelRegistry = new ModelRegistry(authStorage);
		sessionManager = SessionManager.create(tempDir.path(), tempDir.path());

		const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!bundled) throw new Error("Expected built-in anthropic model to exist");
		const model = { ...bundled, contextWindow: 200_000, maxTokens: 64_000 };

		const now = Date.now();
		const usageZero = {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		};
		sessionManager.appendMessage({
			role: "user",
			content: "Investigate every module of the project.",
			timestamp: now - 200,
		});
		sessionManager.appendMessage({
			role: "assistant",
			content: [{ type: "toolCall", id: BIG_CALL_ID, name: "grep", arguments: { pattern: "TODO" } }],
			api: "anthropic-messages",
			provider: "anthropic",
			model: "claude-sonnet-4-5",
			stopReason: "toolUse",
			usage: usageZero,
			timestamp: now - 180,
		});
		// The only prune candidate: a big result flagged useless whose suffix
		// stays inside the cache-warm window, so the pass rewrites it.
		sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: BIG_CALL_ID,
			toolName: "grep",
			content: [{ type: "text", text: "match line\n".repeat(20000) }],
			isError: false,
			useless: true,
			timestamp: now - 170,
		});
		sessionManager.appendMessage({
			role: "assistant",
			content: [{ type: "text", text: "Nothing relevant found; moving on." }],
			api: "anthropic-messages",
			provider: "anthropic",
			model: "claude-sonnet-4-5",
			stopReason: "stop",
			usage: { ...usageZero, input: 100, output: 10, totalTokens: 110 },
			timestamp: now - 160,
		});

		const agent = new Agent({
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
		});
		const subscribe = vi.spyOn(agent, "subscribe");
		session = new AgentSession({
			agent,
			sessionManager,
			settings: Settings.isolated({
				"compaction.enabled": false,
				"compaction.dropUseless": true,
				"compaction.supersedeReads": true,
			}),
			modelRegistry,
		});
		const listener = subscribe.mock.calls[0][0];
		dispatch = event => Promise.resolve(listener(event));
		subscribe.mockRestore();
		session.agent.replaceMessages(session.buildDisplaySessionContext().messages);
		const terminal = session.agent.state.messages.findLast(message => message.role === "assistant");
		if (terminal?.role !== "assistant") throw new Error("Missing assistant fixture");
		await dispatch({ type: "message_end", message: { ...terminal, timestamp: Date.now() } });
	});

	afterEach(async () => {
		try {
			await session?.dispose();
		} finally {
			vi.restoreAllMocks();
			authStorage?.close();
			await tempDir?.remove();
		}
	});

	function liveResultText(): string {
		const message = session.agent.state.messages.find(
			candidate => candidate.role === "toolResult" && candidate.toolCallId === BIG_CALL_ID,
		);
		if (message?.role !== "toolResult" || !Array.isArray(message.content)) {
			throw new Error("Expected the seeded tool result in live agent state");
		}
		const text = message.content.find(block => block.type === "text");
		if (text?.type !== "text") throw new Error("Expected text content on the seeded tool result");
		return text.text;
	}

	it("settles an already-disposed owner's automatic prune without rewriting its sources", async () => {
		const messages = [...session.agent.state.messages];
		const original = liveResultText();
		session.beginDispose();
		await dispatch({ type: "agent_end", messages });
		expect(liveResultText()).toBe(original);
	});

	it("abandons the stale prune when a branch replaces its scope during source preparation", async () => {
		const messages = [...session.agent.state.messages];
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const prepare = session.preparePreservedMessages.bind(session);
		vi.spyOn(session, "preparePreservedMessages").mockImplementationOnce(async () => {
			entered.resolve();
			await release.promise;
			return prepare();
		});
		const settled = Promise.resolve(dispatch({ type: "agent_end", messages })).then(
			() => undefined,
			error => error,
		);
		try {
			await entered.promise;
			const root = sessionManager.getBranch()[0];
			sessionManager.branch(root.id);
			// The replacement scope independently contains an eligible prune victim.
			// An abandoned pass must not rebind itself to this new context.
			for (const message of messages.slice(1)) {
				if (message.role === "assistant" || message.role === "toolResult") {
					sessionManager.appendMessage(structuredClone(message));
				}
			}
			session.agent.replaceMessages(session.buildDisplaySessionContext().messages);
			const replacement = session.agent.state.messages;
			const original = liveResultText();
			release.resolve();
			expect(await settled).toBeUndefined();
			expect(session.agent.state.messages).toBe(replacement);
			expect(liveResultText()).toBe(original);
			const sources = sessionManager.getEntries().flatMap(entry =>
				entry.type === "message" && entry.message.role === "toolResult" ? [entry.message.content] : [],
			);
			expect(sources).toEqual([
				[{ type: "text", text: original }],
				[{ type: "text", text: original }],
			]);
		} finally {
			release.resolve();
			await settled;
		}
	});

	it("keeps source-storage failures operational rather than treating them as cancellation", async () => {
		const failure = new Error("preservation source storage unavailable");
		const original = liveResultText();
		vi.spyOn(sessionManager, "rewriteEntries").mockRejectedValueOnce(failure);
		await expect(Promise.resolve(dispatch({ type: "agent_end", messages: [...session.agent.state.messages] }))).rejects.toBe(failure);
		expect(liveResultText()).toBe(original);
	});

	it("does not price unchanged historical users on a no-candidate ordinary turn", async () => {
		const oldResult = sessionManager.getEntries().find(entry => entry.type === "message" && entry.message.role === "toolResult");
		if (oldResult?.type !== "message" || oldResult.message.role !== "toolResult") throw new Error("Missing result fixture");
		oldResult.message.prunedAt = Date.now();
		const finalAssistant = session.agent.state.messages.findLast(message => message.role === "assistant");
		if (finalAssistant?.role !== "assistant") throw new Error("Missing assistant fixture");
		const history = new Set<object>();
		for (let index = 0; index < 256; index++) {
			const message = { role: "user" as const, content: "Historical source " + index, timestamp: Date.now() };
			sessionManager.appendMessage(message);
			history.add(message);
		}
		const currentUser = sessionManager.appendMessage({ role: "user", content: "Continue normally", timestamp: Date.now() });
		sessionManager.appendCompaction("Historical context", undefined, currentUser, 4096);
		session.agent.replaceMessages(session.buildDisplaySessionContext().messages);
		const tokenizer = session.agent.tokenizer;
		const countMessage = tokenizer.countMessage.bind(tokenizer);
		let historicalPrices = 0;
		tokenizer.countMessage = (...args) => {
			if (history.has(args[0])) historicalPrices++;
			return countMessage(...args);
		};
		try {
			const completed = { ...finalAssistant, timestamp: Date.now() };
			await dispatch({ type: "message_end", message: completed });
			await dispatch({ type: "agent_end", messages: [completed] });
			expect(historicalPrices).toBe(0);
			expect(session.getPreservedMessageQuery()).toBeUndefined();
		} finally {
			tokenizer.countMessage = countMessage;
		}
	});

	it("keeps an admitted Always tool exchange intact when it becomes a real prune candidate", async () => {
		const result = sessionManager.getEntries().find(entry => entry.type === "message" && entry.message.role === "toolResult");
		if (result?.type !== "message") throw new Error("Missing result fixture");
		session.settings.override("compaction.keepLastLimit", "all");
		await session.setPreservedMessageOverride(result.id, "keep");
		const original = liveResultText();
		const previous = session.agent.state.messages.findLast(message => message.role === "assistant");
		if (previous?.role !== "assistant") throw new Error("Missing assistant fixture");
		const completed = { ...previous, timestamp: Date.now() };
		await dispatch({ type: "message_end", message: completed });
		await dispatch({ type: "agent_end", messages: [completed] });
		expect(liveResultText()).toBe(original);

		// Removing the same source decision makes the unchanged victim eligible again.
		await session.setPreservedMessageOverride(result.id, "auto");
		await dispatch({ type: "message_end", message: completed });
		await dispatch({ type: "agent_end", messages: [completed] });
		expect(liveResultText()).toBe(USELESS_NOTICE);
	});


	it("persists the pruned rewrite so a from-disk rebuild matches the live context", async () => {
		const finalAssistant = {
			role: "assistant" as const,
			content: [{ type: "text" as const, text: "Continuing." }],
			api: "anthropic-messages" as const,
			provider: "anthropic" as const,
			model: "claude-sonnet-4-5",
			stopReason: "stop" as const,
			usage: {
				input: 100,
				output: 10,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 110,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		};
		session.agent.emitExternalEvent({ type: "message_end", message: finalAssistant });
		session.agent.emitExternalEvent({ type: "agent_end", messages: [finalAssistant] });
		await session.waitForIdle();

		// The per-turn pass rewrote the live context…
		expect(liveResultText()).toBe(USELESS_NOTICE);

		// …and the persisted file must rebuild to the SAME content (fork/resume
		// read this file; a divergent prefix cold-misses the provider cache).
		await sessionManager.flush();
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a persisted session file");
		const reloaded = await SessionManager.open(sessionFile, tempDir.path());
		const rebuilt = reloaded
			.buildSessionContext()
			.messages.find(candidate => candidate.role === "toolResult" && candidate.toolCallId === BIG_CALL_ID);
		if (rebuilt?.role !== "toolResult" || !Array.isArray(rebuilt.content)) {
			throw new Error("Expected the seeded tool result in the from-disk rebuild");
		}
		const rebuiltText = rebuilt.content.find(block => block.type === "text");
		expect(rebuiltText?.type === "text" ? rebuiltText.text : undefined).toBe(USELESS_NOTICE);
	});
});
