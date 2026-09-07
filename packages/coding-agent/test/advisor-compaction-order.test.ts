import { afterEach, describe, expect, it, vi } from "bun:test";
import { Agent, type AgentTurnEndContext } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, UserMessage } from "@oh-my-pi/pi-ai";
import { createMockModel, type MockHandler, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage, createInMemoryAuthStorage } from "./helpers/agent-session-setup";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
	for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
	vi.restoreAllMocks();
});

function harness(options: { enabled?: boolean; active?: boolean; summarize?: MockHandler } = {}) {
	const temp = TempDir.createSync("@pi-advisor-order-");
	const auth = createInMemoryAuthStorage();
	auth.setRuntimeApiKey("anthropic", "test-key");
	registerMockApi();
	const primary = createMockModel({ handler: { content: ["primary complete"] } });
	const advisor = createMockModel({ handler: { content: ["review complete"] } });
	const summary = createMockModel({
		provider: "anthropic",
		id: "claude-sonnet-4-5",
		contextWindow: 100_000,
		handler: options.summarize ?? { content: ["COMPACTED HISTORY"] },
	});
	const model = summary;
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model, systemPrompt: ["Test"], tools: [] },
		streamFn: primary.stream,
	});
	const boundarySpy = vi.spyOn(agent, "setOnTurnEnd");
	const subscriptionSpy = vi.spyOn(agent, "subscribe");
	const settings = Settings.isolated({
		"advisor.compactBeforeGuidance": options.enabled ?? true,
		"advisor.syncBacklog": "1",
		"compaction.enabled": true,
		"compaction.asyncEnabled": true,
		"compaction.methodOrder": ["soft"],
		"compaction.thresholdPercent": 50,
		"compaction.keepRecentTokens": 500,
		"compaction.autoContinue": false,
		"contextPromotion.enabled": false,
		"todo.reminders": false,
	});
	settings.setModelRole("advisor", "anthropic/claude-sonnet-4-5");
	const manager = SessionManager.create(temp.path(), temp.path());
	const registry = new ModelRegistry(auth, temp.join("models.yml"));
	vi.spyOn(registry, "getApiKey").mockResolvedValue("test-key");
	vi.spyOn(registry, "getAvailable").mockReturnValue([model]);
	const session = new AgentSession({
		agent,
		sessionManager: manager,
		settings,
		modelRegistry: registry,
		advisorTools: [],
		advisorStreamFn: advisor.stream,
		sideStreamFn: summary.stream,
	});
	if (options.active !== false) session.setAdvisorEnabled(true);
	const boundary = boundarySpy.mock.calls[0][0]!;
	const dispatch = subscriptionSpy.mock.calls[0][0];
	cleanups.push(async () => {
		await session.dispose();
		auth.close();
		await temp.remove();
	});
	function append(message: AssistantMessage | UserMessage) {
		manager.appendMessage(message);
		agent.appendMessage(message);
	}
	function assistant(text: string, tokens = 100): AssistantMessage {
		const message = createAssistantMessage(text);
		message.model = model.id;
		message.usage.input = tokens;
		message.usage.totalTokens = tokens;
		return message;
	}
	append({ role: "user", content: "OLD SOURCE ".repeat(4_000), timestamp: 1 });
	append(assistant("old response ".repeat(4_000)));
	append({ role: "user", content: "recent question", timestamp: 2 });
	async function turn(
		tokens: number,
		willContinue: boolean,
		signal?: AbortSignal,
		stopReason: AssistantMessage["stopReason"] = "stop",
	) {
		const message = assistant("current response", tokens);
		message.stopReason = stopReason;
		append(message);
		const context: AgentTurnEndContext = { message, toolResults: [], willContinue };
		await boundary(agent.state.messages, signal, context);
		return message;
	}
	async function settle(messages = [...agent.state.messages]) {
		await dispatch({ type: "agent_end", messages });
	}
	function reviews(): string[] {
		return advisor.calls.map(call => JSON.stringify(call.context.messages));
	}
	return { session, agent, manager, settings, primary, advisor, summary, turn, settle, reviews };
}

describe("advisor maintenance ordering", () => {
	it("reviews compacted continuing context, but disabled ordering reviews the original context", async () => {
		for (const enabled of [true, false]) {
			const h = harness({ enabled });
			await h.turn(60_000, true);
			expect(h.manager.getBranch().some(entry => entry.type === "compaction")).toBe(true);
			expect(h.reviews()[0]?.includes("COMPACTED HISTORY")).toBe(enabled);
		}
	});

	it("leaves below-threshold and inactive sessions on their ordinary non-compacting path", async () => {
		const h = harness();
		await h.turn(100, true);
		expect(h.reviews()).toHaveLength(1);
		expect(h.summary.calls).toHaveLength(0);
		const inactive = harness({ active: false });
		await inactive.turn(100, true);
		expect(inactive.summary.calls).toHaveLength(0);
		expect(inactive.reviews()).toEqual([]);
	});

	it("defers within grace while speculation runs, then reviews the armed committed summary", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const h = harness({
			summarize: async () => {
				started.resolve();
				await release.promise;
				return { content: ["COMPACTED HISTORY"] };
			},
		});
		await h.turn(51_000, true);
		await started.promise;
		expect(h.session.compactionSpeculation).toBe("running");
		expect(h.reviews()[0]).not.toContain("COMPACTED HISTORY");
		await h.turn(52_000, true);
		expect(h.manager.getBranch().some(entry => entry.type === "compaction")).toBe(false);
		release.resolve();
		while (h.session.compactionSpeculation === "running") await Bun.sleep(1);
		expect(h.session.compactionSpeculation).toBe("armed");
		const speculativeRequests = h.summary.calls.length;
		await h.turn(53_000, true);
		expect(h.reviews().at(-1)).toContain("COMPACTED HISTORY");
		expect(h.summary.calls).toHaveLength(speculativeRequests);
	});

	it("consumes terminal review once after final maintenance and samples the setting at turn end", async () => {
		const h = harness();
		await h.turn(60_000, false);
		const settled = [...h.agent.state.messages];
		expect(h.reviews()).toEqual([]);
		h.settings.set("advisor.compactBeforeGuidance", false);
		await h.settle(settled);
		expect(h.reviews()).toHaveLength(1);
		expect(h.reviews()[0]).toContain("COMPACTED HISTORY");
		await h.settle(settled);
		expect(h.reviews()).toHaveLength(1);
	});

	it("uses the current roster after a terminal turn is reconfigured", async () => {
		const h = harness();
		const old = h.session.getAdvisorAgent();
		await h.turn(100, false);
		h.session.setAdvisorEnabled(false);
		h.session.setAdvisorEnabled(true);
		expect(h.session.getAdvisorAgent()).not.toBe(old);
		await h.settle();
		expect(h.reviews()).toHaveLength(0);
		await h.turn(100, true);
		expect(h.reviews()).toHaveLength(1);
	});

	it("does not review errored, aborted or invalidated terminal receipts", async () => {
		for (const stopReason of ["error", "aborted"] as const) {
			const h = harness();
			await h.turn(100, false, undefined, stopReason);
			expect(h.reviews()).toEqual([]);
		}
		const h = harness();
		await h.turn(100, false);
		const settled = [...h.agent.state.messages];
		await h.session.abort();
		await h.settle(settled);
		expect(h.reviews()).toEqual([]);
	});

	it("invalidates a pending receipt on clear, branch, switch, new session and disposal", async () => {
		for (const action of ["clear", "branch", "switch", "new", "dispose"] as const) {
			const h = harness();
			await h.turn(100, false);
			const settled = [...h.agent.state.messages];
			switch (action) {
				case "clear":
					await h.session.resetSessionContext();
					break;
				case "branch": {
					const entry = h.manager.getBranch().find(
						entry => entry.type === "message" && entry.message.role === "user",
					)!;
					await h.session.branch(entry.id);
					break;
				}
				case "switch": {
					const cwd = h.manager.getCwd();
					const target = SessionManager.create(cwd, cwd);
					target.appendMessage({ role: "user", content: "other session", timestamp: 3 });
					await target.flush();
					const file = target.getSessionFile()!;
					await target.close();
					await h.session.switchSession(file);
					break;
				}
				case "new":
					await h.session.newSession();
					break;
				case "dispose":
					h.session.beginDispose();
			}
			await h.settle(settled);
			expect(h.reviews()).toEqual([]);
		}
	});

	it("does not deliver a continuing review after abort interrupts maintenance", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const h = harness({
			summarize: async () => {
				started.resolve();
				await release.promise;
				return { content: ["COMPACTED HISTORY"] };
			},
		});
		const controller = new AbortController();
		const turn = h.turn(60_000, true, controller.signal);
		await started.promise;
		controller.abort();
		const aborted = h.session.abort();
		release.resolve();
		await Promise.all([turn, aborted]);
		expect(h.reviews()).toEqual([]);
		expect(h.manager.getBranch().some(entry => entry.type === "compaction")).toBe(false);
	});

	it("keeps ordinary review after a failed maintenance attempt", async () => {
		const h = harness({ summarize: { throw: new Error("Unauthorized test compactor") } });
		await h.turn(60_000, true);
		expect(h.manager.getBranch().some(entry => entry.type === "compaction")).toBe(false);
		expect(h.reviews()).toHaveLength(1);
		expect(h.reviews()[0]).not.toContain("COMPACTED HISTORY");
	});

	it("manual compaction cancels the old receipt and the next real prompt reviews current context", async () => {
		const h = harness();
		await h.turn(100, false);
		const settled = [...h.agent.state.messages];
		await h.session.compact();
		await h.settle(settled);
		expect(h.reviews()).toEqual([]);
		await h.session.prompt("Continue after manual compact");
		await h.session.waitForIdle();
		expect(h.reviews()).toHaveLength(1);
		expect(h.reviews()[0]).toContain("COMPACTED HISTORY");
	});

	it("reviews the final queued batch once without replaying a terminal receipt", async () => {
		const h = harness();
		h.agent.followUp({ role: "user", content: "queued follow-up", timestamp: Date.now() });
		await h.session.prompt("Start queued run");
		await h.session.waitForIdle();
		expect(h.primary.calls).toHaveLength(2);
		expect(h.reviews()).toHaveLength(1);
		expect(h.reviews().at(-1)).toContain("queued follow-up");
	});

	it("an unrelated stale settle cannot consume the current receipt", async () => {
		const h = harness();
		await h.turn(100, false);
		const stale = [...h.agent.state.messages];
		await h.turn(100, false);
		await h.settle(stale);
		expect(h.reviews()).toEqual([]);
		await h.settle();
		expect(h.reviews()).toHaveLength(1);
	});

	it("drops a terminal review if its turn signal aborts before final maintenance", async () => {
		const h = harness();
		const controller = new AbortController();
		await h.turn(100, false, controller.signal);
		controller.abort();
		await h.settle();
		expect(h.reviews()).toEqual([]);
	});

	it("does not reuse an earlier successful receipt when a queued follow-up fails", async () => {
		const h = harness();
		h.settings.set("retry.enabled", false);
		h.primary.push({ content: ["first answer"] });
		h.primary.push({ content: ["failed answer"], stopReason: "error", errorMessage: "Unauthorized test turn" });
		h.agent.followUp({ role: "user", content: "failing follow-up", timestamp: Date.now() });
		await h.session.prompt("Start queued failure");
		await h.session.waitForIdle();
		expect(h.primary.calls).toHaveLength(2);
		expect(h.reviews()).toEqual([]);
	});
});
