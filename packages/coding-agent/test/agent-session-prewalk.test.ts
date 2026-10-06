import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool, ThinkingLevel } from "@oh-my-pi/pi-agent-core";
import { type Api, Effort, type Model } from "@oh-my-pi/pi-ai";
import { createMockModel, type MockResponse } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import type { InteractiveModeContext } from "@oh-my-pi/pi-coding-agent/modes/types";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { PrewalkSnapshot } from "@oh-my-pi/pi-coding-agent/session/prewalk";
import { executeBuiltinSlashCommand } from "@oh-my-pi/pi-coding-agent/slash-commands/builtin-registry";
import type { TuiSlashCommandRuntime } from "@oh-my-pi/pi-coding-agent/slash-commands/types";
import { createSubagentSettings } from "@oh-my-pi/pi-coding-agent/task/executor";
import { AUTO_THINKING } from "@oh-my-pi/pi-tui/thinking";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

/**
 * Prewalk switches once from planning to execution after an edit/write with
 * the todo gate open, or an optional completed-action maximum. A minimum
 * delays eligibility; early implementation actions are discarded. Hidden
 * plan/continuation nudges precede the switch, with a verification checklist
 * afterward. See the unchanged prompts under `src/prompts/system/prewalk-*.md`.
 */
describe("AgentSession prewalk", () => {
	let tempDir: TempDir;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;
	let session: AgentSession | undefined;

	beforeAll(() => {
		tempDir = TempDir.createSync("@pi-prewalk-");
		authStorage = createInMemoryAuthStorage();
		authStorage.keys.setRuntime("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage, path.join(tempDir.path(), "models.yml"));
	});

	afterEach(async () => {
		if (session) await session.dispose();
		session = undefined;
	});

	afterAll(() => {
		authStorage.close();
		tempDir.removeSync();
	});

	function modelOrThrow(id: string): Model<Api> {
		const model = getBundledModel("anthropic", id);
		if (!model) throw new Error(`Expected bundled model ${id}`);
		return model;
	}

	const recordToolSchema = type({});
	const recordTool: AgentTool<typeof recordToolSchema, undefined> = {
		name: "record",
		label: "Record",
		description: "Read-only step",
		parameters: recordToolSchema,
		async execute() {
			return { content: [{ type: "text", text: "ok" }], details: undefined };
		},
	};
	const bashToolSchema = type({});
	const bashTool: AgentTool<typeof bashToolSchema, undefined> = {
		name: "bash",
		label: "Bash",
		description: "Run a command",
		parameters: bashToolSchema,
		async execute() {
			return { content: [{ type: "text", text: "ran" }], details: undefined };
		},
	};
	const writeToolSchema = type({});
	const writeTool: AgentTool<typeof writeToolSchema, undefined> = {
		name: "write",
		label: "Write",
		description: "Write a file",
		parameters: writeToolSchema,
		async execute() {
			return { content: [{ type: "text", text: "wrote" }], details: undefined };
		},
	};
	const todoToolSchema = type({});
	const todoTool: AgentTool<typeof todoToolSchema, undefined> = {
		name: "todo",
		label: "Todo",
		description: "Track tasks",
		parameters: todoToolSchema,
		async execute() {
			return { content: [{ type: "text", text: "listed" }], details: undefined };
		},
	};
	const toolRegistry = new Map<string, AgentTool>([
		[recordTool.name, recordTool as AgentTool],
		[bashTool.name, bashTool as AgentTool],
		[writeTool.name, writeTool as AgentTool],
		[todoTool.name, todoTool as AgentTool],
	]);

	function toolCall(id: string, name: string): MockResponse {
		return { content: [{ type: "toolCall", id, name, arguments: {} }], stopReason: "toolUse" };
	}

	it("prewalks at the first edit/write after the todo gate opens; bash and todo don't trigger", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// Turn 1: read-only. Turn 2: bash is excluded. Turn 3: todo opens the gate.
		// Turn 4: write is the first post-todo edit/write, so it switches.
		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				toolCall("t2", "bash"),
				toolCall("t3", "todo"),
				toolCall("t4", "write"),
				{ content: ["done"] },
			],
		});
		const calls: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, bashTool as AgentTool, writeTool as AgentTool, todoTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, _context, options) => {
				calls.push(`${model.provider}/${model.id}`);
				return mock.stream(model, _context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("do the task");

		expect(calls).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("an edit before any todo call does not switch while a todo tool exists; the next edit after todo does", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// Turn 1: exploration. Turn 2: write while the gate is closed.
		// Turn 3: todo opens the gate. Turn 4: write switches.
		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				toolCall("t2", "write"),
				toolCall("t3", "todo"),
				toolCall("t4", "write"),
				{ content: ["done"] },
			],
		});
		const calls: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool, todoTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, _context, options) => {
				calls.push(`${model.provider}/${model.id}`);
				return mock.stream(model, _context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("do the task");

		expect(calls).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("keeps the todo gate closed after a failed todo call", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		const failingTodoTool: AgentTool<typeof todoToolSchema, undefined> = {
			...todoTool,
			async execute() {
				return {
					content: [{ type: "text", text: "todo update failed" }],
					details: undefined,
					isError: true,
				};
			},
		};
		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "todo"), toolCall("t3", "write"), { content: ["done"] }],
		});
		const calls: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool, failingTodoTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				calls.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry: new Map([...toolRegistry, ["todo", failingTodoTool as AgentTool]]),
			prewalk: { target },
		});

		await session.prompt("do the task");

		expect(calls).toEqual(Array(5).fill(`${primary.provider}/${primary.id}`));
		expect(session.model?.id).toBe(primary.id);
	});

	it("forces a continuation when the plan nudge gets a text-only reply, instead of silently ending the run", async () => {
		// Regression: the agent loop treats a turn with zero tool calls as a
		// natural stop boundary and ends the session with no further prompting.
		// The plan nudge explicitly asks for a prose reply, making this common
		// right after it — observed killing production runs before any code
		// was written. The safety net must force one more turn.
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				{ content: [{ type: "text", text: "Let me think about this for a moment." }], stopReason: "stop" },
				toolCall("t3", "write"),
				{ content: ["done"] },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry: new Map([
				[recordTool.name, recordTool as AgentTool],
				[writeTool.name, writeTool as AgentTool],
			]),
			prewalk: { target },
		});

		await session.prompt("do the task");

		// All 4 turns must run — the text-only turn 2 must not end the session early.
		expect(requested).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("bounds a completed bash-only task to a single continuation instead of looping", async () => {
		// Regression (#5551): with no edit/write ever run, the continuation net
		// used to re-fire on every text-only reply, looping forever. It must
		// fire at most once — one "continue" nudge — then let the next text-only
		// reply end the run. No mock fallback: a stray extra turn rejects.
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// Turn 1: record (nudge injected after). Turn 2: bash — not an action
		// tool. Turn 3: prose — the single continuation fires. Turn 4: prose
		// again — no more continuation, run ends. A 5th call would exhaust the
		// script and reject.
		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				toolCall("t2", "bash"),
				{ content: [{ type: "text", text: "Commit complete." }], stopReason: "stop" },
				{ content: [{ type: "text", text: "Nothing left to do." }], stopReason: "stop" },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, bashTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("commit the current changes");

		// Exactly one continuation: 4 turns, all on the primary (no edit/write,
		// so no switch), then a clean stop.
		expect(requested).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
		]);
		expect(session.model?.id).toBe(primary.id);
	});

	it("does not switch on a read-only xd:// device dispatched through write (issue #7312)", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// A read-only lsp navigation is dispatched as `write xd://lsp`; the write
		// result carries the wrapped tool's read tier. Like a bash step, it must
		// not arm the hand-off — the model keeps reasoning about code shape on the
		// strong model. Mirrors the bounded-continuation flow: one continuation,
		// four turns, all primary, then a clean stop.
		const readDeviceWrite: AgentTool<typeof writeToolSchema, { xdev: { tool: string; mode: string; tier: string } }> =
			{
				name: "write",
				label: "Write",
				description: "Dispatch a read-only device",
				parameters: writeToolSchema,
				async execute() {
					return {
						content: [{ type: "text", text: "references" }],
						details: { xdev: { tool: "lsp", mode: "execute", tier: "read" } },
					};
				},
			};
		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				toolCall("t2", "write"),
				{ content: [{ type: "text", text: "Still planning." }], stopReason: "stop" },
				{ content: [{ type: "text", text: "Done planning." }], stopReason: "stop" },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, readDeviceWrite as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry: new Map([
				[recordTool.name, recordTool as AgentTool],
				[readDeviceWrite.name, readDeviceWrite as AgentTool],
			]),
			prewalk: { target },
		});

		await session.prompt("investigate the code shape");

		expect(requested).toEqual(Array(4).fill(`${primary.provider}/${primary.id}`));
		expect(session.model?.id).toBe(primary.id);
	});

	it("switches on a write-tier xd:// device dispatched through write (issue #7312)", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// An lsp rename is a write-tier device call — it must arm the hand-off
		// just like a direct edit/write: the write turn stays on the strong model,
		// the next turn runs on the target.
		const writeDeviceWrite: AgentTool<
			typeof writeToolSchema,
			{ xdev: { tool: string; mode: string; tier: string } }
		> = {
			name: "write",
			label: "Write",
			description: "Dispatch a write-tier device",
			parameters: writeToolSchema,
			async execute() {
				return {
					content: [{ type: "text", text: "renamed" }],
					details: { xdev: { tool: "lsp", mode: "execute", tier: "write" } },
				};
			},
		};
		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeDeviceWrite as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry: new Map([
				[recordTool.name, recordTool as AgentTool],
				[writeDeviceWrite.name, writeDeviceWrite as AgentTool],
			]),
			prewalk: { target },
		});

		await session.prompt("rename the symbol");

		expect(requested).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("re-arms continuation after tool progress between prose turns", async () => {
		// Regression: a normal prewalk can split planning across several turns:
		// prose plan, todo init, then prose before implementation. Each tool
		// progress segment must earn one continuation so the second prose turn
		// cannot end the run before edit/write.
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// Turn 1: read-only (nudge injected after). Turn 2: prose plan —
		// bridged. Turn 3: todo — gate opens and re-arms the net. Turn 4:
		// prose — bridged again. Turn 5: write — switch.
		const mock = createMockModel({
			responses: [
				toolCall("t1", "record"),
				{ content: [{ type: "text", text: "Here is the plan." }], stopReason: "stop" },
				toolCall("t3", "todo"),
				{ content: [{ type: "text", text: "Plan captured, starting now." }], stopReason: "stop" },
				toolCall("t5", "write"),
				{ content: ["done"] },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool, todoTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("do the task");

		expect(requested).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("skips the todo gate when todo is registered but not active (subagent-style restricted slates)", async () => {
		// Regression: the gate used to key on the tool REGISTRY, so a session
		// whose active-tool slate excluded `todo` (subagents strip it) while the
		// registry still contained it could never open the gate — the model
		// cannot call an inactive tool — and prewalk never fired.
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		// Turn 1: read-only (nudge injected after). Turn 2: write — first
		// edit/write must switch immediately; no todo call is possible.
		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				// Active slate excludes todo; the session toolRegistry still has it.
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("do the task");

		expect(requested).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);
	});

	it("armPrewalk (the /prewalk slash command) pre-arms the switch for the very next edit/write", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		const sessionManager = SessionManager.inMemory();
		sessionManager.appendCustomMessageEntry(
			"prewalk-plan",
			"legacy plan nudge written by an older OMP version",
			false,
			undefined,
			"agent",
		);

		// No `prewalk` in the session config — this simulates a session that
		// was NOT started with --prewalk, forced on via the slash command.
		const mock = createMockModel({ responses: [toolCall("t1", "write"), { content: ["done"] }] });
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager,
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry: new Map([[writeTool.name, writeTool as AgentTool]]),
		});

		// Arming twice back-to-back must stay a single, idempotent arm.
		expect(session.armPrewalk(target)).toBe(true);
		expect(session.armPrewalk(target)).toBe(true);

		await session.prompt("do the task");

		// Pre-armed before the first turn: the very first write call switches
		// immediately — no second primary-model turn needed.
		expect(requested).toEqual([`${primary.provider}/${primary.id}`, `${target.provider}/${target.id}`]);
		expect(session.model?.id).toBe(target.id);
		expect(
			sessionManager
				.buildSessionContext()
				.messages.some(message => message.role === "custom" && message.customType === "prewalk-plan"),
		).toBe(false);
		// The seeded legacy entry must remain the only transcript copy; persisting
		// the current arm's transient nudge would make this count two.
		expect(
			sessionManager
				.buildSessionContext({ transcript: true })
				.messages.filter(message => message.role === "custom" && message.customType === "prewalk-plan"),
		).toHaveLength(1);
	});

	it("armPrewalk rejects a same-model same-effort no-op", async () => {
		const model = modelOrThrow("claude-sonnet-4-5");

		const mock = createMockModel({ responses: [{ content: ["status only"] }] });
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (streamModel, _context, options) => mock.stream(streamModel, _context, options),
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.Medium,
		});
		const notices: string[] = [];
		session.subscribe(event => {
			if (event.type === "notice" && event.source === "prewalk") notices.push(event.message);
		});

		expect(session.armPrewalk(model, Effort.Medium)).toBe(false);
		await session.prompt("report current status");

		expect(notices.some(message => message.includes("nothing to switch"))).toBe(true);
	});

	it("/prewalk commands report success only when the requested arm becomes active", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		const settings = Settings.isolated({ "compaction.enabled": false });
		const sessionManager = SessionManager.inMemory();
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
		});
		session = new AgentSession({
			agent,
			sessionManager,
			settings,
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.Medium,
		});
		const showStatus = vi.fn();
		const ctx = {
			session,
			sessionManager,
			settings,
			collabGuest: false,
			showStatus,
			editor: { setText: vi.fn() },
			refreshSlashCommandState: vi.fn(),
		} as unknown as InteractiveModeContext;
		const runtime = { ctx } satisfies TuiSlashCommandRuntime;

		settings.setModelRole("smol", `${primary.provider}/${primary.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk", runtime)).toBe(true);
		expect(showStatus).not.toHaveBeenCalled();

		settings.setModelRole("smol", `${target.provider}/${target.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk", runtime)).toBe(true);
		expect(showStatus).toHaveBeenCalledTimes(1);
		expect(session.getPrewalkState()?.target.id).toBe(target.id);

		// A different request cannot report success while the prior target remains armed.
		settings.setModelRole("smol", `${primary.provider}/${primary.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk", runtime)).toBe(true);
		expect(showStatus).toHaveBeenCalledTimes(1);

		// Restart must not move the active model when an existing arm rejects the requested target.
		settings.setModelRole("default", `${target.provider}/${target.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk restart", runtime)).toBe(true);
		expect(session.model?.id).toBe(primary.id);
		expect(session.getPrewalkState()?.target.id).toBe(target.id);
		expect(showStatus).toHaveBeenCalledTimes(1);

		// A matching arm remains active while restart restores the configured planning model.
		await session.setModelTemporary(target, Effort.Medium, { ephemeral: true });
		settings.setModelRole("default", `${primary.provider}/${primary.id}:medium`);
		settings.setModelRole("smol", `${target.provider}/${target.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk restart", runtime)).toBe(true);
		expect(session.model?.id).toBe(primary.id);
		expect(session.getPrewalkState()?.target.id).toBe(target.id);
		expect(showStatus).toHaveBeenCalledTimes(2);

		// If both roles now coincide, restart resets the model and clears the obsolete matching arm.
		settings.setModelRole("default", `${target.provider}/${target.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk restart", runtime)).toBe(true);
		expect(session.model?.id).toBe(target.id);
		expect(session.getPrewalkState()).toBeUndefined();
		expect(
			agent.state.messages.some(message => message.role === "custom" && message.customType === "prewalk-plan"),
		).toBe(false);
		expect(showStatus).toHaveBeenCalledTimes(3);
		expect(showStatus).toHaveBeenCalledWith(expect.stringContaining("Prewalk reset"));
	});

	it("/prewalk restart returns to @default and re-arms @smol", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");
		const mock = createMockModel({
			responses: [
				toolCall("first-todo", "todo"),
				toolCall("first-write", "write"),
				{ content: ["first done"] },
				toolCall("second-todo", "todo"),
				toolCall("second-write", "write"),
				{ content: ["second done"] },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [todoTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false });
		settings.setModelRole("default", `anthropic/missing-model,${primary.provider}/${primary.id}:medium`);
		settings.setModelRole("smol", `${target.provider}/${target.id}:medium`);
		const sessionManager = SessionManager.inMemory();
		session = new AgentSession({
			agent,
			sessionManager,
			settings,
			modelRegistry,
			toolRegistry,
			prewalk: { target, thinkingLevel: Effort.Medium },
			thinkingLevel: Effort.Medium,
		});

		await session.prompt("first task");
		expect(session.model?.id).toBe(target.id);
		const firstRunCallCount = requested.length;

		const showStatus = vi.fn();
		const ctx = {
			session,
			sessionManager,
			settings,
			collabGuest: false,
			showStatus,
			editor: { setText: vi.fn() },
			refreshSlashCommandState: vi.fn(),
		} as unknown as InteractiveModeContext;
		const runtime = { ctx } satisfies TuiSlashCommandRuntime;

		expect(await executeBuiltinSlashCommand("/prewalk restart", runtime)).toBe(true);
		expect(session.model?.id).toBe(primary.id);
		expect(session.getPrewalkState()?.target.id).toBe(target.id);
		expect(showStatus).toHaveBeenCalledTimes(1);
		expect(showStatus).toHaveBeenCalledWith(expect.stringContaining("Prewalk restarted"));
		expect(showStatus).toHaveBeenCalledWith(expect.stringContaining(`${primary.provider}/${primary.id}`));
		expect(showStatus).toHaveBeenCalledWith(expect.stringContaining(`${target.provider}/${target.id}`));

		await session.prompt("second task");
		expect(requested.slice(firstRunCallCount)).toEqual([
			`${primary.provider}/${primary.id}`,
			`${primary.provider}/${primary.id}`,
			`${target.provider}/${target.id}`,
		]);
		expect(session.model?.id).toBe(target.id);

		settings.setModelRole("smol", `${primary.provider}/${primary.id}:medium`);
		expect(await executeBuiltinSlashCommand("/prewalk restart", runtime)).toBe(true);
		expect(session.model?.id).toBe(primary.id);
		expect(session.getPrewalkState()).toBeUndefined();
		expect(showStatus).toHaveBeenCalledTimes(2);
		expect(showStatus).toHaveBeenCalledWith(expect.stringContaining("Prewalk reset"));
	});

	it("requires a fresh todo before a later explicit prewalk can hand off", async () => {
		const primary = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");

		const mock = createMockModel({
			responses: [
				toolCall("first-todo", "todo"),
				toolCall("first-write", "write"),
				{ content: ["first done"] },
				toolCall("second-write-before-todo", "write"),
				toolCall("second-todo", "todo"),
				toolCall("second-write-after-todo", "write"),
				{ content: ["second done"] },
			],
		});
		const requested: string[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: primary,
				systemPrompt: ["Test"],
				tools: [todoTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (model, context, options) => {
				requested.push(`${model.provider}/${model.id}`);
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			prewalk: { target },
		});

		await session.prompt("first task");
		const firstRunCallCount = requested.length;
		expect(session.model?.id).toBe(target.id);

		session.armPrewalk(primary);
		await session.prompt("second task");

		expect(requested.slice(firstRunCallCount)).toEqual([
			`${target.provider}/${target.id}`,
			`${target.provider}/${target.id}`,
			`${target.provider}/${target.id}`,
			`${primary.provider}/${primary.id}`,
		]);
		expect(session.model?.id).toBe(primary.id);
	});

	it("effort-only prewalk on the same model downgrades the thinking level instead of silently skipping", async () => {
		// Regression (#6659): the switch guard compared model identity only, so a
		// same-model target at a cheaper thinking level (a legitimate effort
		// downgrade, common with role aliases like `prewalk: "@task"`) was dropped
		// as a no-op. On a reasoning model the effort is the bulk of the cost, so
		// this must still switch.
		const model = modelOrThrow("claude-sonnet-4-5");

		// todo excluded from the active slate → the gate opens; record then write.
		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (streamModel, _context, options) => mock.stream(streamModel, _context, options),
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.Medium,
			prewalk: { target: model, thinkingLevel: Effort.Low },
		});

		expect(session.thinkingLevel).toBe(Effort.Medium);

		await session.prompt("do the task");

		// The model id never changes, but the effort drops after the first write.
		expect(session.model?.id).toBe(model.id);
		expect(session.thinkingLevel).toBe(Effort.Low);
	});

	it("emits a notice when the prewalk target is a genuine no-op", async () => {
		// Same model and same effective thinking level: no state change.
		const model = modelOrThrow("claude-sonnet-4-5");

		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (streamModel, _context, options) => mock.stream(streamModel, _context, options),
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.Medium,
			prewalk: { target: model, thinkingLevel: Effort.Medium },
		});
		const notices: string[] = [];
		session.subscribe(event => {
			if (event.type === "notice" && event.source === "prewalk") notices.push(event.message);
		});

		await session.prompt("do the task");

		expect(session.model?.id).toBe(model.id);
		expect(session.thinkingLevel).toBe(Effort.Medium);
		// The no-op is announced, not silent.
		expect(notices.some(message => message.includes("nothing to switch"))).toBe(true);
	});

	it("treats a target effort the model clamps back to the active effort as a no-op", async () => {
		// A model capped at high resolves an xhigh target back to high.
		// The equal effective settings must be recognized as a no-op.
		const model = modelOrThrow("claude-sonnet-4-6"); // supported efforts cap at high

		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.High,
			},
			convertToLlm,
			streamFn: (streamModel, _context, options) => mock.stream(streamModel, _context, options),
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.High,
			prewalk: { target: model, thinkingLevel: Effort.XHigh },
		});
		const notices: string[] = [];
		session.subscribe(event => {
			if (event.type === "notice" && event.source === "prewalk") notices.push(event.message);
		});

		await session.prompt("do the task");

		expect(session.thinkingLevel).toBe(Effort.High);
		expect(notices.some(message => message.includes("nothing to switch"))).toBe(true);
	});

	it("switches when a same-model target clears auto mode even though efforts both resolve to undefined", async () => {
		// Review edge case: session in `auto`, same-model prewalk target `:inherit`.
		// Both selectors resolve to an `undefined` effort, but `:inherit` clears
		// per-turn classification, so this is a real change and must switch — not
		// collapse to a no-op.
		const model = modelOrThrow("claude-sonnet-4-5");

		const mock = createMockModel({
			responses: [toolCall("t1", "record"), toolCall("t2", "write"), { content: ["done"] }],
		});
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["Test"],
				tools: [recordTool as AgentTool, writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.Medium,
			},
			convertToLlm,
			streamFn: (streamModel, _context, options) => mock.stream(streamModel, _context, options),
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry,
			toolRegistry,
			thinkingLevel: AUTO_THINKING,
			prewalk: { target: model, thinkingLevel: ThinkingLevel.Inherit },
		});
		const notices: string[] = [];
		session.subscribe(event => {
			if (event.type === "notice" && event.source === "prewalk") notices.push(event.message);
		});

		expect(session.isAutoThinking).toBe(true);

		await session.prompt("do the task");

		// The hand-off clears automatic thinking.
		expect(session.isAutoThinking).toBe(false);
		expect(notices.some(message => message.includes("nothing to switch"))).toBe(false);
	});
	function cycleHarness(
		responses: MockResponse[],
		options: {
			explicit?: boolean;
			disabled?: boolean;
			manager?: AsyncJobManager;
			persistent?: boolean;
			minimum?: number;
			maximum?: number;
			baseSettings?: Settings;
			snapshot?: PrewalkSnapshot;
		} = {},
	) {
		const source = modelOrThrow("claude-sonnet-4-5");
		const target = modelOrThrow("claude-sonnet-4-6");
		const settings = options.baseSettings
			? createSubagentSettings(options.baseSettings)
			: Settings.isolated({ "compaction.enabled": false, "prewalk.afterEveryUserMessage": true });
		settings.set("prewalk.minMessages", options.minimum ?? 0);
		settings.set("prewalk.maxMessages", options.maximum ?? 0);
		settings.setModelRole("default", source.provider + "/" + source.id + ":high");
		settings.setModelRole("smol", target.provider + "/" + target.id + ":low");
		const mock = createMockModel({ responses });
		const requests: { model: string; effort: unknown }[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: options.explicit === false ? target : source,
				systemPrompt: ["Test"],
				tools: [writeTool as AgentTool],
				messages: [],
				thinkingLevel: Effort.High,
			},
			convertToLlm,
			streamFn: (model, context, streamOptions) => {
				requests.push({ model: model.id, effort: streamOptions?.reasoning });
				return mock.stream(model, context, streamOptions);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: options.persistent
				? SessionManager.create(process.cwd(), tempDir.path())
				: SessionManager.inMemory(),
			settings,
			agentKind: options.baseSettings ? "sub" : "main",
			modelRegistry,
			toolRegistry,
			thinkingLevel: Effort.High,
			prewalk: options.disabled
				? false
				: options.explicit === false
					? undefined
					: { target, thinkingLevel: Effort.Low },
			asyncJobManager: options.manager,
			prewalkSnapshot: options.snapshot,
		});
		return { source, target, requests, agent, settings, session };
	}

	it("/noprewalk strips its prefix, skips only that input, and leaves later replanning enabled", async () => {
		const t = cycleHarness([{ content: ["quick answer"] }, toolCall("write", "write"), { content: ["done"] }], {
			explicit: false,
		});
		t.settings.set("prewalk.enabled", true);
		const ctx = {
			session: t.session,
			sessionManager: t.session.sessionManager,
			settings: t.settings,
			collabGuest: false,
			showStatus: vi.fn(),
			editor: { setText: vi.fn() },
			refreshSlashCommandState: vi.fn(),
		} as unknown as InteractiveModeContext;
		const body = await executeBuiltinSlashCommand("/noprewalk quick question", { ctx });
		expect(body).toBe("quick question");
		if (typeof body !== "string") throw new Error("Expected submitted body");
		await t.session.prompt(body);
		expect(t.requests.map(request => request.model)).toEqual([t.target.id]);
		expect(
			t.agent.state.messages.some(message => message.role === "custom" && message.customType === "prewalk-plan"),
		).toBe(false);
		await t.session.prompt("implement next task");
		expect(t.requests.map(request => request.model)).toEqual([t.target.id, t.source.id, t.target.id]);
	});
	it("bare /noprewalk waits for actual user input, not internal continuations or hidden companions", async () => {
		const t = cycleHarness(
			[
				{ content: ["internal"] },
				{ content: ["companion"] },
				{ content: ["quick"] },
				toolCall("write", "write"),
				{ content: ["done"] },
			],
			{ explicit: false },
		);
		t.settings.set("prewalk.enabled", true);
		const ctx = {
			session: t.session,
			sessionManager: t.session.sessionManager,
			settings: t.settings,
			collabGuest: false,
			showStatus: vi.fn(),
			editor: { setText: vi.fn() },
			refreshSlashCommandState: vi.fn(),
		} as unknown as InteractiveModeContext;
		expect(await executeBuiltinSlashCommand("/noprewalk", { ctx })).toBe(true);
		await t.session.prompt("internal", { synthetic: true });
		await t.session.sendCustomMessage(
			{ customType: "image-attachment-description", content: "hidden", display: false, attribution: "user" },
			{ triggerTurn: true },
		);
		await t.session.prompt("quick question");
		await t.session.prompt("next task");
		expect(t.requests.map(request => request.model)).toEqual([
			t.target.id,
			t.target.id,
			t.target.id,
			t.source.id,
			t.target.id,
		]);
	});
	for (const delivery of ["followUp", "steer", "custom"] as const) {
		it("consumes the bypass when " + delivery + " user input is admitted during execution", async () => {
			const t = cycleHarness([
				toolCall("first", "write"),
				{ content: ["done"] },
				toolCall("read", "record"),
				{ content: ["quick"] },
				...(delivery === "followUp" ? [{ content: ["follow-up answer"] }] : []),
				toolCall("next", "write"),
				{ content: ["done"] },
			]);
			await t.session.prompt("first task");
			t.session.setInterruptMode("wait");
			t.agent.setTools([
				{
					...recordTool,
					async execute() {
						t.session.skipNextPrewalk();
						if (delivery === "followUp") await t.session.followUp("quick");
						else if (delivery === "steer") await t.session.sendUserMessage("quick", { deliverAs: "steer" });
						else
							await t.session.sendCustomMessage(
								{ customType: "user-submission", content: "quick", display: true, attribution: "user" },
								{ triggerTurn: true },
							);
						return { content: [{ type: "text", text: "read" }], details: undefined };
					},
				} as AgentTool,
				writeTool as AgentTool,
			]);
			await t.session.prompt("internal continuation", { synthetic: true });
			await t.session.waitForIdle();
			expect(t.requests.map(request => request.model)).toEqual([
				t.source.id,
				t.target.id,
				t.target.id,
				t.target.id,
				...(delivery === "followUp" ? [t.target.id] : []),
			]);
			await t.session.prompt("next task");
			expect(t.requests.slice(-2).map(request => request.model)).toEqual([t.source.id, t.target.id]);
		});
	}
	it("suppresses a new repeat nudge without cancelling the active prewalk or its initial nudge", async () => {
		const t = cycleHarness([
			toolCall("read-1", "record"),
			toolCall("read-2", "record"),
			toolCall("write", "write"),
			{ content: ["done"] },
		]);
		t.settings.set("prewalk.repeatPlanNudge", true);
		t.session.setInterruptMode("wait");
		let plans = 0;
		t.agent.subscribe(event => {
			if (
				event.type === "message_end" &&
				event.message.role === "custom" &&
				event.message.customType === "prewalk-plan"
			)
				plans++;
		});
		let reads = 0;
		t.agent.setTools([
			{
				...recordTool,
				async execute() {
					if (++reads === 2) {
						t.session.skipNextPrewalk();
						await t.session.sendUserMessage("small clarification", { deliverAs: "steer" });
					}
					return { content: [{ type: "text", text: "read" }], details: undefined };
				},
			} as AgentTool,
			writeTool as AgentTool,
		]);
		await t.session.prompt("initial task");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.source.id, t.target.id]);
		expect(plans).toBe(1);
	});
	it("/noprewalk suppresses the delayed first plan nudge in an already-armed cycle", async () => {
		const t = cycleHarness([
			{ content: ["quick answer"] },
			toolCall("read", "record"),
			toolCall("write", "write"),
			{ content: ["done"] },
		]);
		t.agent.setTools([recordTool as AgentTool, writeTool as AgentTool]);
		let plans = 0;
		t.agent.subscribe(event => {
			if (
				event.type === "message_end" &&
				event.message.role === "custom" &&
				event.message.customType === "prewalk-plan"
			)
				plans++;
		});
		const ctx = {
			session: t.session,
			sessionManager: t.session.sessionManager,
			settings: t.settings,
			collabGuest: false,
			showStatus: vi.fn(),
			editor: { setText: vi.fn() },
			refreshSlashCommandState: vi.fn(),
		} as unknown as InteractiveModeContext;
		const body = await executeBuiltinSlashCommand("/noprewalk quick question", { ctx });
		if (typeof body !== "string") throw new Error("Expected submitted body");
		await t.session.prompt(body);
		expect(plans).toBe(0);
		expect(t.requests.map(request => request.model)).toEqual([t.source.id]);
		expect(t.session.getPrewalkState()?.target.id).toBe(t.target.id);
		await t.session.prompt("resume normal task");
		expect(plans).toBe(1);
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.source.id, t.target.id]);
	});
	it("retains the bypass through tools, synthetic continuations and hot revival without resetting handoff accounting", async () => {
		const t = cycleHarness([toolCall("first", "record"), { content: ["pause"] }], { minimum: 9, maximum: 9 });
		t.agent.setTools([recordTool as AgentTool]);
		t.session.skipNextPrewalk();
		await t.session.prompt("quick investigation");
		const snapshot = t.session.getPrewalkSnapshot();
		const messages = structuredClone(t.agent.state.messages);
		await t.session.dispose();
		const revived = cycleHarness(
			[
				toolCall("second", "record"),
				toolCall("third", "record"),
				{ content: ["pause"] },
				toolCall("fourth", "record"),
				{ content: ["done"] },
			],
			{ minimum: 9, maximum: 9, snapshot },
		);
		revived.agent.setTools([recordTool as AgentTool]);
		revived.agent.replaceMessages(messages);
		const nudges: string[] = [];
		revived.agent.subscribe(event => {
			if (event.type === "message_end" && event.message.role === "custom") nudges.push(event.message.customType);
		});
		await revived.session.prompt("continue", { synthetic: true });
		expect(revived.requests.map(request => request.model)).toEqual([
			revived.source.id,
			revived.source.id,
			revived.source.id,
		]);
		expect(nudges).toEqual([]);
		await revived.session.prompt("continue again", { synthetic: true });
		expect(revived.requests.map(request => request.model)).toEqual([
			revived.source.id,
			revived.source.id,
			revived.source.id,
			revived.source.id,
			revived.target.id,
		]);
		expect(nudges).toEqual(["prewalk-checklist"]);
	});
	it("ordinary input in a mixed steering batch still restarts the cycle and its action budget", async () => {
		const t = cycleHarness(
			[
				toolCall("read", "record"),
				toolCall("write-1", "write"),
				toolCall("write-2", "write"),
				{ content: ["done"] },
			],
			{ minimum: 4 },
		);
		t.session.setInterruptMode("wait");
		t.agent.setSteeringMode("all");
		t.agent.setTools([
			{
				...recordTool,
				async execute() {
					t.session.skipNextPrewalk();
					await t.session.sendUserMessage("quick clarification", { deliverAs: "steer" });
					await t.session.sendUserMessage("normal task", { deliverAs: "steer" });
					return { content: [{ type: "text", text: "read" }], details: undefined };
				},
			} as AgentTool,
			writeTool as AgentTool,
		]);
		await t.session.prompt("initial task");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.source.id, t.target.id]);
	});
	it("clears an unused bypass at context reset rather than leaking it into a new task", async () => {
		const t = cycleHarness([toolCall("write", "write"), { content: ["done"] }], { explicit: false });
		t.settings.set("prewalk.enabled", true);
		t.session.skipNextPrewalk();
		await t.session.resetSessionContext();
		await t.session.prompt("new task");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id]);
	});
	it("does not disable an explicit prewalk when per-user replanning is off", async () => {
		const t = cycleHarness([toolCall("write", "write"), { content: ["done"] }]);
		t.settings.set("prewalk.afterEveryUserMessage", false);
		t.session.skipNextPrewalk();
		await t.session.prompt("initial task");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id]);
	});
	for (const explicitArm of [false, true]) {
		it(
			"disables planning nudges without disabling " + (explicitArm ? "explicit" : "startup") + " handoff",
			async () => {
				const t = cycleHarness([toolCall("read", "record"), toolCall("write", "write"), { content: ["done"] }], {
					disabled: explicitArm,
				});
				t.settings.set("prewalk.planNudge", false);
				if (explicitArm) expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
				const nudges: string[] = [];
				t.agent.subscribe(event => {
					if (
						event.type === "message_end" &&
						event.message.role === "custom" &&
						event.message.customType.startsWith("prewalk-")
					)
						nudges.push(event.message.customType);
				});
				t.agent.setTools([recordTool as AgentTool, writeTool as AgentTool]);
				await t.session.prompt("investigate and implement");
				expect(nudges).toEqual(["prewalk-checklist"]);
				expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
			},
		);
	}
	it("continuation off lets an answer finish below the handoff minimum while retaining the plan and later handoff", async () => {
		const t = cycleHarness(
			[
				toolCall("read", "record"),
				{ content: ["investigation complete"] },
				toolCall("todo", "todo"),
				toolCall("write", "write"),
				{ content: ["implemented"] },
			],
			{ minimum: 5 },
		);
		t.settings.set("prewalk.continueNudge", false);
		t.agent.setTools([recordTool as AgentTool, todoTool as AgentTool, writeTool as AgentTool]);
		const nudges: string[] = [];
		t.agent.subscribe(event => {
			if (event.type === "message_end" && event.message.role === "custom") nudges.push(event.message.customType);
		});
		await t.session.prompt("investigate");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id]);
		expect(nudges).toEqual(["prewalk-plan"]);
		await t.session.prompt("implement the result", { synthetic: true });
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.source.id,
			t.source.id,
			t.source.id,
			t.target.id,
		]);
		expect(nudges).toEqual(["prewalk-plan", "prewalk-checklist"]);
	});

	for (const optIn of [false, true]) {
		it(
			optIn
				? "child continuation opt-in forces an extra request even when the parent disables continuation"
				: "child continuation defaults off and finishes despite the parent enabling continuation",
			async () => {
				const baseSettings = Settings.isolated({
					"compaction.enabled": false,
					"prewalk.planNudge": false,
					"prewalk.continueNudge": !optIn,
					...(optIn ? { "task.prewalkContinueNudge": true } : {}),
				});
				const t = cycleHarness(
					[
						toolCall("read", "record"),
						{ content: ["child answer"] },
						...(optIn ? [{ content: ["child finished"] }] : []),
					],
					{ baseSettings, minimum: 9 },
				);
				t.agent.setTools([recordTool as AgentTool]);
				const nudges: string[] = [];
				t.agent.subscribe(event => {
					if (event.type === "message_end" && event.message.role === "custom")
						nudges.push(event.message.customType);
				});
				await t.session.prompt("investigate the child task");
				expect(t.requests.map(request => request.model)).toEqual(
					optIn ? [t.source.id, t.source.id, t.source.id] : [t.source.id, t.source.id],
				);
				expect(nudges).toEqual(optIn ? ["prewalk-plan", "prewalk-continue"] : ["prewalk-plan"]);
			},
		);
	}

	it("continuation off clears pending work so turning it on alone cannot revive an old nudge", async () => {
		const t = cycleHarness([
			toolCall("first", "record"),
			{ content: ["finished with continuation off"] },
			{ content: ["finished after re-enabling"] },
			toolCall("new-work", "record"),
			{ content: ["new answer"] },
			{ content: ["new work finished"] },
		]);
		t.agent.setTools([recordTool as AgentTool]);
		const nudges: string[] = [];
		t.agent.subscribe(event => {
			if (event.type === "message_end" && event.message.role === "custom") nudges.push(event.message.customType);
		});
		t.agent.setBeforeModelCall(() => (t.requests.length >= 1 ? { stop: true } : undefined));
		await t.session.prompt("investigate");
		t.agent.setBeforeModelCall(undefined);
		t.settings.set("prewalk.continueNudge", false);
		await t.session.prompt("finish", { synthetic: true });
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id]);
		t.settings.set("prewalk.continueNudge", true);
		await t.session.prompt("finish again", { synthetic: true });
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.source.id]);
		expect(nudges).toEqual(["prewalk-plan"]);
		await t.session.prompt("investigate something new", { synthetic: true });
		expect(t.requests.map(request => request.model)).toEqual(Array(6).fill(t.source.id));
		expect(nudges).toEqual(["prewalk-plan", "prewalk-continue"]);
	});

	it("hot revival of pending continuation respects the child continuation setting", async () => {
		const baseSettings = Settings.isolated({ "compaction.enabled": false, "task.prewalkContinueNudge": true });
		const t = cycleHarness([toolCall("read", "record")], { baseSettings });
		t.agent.setTools([recordTool as AgentTool]);
		t.agent.setBeforeModelCall(() => (t.requests.length >= 1 ? { stop: true } : undefined));
		await t.session.prompt("investigate child task");
		const snapshot = t.session.getPrewalkSnapshot();
		const messages = structuredClone(t.agent.state.messages);
		await t.session.dispose();
		const revived = cycleHarness([{ content: ["revived child finished"] }], {
			baseSettings: Settings.isolated({ "compaction.enabled": false }),
			snapshot,
		});
		revived.agent.replaceMessages(messages);
		const nudges: string[] = [];
		revived.agent.subscribe(event => {
			if (event.type === "message_end" && event.message.role === "custom") nudges.push(event.message.customType);
		});
		await revived.session.prompt("finish child task", { synthetic: true });
		expect(revived.requests.map(request => request.model)).toEqual([revived.source.id]);
		expect(nudges).not.toContain("prewalk-continue");
	});
	for (const repeat of [false, true]) {
		it(
			(repeat ? "repeats" : "does not repeat") + " the plan nudge for injected user input while still walking",
			async () => {
				const t = cycleHarness([
					toolCall("read-1", "record"),
					toolCall("read-2", "record"),
					toolCall("read-3", "record"),
					toolCall("write", "write"),
					{ content: ["done"] },
					toolCall("fresh-read", "record"),
					toolCall("fresh-write", "write"),
					{ content: ["done"] },
				]);
				if (repeat) t.settings.set("prewalk.repeatPlanNudge", true);
				t.session.setInterruptMode("wait");
				let reads = 0;
				const record: AgentTool<typeof recordToolSchema, undefined> = {
					...recordTool,
					async execute() {
						if (++reads === 2) await t.session.sendUserMessage("additional direction", { deliverAs: "steer" });
						return { content: [{ type: "text", text: "read" }], details: undefined };
					},
				};
				const nudges: string[] = [];
				t.agent.subscribe(event => {
					if (
						event.type === "message_end" &&
						event.message.role === "custom" &&
						event.message.customType === "prewalk-plan"
					)
						nudges.push(event.message.customType);
				});
				t.agent.setTools([record as AgentTool, writeTool as AgentTool]);
				await t.session.prompt("first task");
				expect(nudges).toHaveLength(repeat ? 2 : 1);
				expect(t.requests.map(request => request.model)).toEqual([
					t.source.id,
					t.source.id,
					t.source.id,
					t.source.id,
					t.target.id,
				]);
				await t.session.prompt("new task after handoff");
				expect(nudges).toHaveLength(repeat ? 3 : 2);
				expect(t.requests.slice(-3).map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
			},
		);
	}

	for (const restart of [false, true]) {
		it(`subagent parent input ${restart ? "restarts by opt-in" : "does not restart by default"}, including hot revival`, async () => {
			const baseSettings = Settings.isolated({
				"compaction.enabled": false,
				"prewalk.planNudge": false,
				"prewalk.repeatPlanNudge": true,
				"prewalk.afterEveryUserMessage": true,
				"task.prewalkAfterEveryUserMessage": restart,
			});
			const t = cycleHarness(
				[
					toolCall("read-1", "record"),
					toolCall("read-2", "record"),
					toolCall("write", "write"),
					{ content: ["done"] },
					{ content: ["follow-up"] },
				],
				{ baseSettings },
			);
			const plans: string[] = [];
			t.agent.subscribe(event => {
				if (
					event.type === "message_end" &&
					event.message.role === "custom" &&
					event.message.customType === "prewalk-plan"
				)
					plans.push(event.message.customType);
			});
			t.session.setInterruptMode("wait");
			let reads = 0;
			t.agent.setTools([
				{
					...recordTool,
					async execute() {
						if (++reads === 2)
							await t.session.sendUserMessage("direction during prewalk", { deliverAs: "steer" });
						return { content: [{ type: "text", text: "read" }], details: undefined };
					},
				} as AgentTool,
				writeTool as AgentTool,
			]);
			await t.session.prompt("initial child task");
			expect(plans).toHaveLength(1);
			await t.session.sendUserMessage("later parent direction");
			expect(t.requests.at(-1)?.model).toBe(restart ? t.source.id : t.target.id);
			const snapshot = t.session.getPrewalkSnapshot();
			await t.session.dispose();
			const revived = cycleHarness([{ content: ["revived follow-up"] }], { baseSettings, snapshot });
			await revived.session.setModelTemporary(restart ? t.source : t.target, restart ? Effort.High : Effort.Low, {
				ephemeral: true,
			});
			await revived.session.prompt("parent input after revival");
			expect(revived.requests[0]?.model).toBe(restart ? t.source.id : t.target.id);
		});
	}
	it("repeats a child plan without restarting its action budget or enabling later parent-input rearm", async () => {
		const baseSettings = Settings.isolated({
			"compaction.enabled": false,
			"prewalk.planNudge": false,
			"prewalk.afterEveryUserMessage": true,
			"task.prewalkRepeatPlanNudge": true,
			"task.prewalkMaxMessages": 8,
		});
		const t = cycleHarness(
			[
				toolCall("read-1", "record"),
				toolCall("read-2", "record"),
				toolCall("read-3", "record"),
				toolCall("read-4", "record"),
				{ content: ["done"] },
				{ content: ["parent follow-up"] },
			],
			{ baseSettings, maximum: 8 },
		);
		let plans = 0;
		t.agent.subscribe(event => {
			if (
				event.type === "message_end" &&
				event.message.role === "custom" &&
				event.message.customType === "prewalk-plan"
			)
				plans++;
		});
		t.session.setInterruptMode("wait");
		let reads = 0;
		t.agent.setTools([
			{
				...recordTool,
				async execute() {
					if (++reads === 2) await t.session.sendUserMessage("parent direction", { deliverAs: "steer" });
					return { content: [{ type: "text", text: "read" }], details: undefined };
				},
			} as AgentTool,
		]);
		await t.session.prompt("initial child task");

		expect(plans).toBe(2);
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.source.id,
			t.source.id,
			t.source.id,
			t.target.id,
		]);
		await t.session.sendUserMessage("later parent input");
		expect(t.requests.at(-1)?.model).toBe(t.target.id);
	});
	it("counts primary responses and executed tools, including failures, but excludes side messages", async () => {
		const t = cycleHarness(
			[
				toolCall("first", "record"),
				{
					content: [
						{ type: "toolCall", id: "second", name: "record", arguments: {} },
						{ type: "toolCall", id: "failed", name: "bash", arguments: {} },
					],
					stopReason: "toolUse",
				},
				{ content: ["done"] },
			],
			{ maximum: 5 },
		);
		t.agent.setTools([
			{
				...recordTool,
				async execute() {
					for (const customType of ["advisor", "injection"]) {
						t.agent.appendMessage({
							role: "custom",
							customType,
							content: "side context",
							display: false,
							timestamp: 1,
						});
					}
					t.agent.appendMessage({
						role: "branchSummary",
						summary: "side summary",
						fromId: "earlier",
						timestamp: 1,
					});
					t.agent.appendMessage({
						role: "compactionSummary",
						summary: "compacted context",
						tokensBefore: 100,
						timestamp: 1,
					});
					return { content: [{ type: "text", text: "observed" }], details: undefined };
				},
			} as AgentTool,
			{
				...bashTool,
				async execute() {
					throw new Error("executed tool failed");
				},
			} as AgentTool,
		]);
		await t.session.prompt("investigate");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
	});

	it("discards implementation before the minimum without latching it or recounting a tool batch", async () => {
		const t = cycleHarness(
			[
				{
					content: [
						{ type: "toolCall", id: "early-todo", name: "todo", arguments: {} },
						{ type: "toolCall", id: "early-write", name: "write", arguments: {} },
					],
					stopReason: "toolUse",
				},
				toolCall("read-2", "record"),
				toolCall("read-3", "record"),
				toolCall("eligible-write", "write"),
				{ content: ["done"] },
			],
			{ minimum: 7 },
		);
		t.agent.setTools([todoTool as AgentTool, writeTool as AgentTool, recordTool as AgentTool]);
		await t.session.prompt("plan before implementing");
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.source.id,
			t.source.id,
			t.source.id,
			t.target.id,
		]);
	});

	it("lets the minimum win over the maximum, which bypasses both todo and implementation", async () => {
		const t = cycleHarness(
			[
				toolCall("read-1", "record"),
				toolCall("read-2", "record"),
				toolCall("read-3", "record"),
				{ content: ["done"] },
			],
			{ minimum: 6, maximum: 4 },
		);
		t.agent.setTools([todoTool as AgentTool, recordTool as AgentTool]);
		await t.session.prompt("investigate without editing");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.source.id, t.target.id]);
	});

	it("counts text-only and length-completed responses, not their synthetic tool results", async () => {
		const t = cycleHarness(
			[
				{ content: ["planning"] },
				{ ...toolCall("truncated", "record"), stopReason: "length" },
				{ content: ["done"] },
			],
			{ maximum: 2 },
		);
		t.agent.setTools([recordTool as AgentTool]);
		await t.session.prompt("investigate");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
	});

	it("counts distinct same-millisecond responses even when their text is identical", async () => {
		const t = cycleHarness(
			[{ content: ["planning"] }, { content: ["planning"] }, { content: ["done"] }, { content: ["finished"] }],
			{ maximum: 2 },
		);
		t.agent.transformAssistantMessage = message => {
			message.timestamp = 1;
		};
		await t.session.prompt("investigate");
		expect(t.requests.slice(0, 3).map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
	});

	it("keeps maximum accounting while Eval is running and hands off only after it settles", async () => {
		const gate = Promise.withResolvers<string>();
		const manager = new AsyncJobManager({ onJobComplete: () => {} });
		const t = cycleHarness([toolCall("busy-write", "write"), toolCall("settle", "record"), { content: ["done"] }], {
			maximum: 1,
			manager,
		});
		let jobId = "";
		t.agent.setTools([
			{
				...writeTool,
				async execute() {
					jobId = manager.register("eval", "live cell", () => gate.promise);
					return { content: [{ type: "text", text: "wrote" }], details: undefined };
				},
			} as AgentTool,
			{
				...recordTool,
				async execute() {
					gate.resolve("settled");
					await manager.getJob(jobId)?.promise;
					return { content: [{ type: "text", text: "settled" }], details: undefined };
				},
			} as AgentTool,
		]);
		try {
			await t.session.prompt("work during Eval");
			expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
		} finally {
			gate.resolve("settled");
			await manager.dispose();
		}
	});

	it("keeps unlimited handoffs on the existing action path while Eval is running", async () => {
		const gate = Promise.withResolvers<string>();
		const manager = new AsyncJobManager({ onJobComplete: () => {} });
		const t = cycleHarness([toolCall("write", "write"), { content: ["done"] }], { manager });
		manager.register("eval", "live cell", () => gate.promise);
		try {
			await t.session.prompt("implement");
			expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id]);
		} finally {
			gate.resolve("settled");
			await manager.dispose();
		}
	});

	it("counts nested completions once and fences old cycles without switching before a safe boundary", async () => {
		const t = cycleHarness([toolCall("first", "record"), toolCall("second", "record"), { content: ["done"] }], {
			maximum: 4,
		});
		t.agent.setTools([recordTool as AgentTool]);
		const stale = t.session.beginPrewalkToolCall();
		expect(stale).toBeDefined();
		await t.session.resetSessionContext();
		expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
		const complete = t.session.beginPrewalkToolCall();
		expect(complete).toBeDefined();
		complete!();
		complete!();
		stale!();
		t.agent.appendMessage({
			role: "custom",
			customType: "advisor",
			content: "side advice",
			display: false,
			timestamp: 1,
		});
		expect(t.session.getPrewalkSnapshot().completedActions).toBe(1);
		expect(t.session.model?.id).toBe(t.source.id);
		await t.session.prompt("investigate");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id, t.target.id]);
	});

	it("does not recount responses but counts actual missing-result tool reexecution live and after hot revival", async () => {
		const t = cycleHarness([toolCall("first", "record"), toolCall("second", "record")], { maximum: 8 });
		const executions: string[] = [];
		const replayTool: AgentTool<typeof recordToolSchema, undefined> = {
			...recordTool,
			async execute(id) {
				executions.push(id);
				return { content: [{ type: "text", text: "observed" }], details: undefined };
			},
		};
		t.agent.setTools([replayTool]);
		t.agent.setBeforeModelCall(() => (t.requests.length >= 1 ? { stop: true } : undefined));
		await t.session.prompt("investigate");
		expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
		const firstTail = t.agent.state.messages.findLastIndex(message => message.role === "assistant");
		t.agent.replaceMessages(t.agent.state.messages.slice(0, firstTail + 1));
		t.agent.setBeforeModelCall(() => (t.requests.length >= 2 ? { stop: true } : undefined));
		await t.agent.continue();
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.source.id]);
		expect(t.session.model?.id).toBe(t.source.id);
		expect(executions).toEqual(["first", "first", "second"]);
		const snapshot = t.session.getPrewalkSnapshot();
		const secondTail = t.agent.state.messages.findLastIndex(message => message.role === "assistant");
		const replayMessages = structuredClone(t.agent.state.messages.slice(0, secondTail + 1));
		await t.session.dispose();
		const revived = cycleHarness([toolCall("third", "record"), { content: ["done"] }], { maximum: 8, snapshot });
		revived.agent.setTools([replayTool]);
		revived.agent.replaceMessages(replayMessages);
		await revived.agent.continue();
		expect(revived.requests.map(request => request.model)).toEqual([revived.source.id, revived.target.id]);
		expect(executions).toEqual(["first", "first", "second", "second", "third"]);
	});

	it("retains action accounting across passive history restoration without reexecuting completed tools", async () => {
		const t = cycleHarness([toolCall("first", "record")], { maximum: 4 });
		t.agent.setTools([recordTool as AgentTool]);
		t.agent.setBeforeModelCall(() => (t.requests.length >= 1 ? { stop: true } : undefined));
		await t.session.prompt("investigate");
		const snapshot = t.session.getPrewalkSnapshot();
		const messages = structuredClone(t.agent.state.messages);
		await t.session.dispose();
		const revived = cycleHarness([toolCall("second", "record"), { content: ["done"] }], { maximum: 4, snapshot });
		const executions: string[] = [];
		revived.agent.setTools([
			{
				...recordTool,
				async execute(id) {
					executions.push(id);
					return { content: [{ type: "text", text: "observed" }], details: undefined };
				},
			} as AgentTool,
		]);
		revived.agent.replaceMessages(messages);
		await revived.agent.continue();
		expect(executions).toEqual(["second"]);
		expect(revived.requests.map(request => request.model)).toEqual([revived.source.id, revived.target.id]);
	});

	it("does not spend the action limit on errored or aborted responses", async () => {
		const t = cycleHarness(
			[
				{ content: ["failed"], stopReason: "error", errorMessage: "non-retryable failure" },
				{ content: ["interrupted"], stopReason: "aborted" },
				toolCall("completed-1", "record"),
				toolCall("completed-2", "record"),
				{ content: ["done"] },
			],
			{ maximum: 4 },
		);
		t.settings.set("retry.enabled", false);
		t.settings.set("prewalk.afterEveryUserMessage", false);
		t.agent.setTools([recordTool as AgentTool]);
		await t.session.prompt("first attempt");
		await t.session.prompt("second attempt", { synthetic: true });
		await t.session.prompt("completed attempt", { synthetic: true });
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.source.id,
			t.source.id,
			t.source.id,
			t.target.id,
		]);
	});

	it("restarts the action limit for each newly delivered user cycle", async () => {
		const t = cycleHarness(
			[
				toolCall("first-1", "record"),
				toolCall("first-2", "record"),
				{ content: ["done"] },
				toolCall("second-1", "record"),
				toolCall("second-2", "record"),
				{ content: ["done"] },
			],
			{ maximum: 4 },
		);
		t.agent.setTools([recordTool as AgentTool]);
		await t.session.prompt("first task");
		await t.session.prompt("second task");
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.source.id,
			t.target.id,
			t.source.id,
			t.source.id,
			t.target.id,
		]);
	});

	it("repeats explicit cycles with exact source effort, ignoring synthetic delivery and retained history", async () => {
		const t = cycleHarness([
			toolCall("a", "write"),
			{ content: ["done"] },
			{ content: ["notice"] },
			toolCall("b", "write"),
			{ content: ["done"] },
		]);
		await t.session.prompt("first task");
		expect(t.session.getPrewalkStatus()).toBe("standing");
		await t.session.prompt("internal continuation", { synthetic: true });
		await t.session.prompt("second task");
		expect(t.requests).toEqual([
			{ model: t.source.id, effort: Effort.High },
			{ model: t.target.id, effort: Effort.Low },
			{ model: t.target.id, effort: Effort.Low },
			{ model: t.source.id, effort: Effort.High },
			{ model: t.target.id, effort: Effort.Low },
		]);
		expect(t.session.getPrewalkState()).toBeUndefined();
	});

	it("honors hard automatic disable while allowing explicit one-shot arm", async () => {
		const t = cycleHarness([toolCall("a", "write"), { content: ["done"] }, { content: ["next"] }], {
			disabled: true,
		});
		t.settings.set("prewalk.enabled", true);
		expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
		await t.session.prompt("manual task");
		await t.session.prompt("next task");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id, t.target.id]);
	});

	it("live disable clears active status without switching, and reenable restores retained source on future input", async () => {
		const t = cycleHarness([
			toolCall("a", "write"),
			{ content: ["done"] },
			{ content: ["disabled"] },
			toolCall("b", "write"),
			{ content: ["done"] },
		]);
		await t.session.prompt("first");
		t.session.setPrewalkEnabled(false);
		expect(t.session.getPrewalkStatus()).toBeUndefined();
		expect(t.session.model?.id).toBe(t.target.id);
		await t.session.prompt("disabled turn");
		t.session.setPrewalkEnabled(true);
		expect(t.session.model?.id).toBe(t.target.id);
		await t.session.prompt("reenabled turn");
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.target.id,
			t.target.id,
			t.source.id,
			t.target.id,
		]);
	});

	it("hides standing status on a different model and clears arm status on disable", async () => {
		const t = cycleHarness([toolCall("a", "write"), { content: ["done"] }]);
		await t.session.prompt("first");
		await t.session.setModelTemporary(t.source, Effort.High, { ephemeral: true });
		expect(t.session.getPrewalkStatus()).toBeUndefined();
		expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
		expect(t.session.getPrewalkStatus()).toBe("walking");
		t.session.setPrewalkEnabled(false);
		expect(t.session.getPrewalkState()).toBeUndefined();
		expect(t.session.getPrewalkStatus()).toBeUndefined();
	});

	it("resume starts configured default planning only for new input, never replayed messages", async () => {
		const t = cycleHarness([toolCall("a", "write"), { content: ["done"] }], { explicit: false });
		t.settings.set("prewalk.enabled", true);
		t.agent.replaceMessages([{ role: "user", content: [{ type: "text", text: "historical task" }], timestamp: 1 }]);
		expect(t.session.model?.id).toBe(t.target.id);
		expect(t.session.getPrewalkState()).toBeUndefined();
		await t.session.prompt("new task");
		expect(t.requests).toEqual([
			{ model: t.source.id, effort: Effort.High },
			{ model: t.target.id, effort: Effort.Low },
		]);
	});

	it("rearms queued follow-up and user custom input but not hidden user companions", async () => {
		const t = cycleHarness([
			toolCall("a", "write"),
			{ content: ["done"] },
			toolCall("b", "write"),
			{ content: ["done"] },
			{ content: ["companion"] },
			toolCall("c", "write"),
			{ content: ["done"] },
		]);
		await t.session.prompt("first");
		await t.session.followUp("queued task");
		await t.session.waitForIdle();
		await t.session.sendCustomMessage(
			{ customType: "image-attachment-description", content: "hidden context", display: false, attribution: "user" },
			{ triggerTurn: true },
		);
		await t.session.sendCustomMessage(
			{ customType: "user-submission", content: "new task", display: true, attribution: "user" },
			{ triggerTurn: true },
		);
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.target.id,
			t.source.id,
			t.target.id,
			t.target.id,
			t.source.id,
			t.target.id,
		]);
	});

	it("defers repeat during own background Eval and retries without another user message", async () => {
		const gate = Promise.withResolvers<string>();
		const manager = new AsyncJobManager({ onJobComplete: () => {} });
		const t = cycleHarness(
			[
				toolCall("a", "write"),
				{ content: ["done"] },
				toolCall("b", "record"),
				toolCall("c", "write"),
				{ content: ["done"] },
			],
			{ manager },
		);
		await t.session.prompt("first");
		const jobId = manager.register("eval", "live cell", () => gate.promise);
		const record: AgentTool<typeof recordToolSchema, undefined> = {
			...recordTool,
			async execute() {
				gate.resolve("settled");
				await manager.getJob(jobId)?.promise;
				return { content: [{ type: "text", text: "settled" }], details: undefined };
			},
		};
		t.agent.setTools([record as AgentTool, writeTool as AgentTool]);
		await t.session.prompt("input during Eval");
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.target.id,
			t.target.id,
			t.source.id,
			t.target.id,
		]);
		await manager.dispose();
	});

	it("successful context reset drops retained cycle and uses freshly configured defaults", async () => {
		const t = cycleHarness([
			toolCall("a", "write"),
			{ content: ["done"] },
			toolCall("b", "write"),
			{ content: ["done"] },
		]);
		await t.session.prompt("first");
		t.settings.set("prewalk.enabled", true);
		t.settings.setModelRole("default", t.source.provider + "/" + t.source.id + ":medium");
		const droppedCount = t.session.messages.length;
		expect(await t.session.resetSessionContext()).toEqual({ droppedCount });
		expect(t.session.getPrewalkStatus()).toBeUndefined();
		await t.session.prompt("new task");
		expect(t.requests.slice(-2)).toEqual([
			{ model: t.source.id, effort: Effort.Medium },
			{ model: t.target.id, effort: Effort.Low },
		]);
	});
	it("publishes standing only after the asynchronous handoff succeeds", async () => {
		const t = cycleHarness([toolCall("a", "write"), { content: ["done"] }]);
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const setModel = t.session.setModelTemporary.bind(t.session);
		vi.spyOn(t.session, "setModelTemporary").mockImplementation(async (model, effort, options) => {
			if (model.id === t.target.id) {
				started.resolve();
				await release.promise;
			}
			await setModel(model, effort, options);
		});
		const run = t.session.prompt("first");
		await started.promise;
		expect(t.session.getPrewalkStatus()).toBe("walking");
		expect(t.session.model?.id).toBe(t.source.id);
		release.resolve();
		await run;
		expect(t.session.getPrewalkStatus()).toBe("standing");
	});

	it("canceled session switch retains the cycle; committed switch and new session clear it", async () => {
		const t = cycleHarness(
			[toolCall("a", "write"), { content: ["done"] }, toolCall("b", "write"), { content: ["done"] }],
			{ persistent: true },
		);
		await t.session.prompt("first");
		const targetManager = SessionManager.create(tempDir.path(), tempDir.path());
		targetManager.appendMessage({ role: "user", content: "historical", timestamp: 1 });
		await targetManager.ensureOnDisk();
		await targetManager.flush();
		const targetFile = targetManager.getSessionFile();
		if (!targetFile) throw new Error("Expected persisted session");
		await targetManager.close();
		expect(await t.session.switchSession(targetFile, { onCwdChange: async () => false })).toBe(false);
		expect(t.session.getPrewalkStatus()).toBe("standing");
		await t.session.prompt("after canceled switch");
		expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id, t.source.id, t.target.id]);
		expect(await t.session.switchSession(targetFile, { preserveLocalCwd: true })).toBe(true);
		expect(t.session.getPrewalkStatus()).toBeUndefined();
		await t.session.setModelTemporary(t.source, Effort.High, { ephemeral: true });
		expect(t.session.armPrewalk(t.target, Effort.Low)).toBe(true);
		expect(await t.session.newSession()).toBe(true);
		expect(t.session.getPrewalkState()).toBeUndefined();
		expect(t.session.getPrewalkStatus()).toBeUndefined();
	});

	it("hot revival stays on execution model and retains the dormant source for the next user input", async () => {
		const t = cycleHarness([toolCall("a", "write"), { content: ["done"] }]);
		await t.session.prompt("first");
		const snapshot = t.session.getPrewalkSnapshot();
		const currentModel = t.session.model;
		const thinkingLevel = t.session.configuredThinkingLevel();
		await t.session.dispose();
		const mock = createMockModel({ responses: [toolCall("b", "write"), { content: ["done"] }] });
		const requests: { model: string; effort: unknown }[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model: currentModel, systemPrompt: ["Test"], tools: [writeTool as AgentTool], messages: [] },
			convertToLlm,
			streamFn: (model, context, options) => {
				requests.push({ model: model.id, effort: options?.reasoning });
				return mock.stream(model, context, options);
			},
		});
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: t.settings,
			modelRegistry,
			toolRegistry,
			thinkingLevel,
			prewalkSnapshot: snapshot,
		});
		expect(session.model?.id).toBe(t.target.id);
		expect(session.getPrewalkStatus()).toBe("standing");
		expect(session.getPrewalkState()).toBeUndefined();
		await session.prompt("new task");
		expect(requests).toEqual([
			{ model: t.source.id, effort: Effort.High },
			{ model: t.target.id, effort: Effort.Low },
		]);
	});
	for (const [sender, restart] of [
		["Parent", false],
		["Parent", true],
		["Peer", true],
	] as const) {
		it(`child prewalk treats ${sender} IRC input with restart=${restart} at its next request boundary`, async () => {
			const baseSettings = Settings.isolated({
				"compaction.enabled": false,
				"task.prewalkAfterEveryUserMessage": restart,
			});
			const t = cycleHarness(
				[
					toolCall("first-write", "write"),
					toolCall("read", "record"),
					toolCall("next-write", "write"),
					{ content: ["done"] },
				],
				{ baseSettings },
			);
			const id = `prewalk-child-${crypto.randomUUID()}`;
			const registry = AgentRegistry.global();
			const ref = registry.register({ id, displayName: id, kind: "sub", parentId: "Parent", session: t.session });
			t.session.setInterruptMode("wait");
			t.agent.setTools([
				{
					...recordTool,
					async execute() {
						await t.session.deliverIrcMessage({
							id: "direction",
							from: sender,
							to: id,
							body: "implement the next step",
							ts: Date.now(),
						});
						return { content: [{ type: "text", text: "read" }], details: undefined };
					},
				} as AgentTool,
				writeTool as AgentTool,
			]);
			try {
				await t.session.prompt("first task");
				expect(t.requests.map(request => request.model)).toEqual([
					t.source.id,
					t.target.id,
					sender === "Parent" && restart ? t.source.id : t.target.id,
					t.target.id,
				]);
			} finally {
				registry.unregister(id, ref);
			}
		});
	}
	for (const deliverAs of ["steer", "aside"] as const) {
		it("rearms " + deliverAs + " user input at the next model request boundary", async () => {
			const t = cycleHarness([
				toolCall("a", "write"),
				toolCall("b", "record"),
				toolCall("c", "write"),
				{ content: ["done"] },
			]);
			t.session.setInterruptMode("wait");
			const record: AgentTool<typeof recordToolSchema, undefined> = {
				...recordTool,
				async execute() {
					await t.session.sendUserMessage("new task during tools", { deliverAs });
					return { content: [{ type: "text", text: "read" }], details: undefined };
				},
			};
			t.agent.setTools([record as AgentTool, writeTool as AgentTool]);
			await t.session.prompt("first");
			expect(t.requests.map(request => request.model)).toEqual([t.source.id, t.target.id, t.source.id, t.target.id]);
		});
	}
	it("retains primary admission after skill expansion, magic companions and image normalization", async () => {
		const t = cycleHarness([
			toolCall("a", "write"),
			{ content: ["done"] },
			toolCall("b", "write"),
			{ content: ["done"] },
			toolCall("c", "write"),
			{ content: ["done"] },
		]);
		t.settings.set("magicKeywords.ultrathink", true);
		t.settings.set("images.autoResize", false);
		await t.session.prompt("first ultrathink task");
		await t.session.promptCustomMessage({
			customType: "skill",
			content: "expanded user task",
			display: true,
			attribution: "user",
			details: { name: "workflow", args: "ultrathink task" },
		});
		await t.session.sendCustomMessage(
			{
				customType: "user-image",
				content: [
					{ type: "text", text: "task with image" },
					{
						type: "image",
						data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWZkAAAAASUVORK5CYII=",
						mimeType: "image/png",
					},
				],
				display: true,
				attribution: "user",
			},
			{ triggerTurn: true },
		);
		expect(t.requests.map(request => request.model)).toEqual([
			t.source.id,
			t.target.id,
			t.source.id,
			t.target.id,
			t.source.id,
			t.target.id,
		]);
	});
});
