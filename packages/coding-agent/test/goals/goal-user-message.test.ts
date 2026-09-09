import { afterEach, describe, expect, it } from "bun:test";
import type { AgentTool } from "@oh-my-pi/pi-agent-core";
import type { UserMessage } from "@oh-my-pi/pi-ai";
import { createMockModel, registerMockApi, type MockHandler } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { InteractiveMode } from "@oh-my-pi/pi-coding-agent/modes/interactive-mode";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

registerMockApi();

function userEntries(manager: SessionManager) {
	return manager.getEntries().filter(entry => entry.type === "message").filter((entry): entry is typeof entry & { message: UserMessage } => entry.message.role === "user");
}

async function createHarness(handler: MockHandler = { content: ["Done"] }) {
	const dir = TempDir.createSync("goal-user-message-");
	resetSettingsForTest();
	await Settings.init({ inMemory: true, cwd: dir.path() });
	const auth = await AuthStorage.create(dir.join("auth.db"));
	auth.setRuntimeApiKey("mock", "test-key");
	const mock = createMockModel({ handler });
	const settings = Settings.isolated({
		"goal.injectAsUserMessage": true,
		"compaction.enabled": false,
		"async.enabled": false,
		"memory.backend": "off",
		"tools.xdev": false,
	});
	const manager = SessionManager.create(dir.path(), dir.path());
	const { session } = await createAgentSession({
		cwd: dir.path(),
		agentDir: dir.path(),
		sessionManager: manager,
		authStorage: auth,
		modelRegistry: new ModelRegistry(auth, dir.join("models.yml")),
		settings,
		model: mock.model,
		disableExtensionDiscovery: true,
		skills: [],
		contextFiles: [],
		workspaceTree: { rootPath: dir.path(), rendered: "", truncated: false, totalLines: 0, agentsMdFiles: [] },
		promptTemplates: [],
		slashCommands: [],
		enableMCP: false,
		enableLsp: false,
		toolNames: ["read"],
	});
	await session.setActiveToolsByName(["goal"]);
	const tool = session.agent.state.tools.find(tool => tool.name === "goal")!;
	return { session, manager, tool, settings, mock, dispose: async () => {
		await session.dispose();
		auth.close();
		dir.removeSync();
		resetSettingsForTest();
	} };
}

async function operation(tool: AgentTool, op: string, objective?: string) {
	return tool.execute(`goal-${op}`, { op, objective } as never);
}

async function settle(session: AgentSession) {
	await session.waitForIdle();
	await session.sessionManager.flush();
}

describe("goal objective ordinary delivery", () => {
	let harness: { dispose: () => Promise<void> } | undefined;
	afterEach(async () => { await harness?.dispose(); harness = undefined; });

	it("delivers an idle objective once, preserving its source ID across reload and lifecycle operations", async () => {
		const h = await createHarness();
		harness = h;
		await operation(h.tool, "create", "  Keep ordinary delivery  ");
		expect(h.session.getQueuedMessages().followUp).toEqual(["Keep ordinary delivery"]);
		expect(h.mock.calls).toHaveLength(0);
		// Queue submission is not a durable user source. A crash here must not replay the goal.
		await h.manager.ensureOnDisk();
		await h.manager.flush();
		const beforeDelivery = await SessionManager.open(h.manager.getSessionFile()!);
		expect(userEntries(beforeDelivery)).toEqual([]);
		await h.session.prompt("Begin");
		await settle(h.session);
		const users = userEntries(h.manager);
		expect(users.map(entry => entry.message.content)).toEqual([
			[{ type: "text", text: "Begin" }], [{ type: "text", text: "Keep ordinary delivery" }],
		]);
		expect(users[1]!.message).toMatchObject({ role: "user", attribution: "user", producer: { type: "tool", name: "goal", toolCallId: "goal-create" } });
		const reopened = await SessionManager.open(h.manager.getSessionFile()!);
		expect(userEntries(reopened)).toEqual(JSON.parse(JSON.stringify(users)));
		h.settings.override("goal.injectAsUserMessage", false);
		expect(await h.session.switchSession(h.manager.getSessionFile()!)).toBe(true);
		await operation(h.tool, "get");
		await operation(h.tool, "resume");
		await operation(h.tool, "complete");
		await operation(h.tool, "drop");
		expect(h.session.queuedMessageCount).toBe(0);
		expect(userEntries(h.manager).map(entry => entry.id)).toEqual(users.map(entry => entry.id));
	});

	it("drains a successful model goal create after the tool atom, not as an interrupt or duplicate", async () => {
		let call = 0;
		const h = await createHarness(() => ++call === 1
			? { content: [{ type: "toolCall", id: "model-goal", name: "goal", arguments: { op: "create", objective: "Streaming objective" } }] }
			: { content: ["Done"] });
		harness = h;
		await h.session.prompt("Start the work");
		await settle(h.session);
		const entries = h.manager.getEntries();
		const users = userEntries(h.manager);
		expect(users.map(entry => entry.message.content)).toEqual([
			[{ type: "text", text: "Start the work" }], [{ type: "text", text: "Streaming objective" }],
		]);
		const injectedIndex = entries.findIndex(entry => entry.id === users[1]!.id);
		const resultIndex = entries.findIndex(entry => entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolCallId === "model-goal");
		expect(resultIndex).toBeGreaterThan(-1);
		expect(injectedIndex).toBeGreaterThan(resultIndex);
		expect(h.session.queuedMessageCount).toBe(0);
		expect(users[1]!.message).toMatchObject({ producer: { type: "tool", name: "goal", toolCallId: "model-goal" } });
		const reopened = await SessionManager.open(h.manager.getSessionFile()!);
		expect(userEntries(reopened)).toEqual(JSON.parse(JSON.stringify(users)));
	});

	it("does not inject disabled creates, rejected creates, or duplicate direct /goal input", async () => {
		const h = await createHarness();
		harness = h;
		h.settings.override("goal.injectAsUserMessage", false);
		await operation(h.tool, "create", "Disabled objective");
		expect(h.session.getGoalModeState()?.goal.objective).toBe("Disabled objective");
		h.settings.override("goal.injectAsUserMessage", true);
		await expect(operation(h.tool, "create", "Rejected objective")).rejects.toThrow();
		await operation(h.tool, "drop");
		initTheme();
		const mode = new InteractiveMode(h.session, "test");
		try {
			await mode.handleGoalModeCommand("Direct objective");
			expect(h.session.getGoalModeState()?.goal.objective).toBe("Direct objective");
			expect(h.session.getQueuedMessages()).toEqual({ steering: [], followUp: [] });
			expect(userEntries(h.manager)).toEqual([]);
		} finally {
			mode.stop();
		}
	});
});
