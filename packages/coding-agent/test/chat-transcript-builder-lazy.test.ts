import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage } from "@oh-my-pi/pi-ai";
import { TUI } from "@oh-my-pi/pi-tui";
import { Settings, settings, resetSettingsForTest } from "../src/config/settings";
import { ChatTranscriptBuilder, type ChatTranscriptBuilderDeps } from "../src/modes/components/chat-transcript-builder";
import { appendOutlineEntries, outlineVisibility } from "../src/modes/components/transcript-outline";
import { initTheme } from "../src/modes/theme/theme";
import type { SessionMessageEntry } from "../src/session/session-entries";
import { VirtualTerminal } from "../../tui/test/virtual-terminal";

const timestamp = 1_700_000_000_000;
const usage = {
	input: 4242, output: 17, cacheRead: 0, cacheWrite: 0, totalTokens: 4259,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
function assistant(content: AssistantMessage["content"]): AgentMessage {
	return {
		role: "assistant", content, api: "anthropic-messages", provider: "anthropic", model: "test",
		stopReason: "stop", usage, timestamp, completedAt: timestamp + 1000,
	};
}
function call(name: string, id: string, args: Record<string, unknown> = {}): AssistantMessage["content"][number] {
	return { type: "toolCall", name, id, arguments: args };
}
function result(name: string, id: string, details?: unknown, isError = false): AgentMessage {
	return { role: "toolResult", toolName: name, toolCallId: id, content: [{ type: "text", text: `result ${id}` }], details, isError, timestamp };
}
function entries(messages: AgentMessage[]): SessionMessageEntry[] {
	return messages.map((message, index) => ({ type: "message", id: String(index), parentId: index ? String(index - 1) : null, timestamp: new Date(timestamp).toISOString(), message }));
}
const builders: ChatTranscriptBuilder[] = [];
function builder(deferComponents: boolean, extra: Partial<ChatTranscriptBuilderDeps> = {}): ChatTranscriptBuilder {
	const instance = new ChatTranscriptBuilder({
		ui: new TUI(new VirtualTerminal(100, 24)), cwd: process.cwd(), requestRender: () => {}, deferComponents, ...extra,
	});
	builders.push(instance);
	return instance;
}
function rows(instance: ChatTranscriptBuilder, width = 90): readonly string[] {
	return instance.container.children.flatMap(child => child.render(width).map(row => Bun.stripANSI(row)));
}
beforeEach(async () => {
	await Settings.init({ inMemory: true, cwd: process.cwd() });
	settings.set("display.showTokenUsage", true);
	settings.set("display.showTurnTime", true);
	settings.set("display.cacheMissMarker", true);
	await initTheme();
});
afterEach(() => {
	for (const instance of builders.splice(0)) instance.dispose();
	resetSettingsForTest();
});

describe("source-backed lazy transcript replay", () => {
	it("preserves reaction, read grouping, usage and displaced hub/todo results across eviction", () => {
		const source = entries([
			{ role: "user", content: "Inspect this", timestamp },
			{ role: "custom", customType: "notice", display: false, content: "hidden", timestamp },
			assistant([{ type: "text", text: "\u{1f44d} Starting" }, call("read", "r1", { path: "a.ts:1-2" })]),
			result("read", "r1"),
			assistant([call("read", "r2", { path: "a.ts:3-4" })]),
			result("read", "r2"),
			{ role: "custom", customType: "notice", display: false, content: "hidden", timestamp },
			assistant([call("read", "r3", { path: "b.ts" }), { type: "text", text: "After read" }]),
			result("read", "r3"),
			assistant([call("hub", "h1", { op: "wait" })]),
			result("hub", "h1", { jobs: [{ id: "j1", status: "running" }] }),
			assistant([call("hub", "h2", { op: "wait" })]),
			result("hub", "h2", { jobs: [{ id: "j1", status: "completed" }] }),
			assistant([call("todo", "t1", { op: "write" })]), result("todo", "t1", { phases: [] }),
			assistant([call("todo", "t2", { op: "write" })]), result("todo", "t2", { phases: [] }, true),
			assistant([{ type: "text", text: "Continue" }, call("todo", "t3", { op: "write" })]), result("todo", "t3", { phases: [] }),
			{ role: "user", content: "Next turn", timestamp: timestamp + 2000 },
			assistant([{ type: "text", text: "Done" }]),
		]);
		const eager = builder(false);
		const lazy = builder(true);
		for (const entry of source) { eager.append([entry]); lazy.append([entry]); }
		const expected = rows(eager);
		expect(rows(lazy)).toEqual(expected);
		lazy.releaseOutside(new Set());
		// Rematerialize the reply before its reaction target: source linkage cannot depend on paint order.
		lazy.container.children[1]!.render(90);
		expect(rows(lazy)).toEqual(expected);
		eager.setExpanded(true);
		lazy.setExpanded(true);
		expect(rows(lazy, 55)).toEqual(rows(eager, 55));
		eager.container.setToolActivityVisible(false);
		lazy.container.setToolActivityVisible(false);
		lazy.releaseOutside(new Set());
		expect(rows(lazy, 55)).toEqual(rows(eager, 55));
		eager.container.setToolActivityVisible(true);
		lazy.container.setToolActivityVisible(true);
		expect(rows(lazy, 55)).toEqual(rows(eager, 55));
	});

	it("keeps rewind source identities stable when prior hub and todo panels disappear", () => {
		settings.set("display.showTokenUsage", false);
		const lazy = builder(true);
		const targets = appendOutlineEntries(lazy, entries([
			{ role: "user", content: "Start", timestamp },
			assistant([call("hub", "h1")]), result("hub", "h1", { jobs: [{ status: "running" }] }),
			assistant([call("hub", "h2")]), result("hub", "h2", { jobs: [{ status: "completed" }] }),
			assistant([call("todo", "t1")]), result("todo", "t1", { phases: [] }),
			assistant([call("todo", "t2")]), result("todo", "t2", { phases: [] }),
			assistant([{ type: "text", text: "Finished" }]),
		]));
		const visible = outlineVisibility(lazy.container.children.map(child => child.render(90)), targets);
		expect(targets.filter((_, index) => visible[index]).map(target => [target.turnId, target.entryId])).toEqual([
			["0", "0"], ["3", "4"], ["7", "8"], ["9", "9"],
		]);
	});

	it("constructs extension renderers only on demand and disposes evicted resources", () => {
		let opened = 0;
		let closed = 0;
		const lazy = builder(true, {
			getMessageRenderer: () => message => {
				opened++;
				return { render: () => [String(message.content)], dispose: () => { closed++; } };
			},
		});
		lazy.rebuild(entries(Array.from({ length: 300 }, (_, index) => ({
			role: "custom" as const, customType: "extension", display: true, content: `source ${index}`, timestamp,
		}))));
		expect(opened).toBe(0);
		const last = lazy.container.children.at(-1)!;
		expect(last.render(80)).toEqual(["source 299"]);
		expect(opened).toBe(1);
		lazy.releaseOutside(new Set([last]));
		expect(closed).toBe(0);
		lazy.releaseOutside(new Set());
		expect(closed).toBe(1);
		expect(last.render(80)).toEqual(["source 299"]);
		expect(opened).toBe(2);
	});
});
