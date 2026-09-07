import { describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, ImageContent, Model } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import {
	buildPreservedUserMessageClassifierInput,
	buildPreservedUserMessageClassifierInputFromLookup,
	buildPreservedUserMessageClassifierRequest,
	classifyPreservedUserMessage,
	iteratePreservedUserMessageClassifierInputs,
	parsePreservedUserMessageCategoryMask,
	preservedUserMessageClassifierInputsEqual,
} from "../src/session/preserve-user-messages-classifier";
import type { SessionEntry } from "../src/session/session-entries";

const image: ImageContent = { type: "image", mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aMwoAAAAASUVORK5CYII=" };
const model: Model = getBundledModel("openai", "gpt-4o")!;
function user(content: string | ImageContent[]): AgentMessage {
	return { role: "user", content, timestamp: 1 };
}
function assistant(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant", content, api: "openai-completions", provider: "openai", model: "gpt-4o", stopReason: "stop", timestamp: 1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	};
}
function branch(messages: AgentMessage[]): SessionEntry[] {
	return messages.map((message, i) => ({ type: "message", id: String(i), parentId: i === 0 ? null : String(i - 1), timestamp: "2026-01-01", message }));
}

describe("classifier source projection", () => {
	it("keeps actual empty and image-only previous users in every builder", () => {
		const entries = branch([user("older text"), user(""), user([image]), user("target")]);
		const byId = new Map(entries.map(entry => [entry.id, entry]));
		for (const { entryId, input } of iteratePreservedUserMessageClassifierInputs(entries)) {
			expect(buildPreservedUserMessageClassifierInput(entries, entryId)).toEqual(input);
			expect(buildPreservedUserMessageClassifierInputFromLookup(entryId, id => byId.get(id))).toEqual(input);
		}
		expect(buildPreservedUserMessageClassifierInput(entries, "2")!.previousUserMessage).toEqual([{ type: "text", text: "" }]);
		expect(buildPreservedUserMessageClassifierInput(entries, "3")!.previousUserMessage).toEqual([image]);
	});

	it("counts two assistant messages, retains all calls, excludes results/reasoning, snapshots arguments", () => {
		const args = { nested: { value: "original" } };
		const entries = branch([
			user("previous"), assistant([{ type: "text", text: "older" }]),
			assistant([{ type: "thinking", thinking: "private" }]),
			assistant([{ type: "toolCall", id: "a", name: "read", arguments: args }, { type: "toolCall", id: "b", name: "bash", arguments: { command: "pwd" } }]),
			{ role: "toolResult", toolCallId: "a", toolName: "read", content: [{ type: "text", text: "excluded output" }], isError: false, timestamp: 1 },
			assistant([{ type: "text", text: "recent" }]), user("target"),
		]);
		const input = buildPreservedUserMessageClassifierInput(entries, "6")!;
		expect(input.previousAssistantMessages).toEqual([
			[{ type: "toolCall", name: "read", arguments: args }, { type: "toolCall", name: "bash", arguments: { command: "pwd" } }],
			[{ type: "text", text: "recent" }],
		]);
		args.nested.value = "changed";
		expect(preservedUserMessageClassifierInputsEqual(input, buildPreservedUserMessageClassifierInput(entries, "6"))).toBe(false);
		expect(input.previousAssistantMessages[0]![0]).toEqual({ type: "toolCall", name: "read", arguments: { nested: { value: "original" } } });
	});

	it("resets relative context at clear and ignores synthetic users", () => {
		const entries = branch([user("old"), user("ignored"), { ...user("continue"), synthetic: true } as AgentMessage, user("new")]);
		entries[1] = { type: "reset_boundary", id: "1", parentId: "0", timestamp: "2026-01-01" };
		const input = buildPreservedUserMessageClassifierInput(entries, "3")!;
		expect(input.previousUserMessage).toBeNull();
		expect(input.previousAssistantMessages).toEqual([]);
		expect([...iteratePreservedUserMessageClassifierInputs(entries)].map(item => item.entryId)).toEqual(["0", "3"]);
	});
});

describe("classifier request boundary", () => {
	it("sends complete current text beyond title limits and original images", () => {
		const text = "complete current message ".repeat(1000);
		const entries = branch([{ role: "user", content: [{ type: "text", text }, image], timestamp: 1 }]);
		const request = buildPreservedUserMessageClassifierRequest(buildPreservedUserMessageClassifierInput(entries, "0")!, model);
		expect(request.messages[0]!.content).toContainEqual({ type: "text", text });
		expect(request.messages[0]!.content).toContainEqual(image);
	});

	it("bounds auxiliaries without truncating current and rejects oversized current", () => {
		const entries = branch([user("auxiliary ".repeat(20000)), user("the complete current")]);
		const constrained = { ...model, contextWindow: 2000, maxTokens: 128 };
		const request = buildPreservedUserMessageClassifierRequest(buildPreservedUserMessageClassifierInput(entries, "1")!, constrained);
		expect(request.messages[0]!.content).toContainEqual({ type: "text", text: "the complete current" });
		expect(JSON.stringify(request.messages)).not.toContain("auxiliary auxiliary");
		expect(() => buildPreservedUserMessageClassifierRequest(buildPreservedUserMessageClassifierInput(branch([user("target ".repeat(20000))]), "0")!, constrained)).toThrow("complete current message exceeds");
	});

	it("does not assign zero to image-only or unsupported targets", () => {
		const input = buildPreservedUserMessageClassifierInput(branch([user([image])]), "0")!;
		expect(() => buildPreservedUserMessageClassifierRequest(input, { ...model, input: ["text"] })).toThrow("does not support");
		expect(buildPreservedUserMessageClassifierRequest(input, model).messages[0]!.content).toContainEqual(image);
	});

	it("accepts exactly eleven ordered category bits, including valid all-false", () => {
		expect(parsePreservedUserMessageCategoryMask("<labels>10000000001</labels>")).toBe(1025);
		expect(parsePreservedUserMessageCategoryMask("<labels>00000000000</labels>")).toBe(0);
		expect(parsePreservedUserMessageCategoryMask("reason <labels>00000000000</labels>")).toBeUndefined();
		expect(parsePreservedUserMessageCategoryMask("<labels>0000000000</labels>")).toBeUndefined();
	});

	it("rejects interrupted output even when its text looks complete", async () => {
		const input = buildPreservedUserMessageClassifierInput(branch([user("target")]), "0")!;
		await expect(classifyPreservedUserMessage(input, {
			model,
			complete: async () => ({ ...assistant([{ type: "text", text: "<labels>00000000000</labels>" }]), stopReason: "length" }),
		})).rejects.toThrow("length");
	});
});
