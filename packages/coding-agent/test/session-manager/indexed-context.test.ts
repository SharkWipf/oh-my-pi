import { describe, expect, it } from "bun:test";
import { buildSessionContext } from "@oh-my-pi/pi-coding-agent/session/session-context";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { makeAssistantMessage } from "./helpers";

describe("indexed session context", () => {
	it("keeps pre-clear controls and branch-local pins without restoring cleared messages", () => {
		const session = SessionManager.inMemory();
		session.appendModelChange("anthropic/chosen");
		session.appendThinkingLevelChange("high");
		session.appendServiceTierChange({ anthropic: "priority" });
		session.appendCredentialPin("anthropic", "original");
		session.appendMessage({ role: "user", content: "cold history", timestamp: 1 });
		const boundary = session.appendResetBoundary();
		const first = session.appendMessage({ role: "user", content: "first branch", timestamp: 2 });
		session.branch(boundary);
		session.appendCredentialPin("anthropic", "sibling");
		const second = session.appendMessage({ role: "user", content: "second branch", timestamp: 3 });
		for (const [leaf, text, pin] of [
			[first, "first branch", "original"],
			[second, "second branch", "sibling"],
			[first, "first branch", "original"],
		]) {
			session.branch(leaf);
			const context = session.buildSessionContext();
			expect(context.messages).toEqual([{ role: "user", content: text, timestamp: leaf === first ? 2 : 3 }]);
			expect(context.models.default).toBe("anthropic/chosen");
			expect(context.thinkingLevel).toBe("high");
			expect(session.getCredentialPins().get("anthropic")?.hash).toBe(pin);
			expect(context).toEqual(buildSessionContext(session.getEntries(), leaf));
			context.models.default = "caller mutation";
			context.serviceTier!.anthropic = "default";
		}
		expect(session.buildSessionContext({ transcript: true }).messages[0]).toMatchObject({ content: "cold history" });
	});

	it("preserves replay-through tails and collapsed transcript order across remote compaction", () => {
		const session = SessionManager.inMemory();
		session.appendMessage({ role: "user", content: "cold", timestamp: 1 });
		const first = session.appendMessage({ role: "user", content: "provider-owned", timestamp: 2 });
		session.appendResetBoundary();
		session.appendMessage({ role: "user", content: "replay after provider boundary", timestamp: 3 });
		session.appendCompaction("summary", undefined, first, 100, {
			providerReplayThroughEntryId: first,
			preserveData: { openaiRemoteCompaction: { provider: "openai", replacementHistory: [{ type: "message", role: "user", content: "provider-owned" }] } },
		});
		session.appendMessage({ role: "user", content: "new", timestamp: 4 });
		for (const options of [undefined, { transcript: true, collapseCompactedHistory: true }, { transcript: true }]) {
			expect(session.buildSessionContext(options)).toEqual(buildSessionContext(session.getEntries(), session.getLeafId(), undefined, options));
		}
		expect(session.buildSessionContext().messages.map(message => message.role)).toEqual(["compactionSummary", "user", "user"]);
		session.appendResetBoundary();
		expect(session.buildSessionContext().messages).toEqual([]);
	});

	it("invalidates rewritten controls, credentials and spend while preserving the selected branch", async () => {
		const session = SessionManager.inMemory();
		const model = session.appendModelChange("anthropic/original");
		const pin = session.appendCredentialPin("anthropic", "original");
		const assistant = makeAssistantMessage();
		assistant.timestamp = 9_000_000_000_000;
		session.appendMessage(assistant);
		const boundary = session.appendResetBoundary();
		const selected = session.appendMessage({ role: "user", content: "selected", timestamp: 2 });
		session.branch(boundary);
		session.appendMessage({ role: "user", content: "other", timestamp: 3 });
		session.branch(selected);
		session.buildSessionContext();
		const returnedPins = session.getCredentialPins();
		returnedPins.get("anthropic")!.hash = "caller mutation";
		expect(session.getCredentialPins().get("anthropic")?.hash).toBe("original");
		const modelEntry = session.getEntry(model)!;
		if (modelEntry.type !== "model_change") throw new Error("Expected model change");
		modelEntry.model = "anthropic/rewritten";
		const pinEntry = session.getEntry(pin)!;
		if (pinEntry.type !== "credential_pin") throw new Error("Expected credential pin");
		pinEntry.hash = "rewritten";
		assistant.usage.input = 17;
		await session.rewriteEntries();
		expect(session.getLeafId()).toBe(selected);
		expect(session.buildSessionContext().models.default).toBe("anthropic/rewritten");
		expect(session.getCredentialPins().get("anthropic")).toEqual({ hash: "rewritten", lastUsedAt: assistant.timestamp });
		expect(session.getAssistantUsageStatistics().input).toBe(17);
		expect(session.getUsageStatistics().input).toBe(17);
	});

	it("excludes nested and background usage from footer spend across branch changes and rebuilds", async () => {
		const session = SessionManager.inMemory();
		const assistant = makeAssistantMessage();
		assistant.usage.cost.total = 0.25;
		const first = session.appendMessage(assistant);
		session.appendMessage({ role: "toolResult", toolCallId: "task", toolName: "task", content: [], details: { usage: { ...assistant.usage, cost: { ...assistant.usage.cost, total: 2 } } }, isError: false, timestamp: 2 });
		session.appendModelUsage({ purpose: "background", role: "smol", api: assistant.api, provider: assistant.provider, model: assistant.model, stopReason: "stop", usage: { ...assistant.usage, cost: { ...assistant.usage.cost, total: 3 } } }, { sessionId: session.getSessionId(), parentId: session.getLeafId() });
		session.branch(first);
		await session.rewriteEntries();
		expect(session.getAssistantUsageStatistics().cost).toBe(0.25);
		expect(session.getUsageStatistics().cost).toBe(5.25);
	});
});
