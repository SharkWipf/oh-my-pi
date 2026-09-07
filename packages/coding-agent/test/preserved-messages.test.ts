import { describe, expect, it } from "bun:test";
import { Tokenizer } from "@oh-my-pi/pi-agent-core/tokenizer";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import {
	compilePreservedUserMessageRegexRules,
	DEFAULT_PRESERVATION_CATEGORY_ACTIONS,
	MESSAGE_OVERRIDE_CUSTOM_TYPE,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	type PreservationPolicySettings,
} from "../src/session/preserved-message-settings";
import {
	evaluatePreservationPolicy,
	prechargeNonUsers,
	preservationCandidate,
	PreservedMessageQuery,
	readPreservedUserMessageClassificationMasks,
} from "../src/session/preserved-messages";
import type { SessionEntry, SessionMessageEntry } from "../src/session/session-entries";

const tokenizer = new Tokenizer();
function policy(overrides: Partial<PreservationPolicySettings> = {}): PreservationPolicySettings {
	return { enabled: true, first: { mode: "off" }, recent: { mode: "off" }, hardRecent: { mode: "off" }, alwaysCap: "keep-last", prune: "no", maxTokens: 2000, heuristics: false, regexRules: [], classifier: true, categoryActions: DEFAULT_PRESERVATION_CATEGORY_ACTIONS, ...overrides };
}
function user(id: string, content: string | Extract<AgentMessage, { role: "user" }>["content"], compactionOverride?: "keep" | "exclude"): SessionMessageEntry {
	return { type: "message", id, parentId: null, timestamp: "2026-01-01T00:00:00Z", message: { role: "user", content, timestamp: 0 }, ...(compactionOverride ? { compactionOverride } : {}) };
}
function control(id: string, customType: string, data: unknown): SessionEntry {
	return { type: "custom", id, parentId: null, timestamp: "2026-01-01T00:00:00Z", customType, data };
}
function link(entries: SessionEntry[], parentId: string | null = null): SessionEntry[] {
	for (const entry of entries) { entry.parentId = parentId; parentId = entry.id; }
	return entries;
}
async function query(entries: SessionEntry[], settings = policy()) {
	return (await PreservedMessageQuery.build(link(entries), settings, tokenizer, { isCurrent: () => true }))!;
}
function exchange(): SessionMessageEntry[] {
	return [
		{ type: "message", id: "a", parentId: null, timestamp: "", message: { role: "assistant", content: [{ type: "toolCall", id: "call1", name: "read", arguments: { path: "a" } }, { type: "toolCall", id: "call2", name: "read", arguments: { path: "b" } }], api: "anthropic-messages", provider: "anthropic", model: "test", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "toolUse", timestamp: 0 } },
		{ type: "message", id: "r1", parentId: null, timestamp: "", message: { role: "toolResult", toolCallId: "call1", toolName: "read", content: [{ type: "text", text: "first" }], isError: false, timestamp: 0 } },
		{ type: "message", id: "r2", parentId: null, timestamp: "", message: { role: "toolResult", toolCallId: "call2", toolName: "read", content: [{ type: "text", text: "second" }], isError: false, timestamp: 0 } },
	];
}

describe("source preservation policy", () => {
	it("keeps Auto neutral across stages and Keep dominant only within one stage", () => {
		const message = user("u", "fix this").message as Extract<AgentMessage, { role: "user" }>;
		const settings = policy({ regexRules: compilePreservedUserMessageRegexRules({ fix: { state: "keep", caseInsensitive: true }, this: { state: "exclude", caseInsensitive: true }, "fix this": { state: "auto", caseInsensitive: true, final: true } }) });
		expect(evaluatePreservationPolicy(message, settings, "auto", 1 << 5).resolved).toBe("exclude");
		expect(evaluatePreservationPolicy(message, settings, "auto", (1 << 5) | 1).resolved).toBe("keep");
		expect(evaluatePreservationPolicy(message, settings, "exclude", 1).resolved).toBe("exclude");
		expect(evaluatePreservationPolicy(message, { ...settings, classifier: false }).resolved).toBe("keep");
		expect(evaluatePreservationPolicy({ role: "user", content: "thanks", timestamp: 0 }, policy({ heuristics: true })).resolved).toBe("exclude");
		expect(evaluatePreservationPolicy({ role: "user", content: "new instruction", timestamp: 0 }, policy({ heuristics: true })).resolved).toBe("auto");
	});

	it("counts overlap in each independent window without refilling or merging equal text identities", async () => {
		const q = await query([user("u1", "same", "keep"), user("u2", "same", "keep"), user("u3", "same", "keep")], policy({ first: { mode: "messages", value: 2 }, recent: { mode: "messages", value: 2 }, alwaysCap: "keep-first" }));
		const s = q.select({ maximumContext: 1000 });
		expect([...s.P]).toEqual(["u1", "u2", "u3"]);
		expect(s.quota.first.count).toBe(2);
		expect(s.quota.recent.count).toBe(2);
		expect(s.quota.P.count).toBe(3);
		expect(s.reasons("u3")).toEqual({ first: false, recent: true, hardRecent: false, always: false, capDenied: true });
		expect(s.positions("u2")).toEqual({ first: 2, recent: 2 });
	});

	it("temporarily protects Never raw through Exclude and never persists the bypass", async () => {
		const old = user("old", "x".repeat(100), "exclude");
		const q = await query([old], policy({ hardRecent: { mode: "messages", value: 1 }, first: { mode: "all" }, prune: "exclude", maxTokens: 1 }));
		const s = q.select({ maximumContext: 1000 });
		expect([...s.H]).toEqual(["old"]);
		expect(s.candidate("old")!.message).toBe(old.message);
		expect(q.inspect("old")!.resolved).toBe("exclude");
		expect(q.appendEntries(link([user("new", "x")], "old"))).toBe(true);
		expect([...q.select({ maximumContext: 1000 }).P]).toEqual(["new"]);
		expect(q.getManualGroup("old")!.members[0]!.state).toBe("exclude");
	});

	it("uses raw quota for manual Always but candidate quota for inferred Keep", async () => {
		const entries = [user("manual", "a".repeat(160), "keep"), user("inferred", "b".repeat(160)), control("tags", USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 1, c: ["inferred", 1] })];
		const q = await query(entries, policy({ recent: { mode: "tokens", value: 10 }, prune: "head-only", maxTokens: 10 }));
		const s = q.select({ maximumContext: 1000 });
		expect([...s.P]).toEqual(["inferred"]);
		expect(s.candidate("inferred")!.truncated).toBe(true);
		expect(q.inspectCandidate("manual")!.quotaTokens).toBe(40);
		expect(s.reasons("manual").capDenied).toBe(true);
		expect([...q.select({ maximumContext: 1000, enabled: false, recent: { mode: "off" } }).P]).toEqual(["manual"]);
	});

	it("uses finite-zero arithmetic and MAX context percentages, not an Off shortcut", async () => {
		const q = await query([user("zero1", ""), user("positive", "abcd"), user("zero2", "")], policy({ first: { mode: "tokens", value: 0 }, recent: { mode: "tokens", value: 0 } }));
		expect([...q.select({ maximumContext: 100 }).P]).toEqual(["zero1", "zero2"]);
		expect([...q.select({ maximumContext: 100, first: { mode: "off" }, recent: { mode: "off" } }).P]).toEqual([]);
		expect([...q.select({ maximumContext: 100, first: { mode: "context-percent", value: 1 }, recent: { mode: "off" } }).P]).toEqual(["zero1", "positive", "zero2"]);
	});

	it("admits complete tool closures at distinct source count and precharges them exactly once", async () => {
		const toolEntries = exchange();
		const q = await query([...toolEntries, control("keep", MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: ["r1", "r2"], state: "keep" })], policy({ enabled: false, recent: { mode: "messages", value: 2 } }));
		expect(q.select({ maximumContext: 1000 }).N.size).toBe(0);
		const s = q.select({ maximumContext: 1000, recent: { mode: "messages", value: 3 } });
		expect([...s.N]).toEqual(["a", "r1", "r2"]);
		expect(s.quota.N.count).toBe(3);
		expect(q.getManualGroup("r1")!.memberIds).toEqual(["a", "r1", "r2"]);
		const atoms = [...s.nonUserAtoms()];
		const price = tokenizer.countMessages(toolEntries.map(entry => entry.message));
		const charged = prechargeNonUsers([...atoms, ...atoms], price - 1, entry => tokenizer.countMessage(entry.message));
		expect(charged.tokens).toBe(price);
		expect(charged.residualBudget).toBe(0);
		expect([...charged.sourceIds]).toEqual(["a", "r1", "r2"]);
	});

	it("completes delayed tool atoms on append and preserves journal override revisions", async () => {
		const [a, r1, r2] = exchange();
		const q = await query([a!, r1!, control("keep", MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: ["a"], state: "keep" })]);
		expect(q.getManualGroup("a")).toBeUndefined();
		expect(q.appendEntries(link([r2!], "keep"))).toBe(true);
		expect(q.getManualGroup("r2")!.members[0]).toEqual({ sourceId: "a", state: "keep", revisionId: "keep" });
		expect(q.nonAutoCount()).toBe(1);
		expect([...q.select({ maximumContext: 1000 }).N]).toEqual(["a", "r1", "r2"]);
		expect(q.appendEntries(link([control("auto", MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: ["a", "r1", "r2"], state: "auto" })], "r2"))).toBe(true);
		expect(q.nonAutoCount()).toBe(0);
		expect(q.getManualGroup("a")!.members.every(member => member.revisionId === "auto")).toBe(true);
	});

	it("preserves original image blocks and source spans under every text pruning mode", () => {
		const image = { type: "image" as const, data: "original", mimeType: "image/png" };
		const entry = user("image", [{ type: "text", text: "abcdefgh".repeat(100) }, image, { type: "text", text: "qrstuvwx".repeat(100) }]);
		for (const prune of ["head-only", "middle-out", "tail-only"] as const) {
			const candidate = preservationCandidate(entry, { prune, maxTokens: 1250 }, tokenizer)!;
			expect(candidate.quotaTokens).toBeLessThanOrEqual(1250);
			expect(candidate.rawTokens).toBe(1600);
			expect((candidate.message as Extract<AgentMessage, { role: "user" }>).content).toContain(image);
			expect(candidate.spans).toContainEqual({ sourceId: "image", blockIndex: 1 });
			expect(JSON.stringify(candidate.message)).toContain("[truncated]");
		}
		const limited = preservationCandidate(entry, { prune: "middle-out", maxTokens: 10 }, tokenizer)!;
		expect(limited.limitation?.kind).toBe("immutable-content-exceeds-limit");
		expect(limited.quotaTokens).toBeGreaterThan(1200);
		expect((entry.message as Extract<AgentMessage, { role: "user" }>).content).toEqual([{ type: "text", text: "abcdefgh".repeat(100) }, image, { type: "text", text: "qrstuvwx".repeat(100) }]);
	});

	it("scopes metadata and sources to post-clear and abandons obsolete cooperative builds", async () => {
		const entries: SessionEntry[] = [user("old", "old", "keep"), { type: "reset_boundary", id: "reset", parentId: "old", timestamp: "" } as SessionEntry, user("new", "new"), control("tags", USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 1, c: ["old", 1, "new", 0] })];
		let current = true;
		const q = (await PreservedMessageQuery.build(link(entries), policy({ first: { mode: "all" } }), tokenizer, { isCurrent: () => current }))!;
		expect(q.resetId).toBe("reset");
		expect([...q.select({ maximumContext: 1000 }).P]).toEqual(["new"]);
		expect(q.classificationStatus("new")).toBe("valid");
		expect(await readPreservedUserMessageClassificationMasks(entries, { isCurrent: () => true })).toEqual(new Map([["new", 0]]));
		current = false;
		expect(() => q.select({ maximumContext: 1000 })).toThrow("no longer current");
		let owned = true;
		expect(await PreservedMessageQuery.build(entries, policy(), tokenizer, { isCurrent: () => owned, sliceMs: 0, yieldControl: async () => { owned = false; } })).toBeUndefined();
	});

	it("invalidates successful masks rather than inventing false labels after a source rewrite", async () => {
		const q = await query([user("u", "rule"), control("good", USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 1, c: ["u", 1] })]);
		expect([...q.select({ maximumContext: 1000 }).P]).toEqual(["u"]);
		q.invalidateClassifications(["u"]);
		expect(q.classificationStatus("u")).toBe("invalidated");
		expect(q.rowAt(0)!.categoryMask).toBeUndefined();
		expect([...q.select({ maximumContext: 1000 }).P]).toEqual([]);
		q.applyClassifications([{ id: "u", mask: 0 }]);
		expect(q.classificationStatus("u")).toBe("valid");
		expect(q.rowAt(0)!.categoryMask).toBe(0);
	});
});
