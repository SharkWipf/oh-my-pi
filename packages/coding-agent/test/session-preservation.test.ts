import { afterEach, describe, expect, it } from "bun:test";
import { rejects } from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { Tokenizer } from "@oh-my-pi/pi-agent-core/tokenizer";
import { TempDir } from "@oh-my-pi/pi-utils";
import {
	DEFAULT_PRESERVATION_CATEGORY_ACTIONS,
	MESSAGE_OVERRIDE_CUSTOM_TYPE,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	type PreservationPolicySettings,
} from "../src/session/preserved-message-settings";
import { PreservedMessageQuery } from "../src/session/preserved-messages";
import type { SessionEntry } from "../src/session/session-entries";
import { SessionManager } from "../src/session/session-manager";
import { ensurePreservedMessageStateOnDisk, SessionPreservation } from "../src/session/session-preservation";
import { FileSessionStorage, type WriteTextAtomicOptions } from "../src/session/session-storage";
import { toRestoredQueuedMessage } from "../src/session/queued-messages";

const policy: PreservationPolicySettings = {
	enabled: false,
	first: { mode: "off" }, recent: { mode: "off" }, hardRecent: { mode: "off" },
	alwaysCap: "uncapped", prune: "no", maxTokens: 2_000,
	heuristics: false, regexRules: [], classifier: true,
	categoryActions: DEFAULT_PRESERVATION_CATEGORY_ACTIONS,
};

class FaultStorage extends FileSessionStorage {
	failAtomic = false;
	failDrain = false;
	drainGate?: Promise<void>;
	atomicWriteGate?: { reached(): void; release: Promise<void> };
	override async writeTextAtomic(file: string, content: string, options?: WriteTextAtomicOptions): Promise<void> {
		if (this.failAtomic) {
			this.failAtomic = false;
			throw new Error("manual atomic publication failed");
		}
		if (this.atomicWriteGate) {
			this.atomicWriteGate.reached();
			await this.atomicWriteGate.release;
		}
		await super.writeTextAtomic(file, content, options);
	}
	override async appendTextAtomic(file: string, content: string, options?: WriteTextAtomicOptions): Promise<void> {
		if (this.failAtomic) {
			this.failAtomic = false;
			throw new Error("manual atomic publication failed");
		}
		await super.appendTextAtomic(file, content, options);
	}
	override async drain(): Promise<void> {
		if (this.failDrain) {
			this.failDrain = false;
			throw new Error("manual flush failed");
		}
		if (this.drainGate) await this.drainGate;
		await super.drain();
	}
}

const fixtures: { directory: TempDir; manager: SessionManager; storage: FaultStorage }[] = [];
afterEach(async () => {
	for (const fixture of fixtures.splice(0)) {
		fixture.storage.failAtomic = false;
		fixture.storage.failDrain = false;
		fixture.storage.drainGate = undefined;
		await fixture.manager.close();
		await fixture.directory.remove();
	}
});

async function fixture(seed: (manager: SessionManager) => void) {
	const directory = TempDir.createSync("@pi-manual-preservation-");
	const storage = new FaultStorage();
	const manager = SessionManager.create(directory.path(), path.join(directory.path(), "sessions"), storage);
	fixtures.push({ directory, manager, storage });
	seed(manager);
	let query: PreservedMessageQuery;
	let ownership: object = {};
	const changed: string[][] = [];
	const prepare = async () => {
		await ensurePreservedMessageStateOnDisk(manager);
		const built = await PreservedMessageQuery.build(manager.getBranch(), policy, new Tokenizer(), { isCurrent: () => true });
		if (!built) throw new Error("Query preparation unexpectedly cancelled");
		query = built;
		return query;
	};
	await prepare();
	let onAppend: ((entry: SessionEntry) => void) | undefined;
	manager.onEntryAppended = entry => {
		query.appendEntries([entry]);
		onAppend?.(entry);
	};
	const preservation = new SessionPreservation({
		sessionManager: manager,
		getQuery: () => query,
		preparePreservedMessages: prepare,
		ownership: () => ownership,
		onChanged: ids => changed.push([...ids]),
	});
	return {
		manager, storage, preservation, changed, prepare,
		query: () => query,
		transition: () => { ownership = {}; },
		onAppend: (callback: (entry: SessionEntry) => void) => { onAppend = callback; },
	};
}

function user(manager: SessionManager, content: string): string {
	return manager.appendMessage({ role: "user", content, timestamp: 1 });
}

function overrides(manager: SessionManager) {
	return manager.getEntries().filter(entry => entry.type === "custom" && entry.customType === MESSAGE_OVERRIDE_CUSTOM_TYPE);
}

async function reopenedQuery(manager: SessionManager): Promise<PreservedMessageQuery> {
	const reopened = await SessionManager.open(manager.getSessionFile()!);
	try {
		const query = await PreservedMessageQuery.build(reopened.getBranch(), policy, new Tokenizer(), { isCurrent: () => true });
		if (!query) throw new Error("Query preparation unexpectedly cancelled");
		return query;
	} finally {
		await reopened.close();
	}
}

describe("durable manual preservation actions", () => {
	it("cooperates during large captures, excludes suffixes, and rejects ownership loss mid-capture", async () => {
		const sources: string[] = [];
		let revision = "";
		const f = await fixture(manager => {
			for (let index = 0; index < 20_000; index++) sources.push(user(manager, `source ${index}`));
			revision = manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: sources, state: "keep" });
		});
		let suffix = "";
		setTimeout(() => {
			suffix = user(f.manager, "after capture boundary");
			f.manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [suffix], state: "keep" });
			f.manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [sources[0]], state: "exclude" });
		}, 0);
		const snapshot = await f.preservation.capturePreservedMessageOverrideReset();
		expect(suffix).not.toBe("");
		expect(snapshot.sourceCount).toBe(sources.length);
		expect(snapshot.groups.some(group => group.memberIds.includes(suffix))).toBe(false);
		expect(snapshot.groups[0]?.members[0]).toEqual({ sourceId: sources[0], state: "keep", revisionId: revision });
		expect(f.query().getManualGroup(sources[0]!)?.members[0]?.state).toBe("exclude");
		setTimeout(() => f.transition(), 0);
		await expect(f.preservation.capturePreservedMessageOverrideReset()).rejects.toThrow("changed");
	});

	it("does not expose a reset snapshot until its source journal is durably flushed", async () => {
		let source = "";
		const f = await fixture(manager => {
			source = user(manager, "snapshot publication");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [source], state: "keep" });
		});
		f.storage.failDrain = true;
		await expect(f.preservation.capturePreservedMessageOverrideReset()).rejects.toThrow("flush failed");
		const snapshot = await f.preservation.capturePreservedMessageOverrideReset();
		expect(snapshot.groups[0]?.memberIds).toEqual([source]);
		expect(f.changed).toEqual([]);
	});

	it("resets captured hidden sources, allows suffixes, and skips newer same-value journal revisions", async () => {
		let first = "", second = "", tag = "";
		const f = await fixture(manager => {
			first = user(manager, "same content");
			second = user(manager, "same content");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [first, second], state: "keep" });
			tag = manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 1, c: [first, 1, second, 2] });
		});
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset());
		expect(snapshot.sourceCount).toBe(2);
		expect(snapshot.groupCount).toBe(2);
		const suffix = user(f.manager, "outside the confirmation");
		f.manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [suffix], state: "exclude" });
		f.manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [second], state: "keep" });
		const savedTag = structuredClone(f.manager.getEntry(tag));
		expect(await f.preservation.resetPreservedMessageOverrides(snapshot)).toEqual({ reset: 1, skipped: 1 });
		expect(f.changed).toEqual([[first]]);
		const disk = await reopenedQuery(f.manager);
		expect(disk.getManualGroup(first)?.members[0]?.state).toBe("auto");
		expect(disk.getManualGroup(second)?.members[0]?.state).toBe("keep");
		expect(disk.getManualGroup(suffix)?.members[0]?.state).toBe("exclude");
		expect(f.manager.getEntry(tag)).toEqual(savedTag);
		const before = fs.readFileSync(f.manager.getSessionFile()!, "utf8");
		expect(await f.preservation.resetPreservedMessageOverrides(snapshot)).toEqual({ reset: 0, skipped: 0 });
		await f.preservation.setPreservedMessageOverride(first, "auto");
		expect(fs.readFileSync(f.manager.getSessionFile()!, "utf8")).toBe(before);
	});

	it("orders a queued Auto edit after an unflushed Keep rather than treating it as a no-op", async () => {
		let source = "";
		const f = await fixture(manager => { source = user(manager, "queued edits"); });
		const keep = f.preservation.setPreservedMessageOverride(source, "keep");
		const auto = f.preservation.setPreservedMessageOverride(source, "auto");
		await Promise.all([keep, auto]);
		expect((await reopenedQuery(f.manager)).getManualGroup(source)?.members[0]?.state).toBe("auto");
		expect(f.changed).toEqual([[source], [source]]);
	});

	it("sets and resets complete assistant/tool atoms and skips the whole atom after a companion edit", async () => {
		let assistant = "", resultA = "", resultB = "";
		const f = await fixture(manager => {
			user(manager, "run two tools");
			assistant = manager.appendMessage({ role: "assistant", content: [
				{ type: "toolCall", id: "a", name: "read", arguments: {} },
				{ type: "toolCall", id: "b", name: "read", arguments: {} },
			], api: "anthropic-messages", provider: "anthropic", model: "test", stopReason: "toolUse", timestamp: 1,
			usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
			resultA = manager.appendMessage({ role: "toolResult", toolCallId: "a", toolName: "read", content: [{ type: "text", text: "a" }], isError: false, timestamp: 1 });
			resultB = manager.appendMessage({ role: "toolResult", toolCallId: "b", toolName: "read", content: [{ type: "text", text: "b" }], isError: false, timestamp: 1 });
		});
		await f.preservation.setPreservedMessageOverride(resultB, "keep");
		expect(f.changed).toEqual([[assistant, resultA, resultB]]);
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset([assistant, resultB]));
		expect(snapshot.sourceCount).toBe(3);
		expect(snapshot.groupCount).toBe(1);
		f.manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [resultA], state: "exclude" });
		expect(await f.preservation.resetPreservedMessageOverrides(snapshot)).toEqual({ reset: 0, skipped: 3 });
		expect(await f.preservation.resetPreservedMessageOverrides((await f.preservation.capturePreservedMessageOverrideReset()))).toEqual({ reset: 3, skipped: 0 });
		expect((await reopenedQuery(f.manager)).getManualGroup(resultA)?.members.map(member => member.state)).toEqual(["auto", "auto", "auto"]);
	});

	it("rejects lost ancestry, clear epochs, and an A-to-B-to-A ownership change", async () => {
		let source = "";
		const f = await fixture(manager => {
			source = user(manager, "origin");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [source], state: "keep" });
		});
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset());
		const leaf = f.manager.getLeafId()!;
		f.manager.branch(source);
		await f.prepare();
		await expect(f.preservation.resetPreservedMessageOverrides(snapshot)).rejects.toThrow("branch");
		f.manager.branch(leaf);
		f.transition();
		await f.prepare();
		await expect(f.preservation.resetPreservedMessageOverrides(snapshot)).rejects.toThrow("changed");
		const beforeClear = (await f.preservation.capturePreservedMessageOverrideReset());
		f.manager.appendResetBoundary();
		f.transition();
		await f.prepare();
		await expect(f.preservation.resetPreservedMessageOverrides(beforeClear)).rejects.toThrow("changed");
		expect((await f.preservation.capturePreservedMessageOverrideReset()).sourceCount).toBe(0);
		expect(overrides(f.manager)).toHaveLength(1);
		expect(f.changed).toEqual([]);
	});

	it("rolls back a failed atomic write and retries the captured reset without duplicate events", async () => {
		let source = "";
		const f = await fixture(manager => {
			source = user(manager, "keep until durable reset");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [source], state: "keep" });
		});
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset());
		const repairStarted = Promise.withResolvers<void>();
		const repairRelease = Promise.withResolvers<void>();
		f.storage.atomicWriteGate = { reached: repairStarted.resolve, release: repairRelease.promise };
		f.storage.failAtomic = true;
		// Await the real transaction; Bun's synchronous rejection matcher pumps a nested event loop.
		const failure = rejects(f.preservation.resetPreservedMessageOverrides(snapshot), {
			message: "manual atomic publication failed",
		});
		try {
			await repairStarted.promise;
			expect(f.query().getManualGroup(source)?.members[0]?.state).toBe("keep");
			expect(f.changed).toEqual([]);
		} finally {
			repairRelease.resolve();
		}
		await failure;
		expect(f.query().getManualGroup(source)?.members[0]?.state).toBe("keep");
		expect((await reopenedQuery(f.manager)).getManualGroup(source)?.members[0]?.state).toBe("keep");
		expect(f.changed).toEqual([]);
		expect(await f.preservation.resetPreservedMessageOverrides(snapshot)).toEqual({ reset: 1, skipped: 0 });
		expect(overrides(f.manager)).toHaveLength(2);
	});

	it("retries the same logically committed transition after a flush failure", async () => {
		let source = "";
		const f = await fixture(manager => {
			source = user(manager, "durable identity");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [source], state: "keep" });
		});
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset());
		f.onAppend(entry => {
			if (entry.type === "custom" && entry.customType === MESSAGE_OVERRIDE_CUSTOM_TYPE) f.storage.failDrain = true;
		});
		await expect(f.preservation.resetPreservedMessageOverrides(snapshot)).rejects.toThrow("flush failed");
		const committedId = overrides(f.manager).at(-1)!.id;
		expect(f.changed).toEqual([]);
		f.onAppend(() => {});
		expect(await f.preservation.resetPreservedMessageOverrides(snapshot)).toEqual({ reset: 1, skipped: 0 });
		expect(overrides(f.manager).map(entry => entry.id)).toContain(committedId);
		expect(overrides(f.manager)).toHaveLength(2);
		expect(f.changed).toEqual([[source]]);
		expect((await reopenedQuery(f.manager)).getManualGroup(source)?.members[0]?.state).toBe("auto");
	});

	it("does not publish stale current-session success when a branch changes during flush", async () => {
		let source = "";
		const f = await fixture(manager => {
			source = user(manager, "branch A");
			manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [source], state: "keep" });
		});
		const snapshot = (await f.preservation.capturePreservedMessageOverrideReset());
		const gate = Promise.withResolvers<void>();
		const committed = Promise.withResolvers<void>();
		f.onAppend(() => { f.storage.drainGate = gate.promise; committed.resolve(); });
		const pending = f.preservation.resetPreservedMessageOverrides(snapshot);
		await committed.promise;
		f.manager.branch(source);
		f.transition();
		gate.resolve();
		await expect(pending).rejects.toThrow("changed");
		expect(overrides(f.manager)).toHaveLength(2);
		expect(f.changed).toEqual([]);
	});

	it("converts legacy pins and singular overrides in place without changing tags or binary history", async () => {
		let pin = "", singular = "", binary = "", source = "";
		let original: SessionEntry[] = [];
		const f = await fixture(manager => {
			source = user(manager, "legacy");
			pin = manager.appendCustomEntry("com.omp.compaction-preserved", { messageId: source, pinned: true });
			singular = manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageId: source, state: "exclude" });
			binary = manager.appendCustomEntry("com.omp.compaction.preserved-user-messages.v1", { version: 1, preservedIds: [source], classifiedIds: [source] });
			original = structuredClone(manager.getEntries());
		});
		expect(f.manager.getEntries().map(entry => [entry.id, entry.parentId, entry.timestamp])).toEqual(original.map(entry => [entry.id, entry.parentId, entry.timestamp]));
		expect(f.manager.getEntry(pin)).toMatchObject({ customType: MESSAGE_OVERRIDE_CUSTOM_TYPE, data: { messageIds: [source], state: "keep" } });
		expect(f.manager.getEntry(singular)).toMatchObject({ data: { messageIds: [source], state: "exclude" } });
		expect(f.manager.getEntry(binary)).toEqual(original.find(entry => entry.id === binary));
		expect((await reopenedQuery(f.manager)).getManualGroup(source)?.members[0]?.state).toBe("exclude");
		const before = fs.readFileSync(f.manager.getSessionFile()!, "utf8");
		await ensurePreservedMessageStateOnDisk(f.manager);
		expect(fs.readFileSync(f.manager.getSessionFile()!, "utf8")).toBe(before);
	});
	it("keeps original images and links across the first representation rewrite and reload", async () => {
		const image = { type: "image" as const, mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC" };
		const imageLinks = ["https://example.test/original.png"];
		let id = "";
		const f = await fixture(manager => {
			id = manager.appendMessage({ role: "user", content: [{ type: "text", text: "look" }, image], imageLinks, timestamp: 1 });
		});
		const entry = f.manager.getEntry(id)!;
		if (entry.type !== "message" || entry.message.role !== "user") throw new Error("Missing source");
		const message = entry.message;
		await f.manager.rewriteEntries([{ entryId: id, blocks: [{ oldBlockIndex: 0, newBlockIndex: 0 }, { oldBlockIndex: 1, newBlockIndex: null }] }], () => {
			entry.message = { ...message, content: [{ type: "text", text: "look" }] };
		});
		const reopened = await SessionManager.open(f.manager.getSessionFile()!);
		try {
			const restored = reopened.getEntry(id)!;
			if (restored.type !== "message" || restored.message.role !== "user") throw new Error("Missing restored source");
			expect(restored.message.content).toEqual([{ type: "text", text: "look" }]);
			const query = await PreservedMessageQuery.build(reopened.getBranch(), policy, new Tokenizer(), { isCurrent: () => true });
			expect(query!.inspectCandidate(id, true)!.message).toMatchObject({ content: [{ type: "text", text: "look" }, image] });
			expect(toRestoredQueuedMessage(restored.message)).toMatchObject({ text: "look", images: [image], imageLinks });
		} finally { await reopened.close(); }
	});
});
