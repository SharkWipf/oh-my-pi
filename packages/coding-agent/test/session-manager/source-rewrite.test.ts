import { describe, expect, it } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { SourceRepresentation, SourceRewrite } from "@oh-my-pi/pi-agent-core/compaction/source";
import type { AssistantMessage, Message } from "@oh-my-pi/pi-ai";
import type { NativeItemOrigin, NativeSourcePart } from "@oh-my-pi/pi-ai/utils/source-origin";
import {
	decodePreservedUserMessageClassifications,
	INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	MESSAGE_OVERRIDE_CUSTOM_TYPE,
	packPreservedUserMessageClassifications,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
} from "../../src/session/preserved-message-settings";
import type { CompactionEntry, CustomEntry, SessionEntry, SessionMessageEntry } from "../../src/session/session-entries";
import { SessionManager } from "../../src/session/session-manager";
import { FileSessionStorage, type WriteTextAtomicOptions } from "../../src/session/session-storage";

class GatedFileStorage extends FileSessionStorage {
	gate = false;
	readonly started = Promise.withResolvers<void>();
	readonly release = Promise.withResolvers<void>();
	override async writeTextAtomic(path: string, content: string, options?: WriteTextAtomicOptions): Promise<void> {
		if (this.gate) {
			this.started.resolve();
			await this.release.promise;
		}
		await super.writeTextAtomic(path, content, options);
	}
}

function user(manager: SessionManager, content: string): string {
	return manager.appendMessage({ role: "user", content, timestamp: 0 });
}
function tags(manager: SessionManager, ids: string[]): string {
	return manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
		packPreservedUserMessageClassifications(ids.map(id => ({ id, mask: 1 }))));
}
function message(manager: SessionManager, id: string): SessionMessageEntry & { message: Message } {
	return manager.getEntry(id) as SessionMessageEntry & { message: Message };
}
function categories(entries: readonly SessionEntry[], id: string): string[] {
	const entry = entries.find(entry => entry.id === id) as CustomEntry;
	const decoded = decodePreservedUserMessageClassifications(entry.data);
	if (decoded.status !== "valid") throw new Error("Expected successful classification record");
	return decoded.classifications.map(item => item.id);
}
function representation(entryId: string): SourceRepresentation {
	return {
		version: 1,
		coverage: [{ entryId, order: 0, snapshot: { blockIndex: 1, start: 0, end: 6 },
			current: { blockIndex: 1, start: 0, end: 6 }, normalized: { start: 0, end: 6 }, status: "exact-current" }],
		layout: [{ kind: "frame", frameIndex: 0, range: { start: 0, end: 6 } }],
	};
}
function assistant(manager: SessionManager, text: string): string {
	const entry: AssistantMessage = {
		role: "assistant", content: [{ type: "text", text }], api: "openai-responses", provider: "openai", model: "gpt-4.1",
		stopReason: "stop", timestamp: 0,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	};
	return manager.appendMessage(entry);
}

describe("SessionManager controlled source rewrites", () => {
	it("publishes replacement plus all old/sibling coverage and shadow invalidation together, then reloads", async () => {
		const directory = await mkdtemp(join(tmpdir(), "omp-source-rewrite-"));
		const storage = new GatedFileStorage();
		const manager = await SessionManager.inMemory(directory).persistCopy({ sessionDir: directory, suppressBreadcrumb: true }, storage);
		try {
			const unaffected = user(manager, "independent");
			const unrelatedOpaque = manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 99, c: [unaffected, 1] });
			const rawUnrelatedOpaque = structuredClone(manager.getEntry(unrelatedOpaque));
			const root = manager.appendMessage({ role: "user", timestamp: 0, content: [
				{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }, { type: "text", text: "ABCDEF" },
			] }, { compactionOverride: "keep" });
			const oldTags = tags(manager, [unaffected, root]);
			const captured: NativeSourcePart = { entryId: root, order: 0, blockIndex: 1, coverage: "full", representation: "native", sourceSpan: { start: 0, end: 6 }, transportSpan: { start: 0, end: 6 } };
			const origins: NativeItemOrigin[] = [{ kind: "source", parts: [captured] }];
			const items = [{ type: "message", role: "user", content: [{ type: "input_text", text: "ABCDEF" }] }];
			const native = { replacementHistory: items, replacementOrigins: origins, allUserSources: [captured] };
			const payloadEntry = manager.appendMessage({ role: "user", content: "", timestamp: 0, synthetic: true,
				providerPayload: { type: "openaiResponsesHistory", items, origins } });
			const oldCompaction = manager.appendCompaction("older", undefined, root, 10,
				{ preserveData: { sourceRepresentation: representation(root), openaiRemoteCompaction: native, archive: { text: "ABCDEF", frames: ["immutable-png"] } } });
			const left = user(manager, "left");
			const leftTags = tags(manager, [root, left, unaffected]);
			const opaque = manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 99, c: [root, 7], evidence: "retain raw" });
			const malformed = manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 1, c: [root, "not a mask"] });
			const rawMalformed = structuredClone(manager.getEntry(malformed)) as CustomEntry;
			const manual = manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: [root], state: "keep" });
			const siblingCompaction = manager.appendCompaction("sibling", undefined, root, 10,
				{ preserveData: { sourceRepresentation: representation(root), openaiRemoteCompaction: native } });
			manager.branch(oldCompaction);
			const right = user(manager, "right");
			const rightTags = tags(manager, [root, right]);
			manager.appendResetBoundary();
			const afterClear = user(manager, "after clear");
			const clearTags = tags(manager, [afterClear]);
			await manager.ensureOnDisk();
			await manager.flush();
			const file = manager.getSessionFile()!;
			const before = await readFile(file, "utf8");
			const rawOpaque = structuredClone(manager.getEntry(opaque)) as CustomEntry;
			const rawManual = structuredClone(manager.getEntry(manual));
			const rewrites: SourceRewrite[] = [{ entryId: root, blocks: [
				{ oldBlockIndex: 0, newBlockIndex: null },
				{ oldBlockIndex: 1, newBlockIndex: 0, textEdits: [{ start: 2, end: 4, replacementLength: 3 }] },
			] }];
			let affected: readonly string[] = [];
			storage.gate = true;
			const publication = manager.rewriteEntries(rewrites, () => {
				message(manager, root).message.content = [{ type: "text", text: "ABxyZEF" }];
			}, ids => { affected = ids; });
			await storage.started.promise;
			expect(await readFile(file, "utf8")).toBe(before);
			expect(new Set(affected)).toEqual(new Set([root, left, right]));
			storage.release.resolve();
			await publication;
			await manager.close();
			const reloaded = await SessionManager.open(file, directory, new FileSessionStorage(), { suppressBreadcrumb: true });
			try {
				const entries = reloaded.getEntries();
				expect(message(reloaded, root).message.content).toEqual([{ type: "text", text: "ABxyZEF" }]);
				expect(message(reloaded, root).compactionOverride).toBe("keep");
				expect(categories(entries, oldTags)).toEqual([unaffected]);
				expect(categories(entries, leftTags)).toEqual([unaffected]);
				expect(categories(entries, rightTags)).toEqual([]);
				expect(categories(entries, clearTags)).toEqual([afterClear]);
				expect(reloaded.getEntry(opaque)).toEqual({ ...rawOpaque, customType: INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE });
				expect(reloaded.getEntry(manual)).toEqual(rawManual);
				expect(reloaded.getEntry(unrelatedOpaque)).toEqual(rawUnrelatedOpaque);
				expect(reloaded.getEntry(malformed)).toEqual({ ...rawMalformed, customType: INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE });
				for (const id of [oldCompaction, siblingCompaction]) {
					const data = (reloaded.getEntry(id) as CompactionEntry).preserveData!;
					const mapped = data.sourceRepresentation as SourceRepresentation;
					expect(mapped.coverage.filter(run => run.status === "exact-current").map(run => run.current))
						.toEqual([{ blockIndex: 0, start: 0, end: 2 }, { blockIndex: 0, start: 5, end: 7 }]);
					expect(mapped.coverage.find(run => run.status === "historical-not-current")?.snapshot)
						.toEqual({ blockIndex: 1, start: 2, end: 4 });
					const remote = data.openaiRemoteCompaction as typeof native;
					const mappedOrigin = remote.replacementOrigins[0];
					if (mappedOrigin?.kind !== "source") throw new Error("Expected source origin");
					expect(mappedOrigin.parts.filter(part => part.status === "exact-current").map(part => part.currentSourceSpan))
						.toEqual([{ start: 0, end: 2 }, { start: 5, end: 7 }]);
					expect(remote.allUserSources).toEqual(mappedOrigin.parts);
					expect(remote.replacementHistory).toEqual(items);
				}
				const payload = message(reloaded, payloadEntry).message;
				if (!("providerPayload" in payload) || payload.providerPayload?.type !== "openaiResponsesHistory") throw new Error("Missing native payload");
				expect(payload.providerPayload.items).toEqual(items);
				expect(payload.providerPayload.origins).toEqual(((reloaded.getEntry(oldCompaction) as CompactionEntry).preserveData!.openaiRemoteCompaction as typeof native).replacementOrigins);
				expect((reloaded.getEntry(oldCompaction) as CompactionEntry).preserveData?.archive)
					.toEqual({ text: "ABCDEF", frames: ["immutable-png"] });
			} finally { await reloaded.close(); }
		} finally {
			storage.release.resolve();
			await manager.close();
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("uses the exact auxiliary projection: tool results/reasoning do not invalidate, assistant text does until displaced", async () => {
		const manager = SessionManager.inMemory();
		user(manager, "first");
		const changed = assistant(manager, "guidance");
		const tool = manager.appendMessage({ role: "toolResult", toolCallId: "call", toolName: "read", isError: false,
			content: [{ type: "text", text: "result" }], timestamp: 0 });
		const one = user(manager, "one");
		assistant(manager, "newer guidance");
		const two = user(manager, "two");
		assistant(manager, "second newer guidance");
		const three = user(manager, "three");
		const record = tags(manager, [one, two, three]);
		await manager.rewriteEntries([{ entryId: tool }], () => {
			message(manager, tool).message.content = [{ type: "text", text: "different result" }];
		});
		await manager.rewriteEntries([{ entryId: changed }], () => {
			(message(manager, changed).message as AssistantMessage).content.push({ type: "thinking", thinking: "private" });
		});
		expect(categories(manager.getEntries(), record)).toEqual([one, two, three]);
		await manager.rewriteEntries([{ entryId: changed }], () => {
			message(manager, changed).message.content = [{ type: "text", text: "corrected guidance" }];
		});
		expect(categories(manager.getEntries(), record)).toEqual([three]);
	});
});
