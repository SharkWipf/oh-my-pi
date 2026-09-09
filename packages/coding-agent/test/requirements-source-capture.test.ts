import { afterEach, describe, expect, test } from "bun:test";
import { unlink, writeFile } from "node:fs/promises";
import { TempDir } from "@oh-my-pi/pi-utils";
import { resolveRequirementsSource } from "../src/requirements/source-capture";
import { SessionManager } from "../src/session/session-manager";
import { FileSessionStorage } from "../src/session/session-storage";
import { createAssistantMessage } from "./helpers/agent-session-setup";

const managers: SessionManager[] = [];
afterEach(async () => {
	for (const manager of managers.splice(0)) await manager.close();
});
function manager(): SessionManager {
	const result = SessionManager.inMemory("/tmp/requirements-source-regression");
	managers.push(result);
	return result;
}
function append(session: SessionManager, text: string) {
	return session.appendMessage({ role: "user", producer: { type: "human" }, content: text, timestamp: 1 });
}

// These regressions replace the former capture-ID/byte-remapping contract.
test("indexed originals bind the current journal namespace and revoke a reset during the first point read", async () => {
	class PausedIndexStorage extends FileSessionStorage {
		pause = false;
		entered = Promise.withResolvers<void>();
		release = Promise.withResolvers<void>();
		override async readJsonlLinesById(filePath: string, ids: ReadonlySet<string>) {
			const read = super.readJsonlLinesById(filePath, ids);
			if (this.pause) {
				this.pause = false;
				this.entered.resolve();
				await this.release.promise;
			}
			return read;
		}
	}
	using temp = TempDir.createSync("requirements-index-authority-");
	const storage = new PausedIndexStorage();
	const session = SessionManager.create(temp.path(), temp.path(), storage);
	try {
		const id = append(session, "Authoritative original A");
		await session.ensureOnDisk(); await session.flush();
		const source = session.getRequirementsSource(id)!;
		storage.pause = true;
		const interrupted = session.resolveRequirementsEvidence(source.key, source);
		await storage.entered.promise;
		session.appendResetBoundary(); await session.flush();
		storage.release.resolve();
		expect(await interrupted).toBeUndefined();
		expect((await session.resolveRequirementsEvidence(source.key, source, { context: false }))?.units[0].text).toBe("Authoritative original A");
		const file = session.getSessionFile()!;
		const journal = await storage.readText(file);
		const currentId = session.getSessionId();
		const otherId = (currentId[0] === "0" ? "1" : "0") + currentId.slice(1);
		await storage.writeText(file, journal.replace(JSON.stringify(session.getSessionId()), JSON.stringify(otherId)));
		expect(await session.resolveRequirementsEvidence(source.key, source, { context: false })).toBeUndefined();
	} finally { await session.close(); }
});


describe("accepted original source authority", () => {
	test("independent accepted entries retain identity and pre-expansion image ordering", async () => {
		const session = manager();
		const originalSubmission = { text: "literal /command", images: [{ type: "image" as const, data: "aW1n", mimeType: "image/png" }], imageLinks: ["file:///original.png"] };
		const first = session.appendMessage({ role: "user", producer: { type: "human" }, content: "expanded", originalSubmission, timestamp: 1 });
		const second = session.appendMessage({ role: "user", producer: { type: "human" }, content: "expanded", originalSubmission, timestamp: 1 });
		const catalog = await session.getRequirementsSources();
		expect(catalog.sources.map(source => source.original.entryId)).toEqual([first, second]);
		expect(catalog.context).toEqual([]);
		expect(catalog.sources.every(source => !source.integrityAvailable && source.units.length === 0)).toBe(true);
		const resolved = (await resolveRequirementsSource(session, catalog.sources[0].key))!;
		expect(resolved.units).toEqual([{ id: "0", text: "literal /command" }, { id: "1", image: originalSubmission.images[0] }]);
		expect(resolved.source.origin.kind).toBe("human");
		expect(resolved.source.units.map(unit => unit.kind)).toEqual(["text", "image"]);
	});

	test("role and attribution cannot attest legacy or tool-generated input", async () => {
		const session = manager();
		session.appendMessage({ role: "user", attribution: "user", content: "legacy", timestamp: 1 });
		session.appendMessage({ role: "user", attribution: "user", producer: { type: "tool", name: "goal", toolCallId: "call" }, content: "generated", timestamp: 2 });
		const sources = (await session.getRequirementsSources()).sources;
		expect(sources.map(source => [source.origin.kind, source.state])).toEqual([["unknown", "unsupported"], ["tool", "unsupported"]]);
		expect((await resolveRequirementsSource(session, sources[0].key))?.source.origin.kind).toBe("unknown");
		expect(sources[1].origin.producerId).toBe("call");
	});

	test("catalog records missing originals without eagerly loading their depot", async () => {
		const session = manager();
		const id = session.appendMessage({ role: "user", producer: { type: "human" }, content: "expanded", originalSubmission: {
			text: "inspect image", images: [{ type: "image", data: `blob:sha256:${"0".repeat(64)}`, mimeType: "image/png" }],
		}, timestamp: 1 });
		const source = session.getRequirementsSource(id)!;
		expect(source.state).toBe("pending");
		expect((await session.getRequirementsSources()).sources[0].key).toBe(source.key);
		expect(await resolveRequirementsSource(session, source.key)).toBeUndefined();
		expect((await session.observeRequirementsEvidence([source]))[0].integrity).toBeNull();
	});

	test("branch and reset invalidate current dependencies without destroying historical inspection", async () => {
		const session = manager();
		const first = append(session, "standing instruction");
		const firstSource = (await resolveRequirementsSource(session, session.getRequirementsSource(first)!.key))!.source;
		const sibling = append(session, "abandoned correction");
		const siblingSource = (await resolveRequirementsSource(session, session.getRequirementsSource(sibling)!.key))!.source;
		session.branch(first);
		expect((await session.observeRequirementsEvidence([firstSource, siblingSource])).map(item => item.integrity)).toEqual([firstSource.integrity, null]);
		session.appendResetBoundary();
		expect(session.getRequirementsEpoch()).toBe(1);
		expect((await session.getRequirementsSources()).sources).toEqual([]);
		expect((await session.observeRequirementsEvidence([firstSource]))[0].integrity).toBeNull();
		expect((await resolveRequirementsSource(session, firstSource.key))!.units[0].text).toBe("standing instruction");
	});

	test("changed whole units invalidate exact old versions while frozen units never shift", async () => {
		const session = manager();
		const id = append(session, "é😀 old suffix");
		const frozen = (await resolveRequirementsSource(session, session.getRequirementsSource(id)!.key))!;
		const entry = session.getEntry(id)!;
		if (entry.type !== "message" || entry.message.role !== "user") throw new Error("Missing accepted source");
		entry.message.content = "suffix";
		const observation = (await session.observeRequirementsEvidence([frozen.source]))[0];
		expect(observation.integrity).not.toBe(frozen.source.integrity);
		expect(frozen.units[0].text).toBe("é😀 old suffix");
		session.invalidateRequirementsSources([id]);
		expect(await resolveRequirementsSource(session, frozen.source.key)).toBeUndefined();
		expect(session.getRequirementsSource(id)!.state).toBe("orphaned");
	});

	test("nonhuman whole-unit context explains a decision without joining its authority chain", async () => {
		const session = manager();
		const tool = session.appendMessage({ role: "toolResult", toolCallId: "call", toolName: "read", content: [{ type: "text", text: "Option 2 is 42" }], isError: false, timestamp: 1 });
		const id = append(session, "Use option 2");
		const resolved = (await resolveRequirementsSource(session, session.getRequirementsSource(id)!.key))!;
		expect((await session.getRequirementsSources()).sources.map(source => source.original.entryId)).toEqual([id]);
		expect(resolved.referents[0].source.original.entryId).toBe(tool);
		expect(resolved.referents[0].source.referenceOnly).toBe(true);
		expect(resolved.referents[0].units[0].text).toBe("Option 2 is 42");
		expect(resolved.contextIndex).toBe(1);
	});

	test("journal origins survive fork while foreign deletion invalidates current dependency tokens", async () => {
		using temp = TempDir.createSync("requirements-journal-originals-");
		const session = SessionManager.create(temp.path(), temp.path());
		try {
			const id = append(session, "original policy");
			await session.ensureOnDisk();
			await session.flush();
			const key = session.getRequirementsSource(id)!.key;
			const original = (await resolveRequirementsSource(session, key))!.source;
			const filename = session.getSessionFile()!;
			const fork = await SessionManager.forkFrom(filename, temp.path(), temp.path());
			try { expect((await resolveRequirementsSource(fork, key))!.source.original).toEqual(original.original); }
			finally { await fork.close(); }
			const foreign = manager();
			expect((await foreign.observeRequirementsEvidence([original]))[0].integrity).toBe(original.integrity);
			const version = foreign.getRequirementsSourceVersion();
			await unlink(filename);
			expect(foreign.getRequirementsSourceVersion()).not.toBe(version);
			expect((await foreign.observeRequirementsEvidence([original]))[0].integrity).toBeNull();
		} finally { await session.close(); }
	});
	test("explicit retention survives deletion but never overrides an existing rewrite or authored invalidation", async () => {
		using temp = TempDir.createSync("requirements-selected-retention-");
		const session = manager();
		const id = append(session, "selected original");
		const source = (await resolveRequirementsSource(session, session.getRequirementsSource(id)!.key))!.source;
		const journalPath = temp.join("original.jsonl");
		const journal = () => [session.getHeader(), ...session.getEntries()].map(entry => JSON.stringify(entry)).join("\n") + "\n";
		await writeFile(journalPath, journal());
		source.locators[0].journalPath = journalPath;
		const retained = await session.retainRequirementsEvidence(source);
		await unlink(journalPath);
		expect((await session.observeRequirementsEvidence([retained]))[0].integrity).toBe(source.integrity);
		expect((await session.observeRequirementsEvidence([source]))[0].integrity).toBeNull();
		const entry = session.getEntry(id)!;
		if (entry.type !== "message" || entry.message.role !== "user") throw new Error("Missing source");
		entry.message.content = "authored replacement";
		await writeFile(journalPath, journal());
		expect((await session.observeRequirementsEvidence([retained]))[0].integrity).not.toBe(source.integrity);
		session.invalidateRequirementsSources([id]);
		await unlink(journalPath);
		expect((await session.observeRequirementsEvidence([retained]))[0].integrity).toBeNull();
	});
	test("processing uses the normal compacted context and keeps whole originals outside that view", async () => {
		const session = manager();
		const distant = append(session, "Complete distant authored instruction, never truncate this original.");
		for (let index = 0; index < 4096; index++) append(session, `omitted history ${index}`);
		const kept = session.appendMessage({ role: "user", producer: { type: "human" }, content: "expanded retained delivery", originalSubmission: { text: "Whole retained pre-expansion instruction" }, timestamp: 1 });
		session.appendCompaction("Context summary, not authored source evidence", undefined, kept, 100000);
		const current = session.appendMessage({ role: "user", producer: { type: "human" }, content: "expanded current delivery", originalSubmission: { text: "Whole current pre-expansion instruction" }, timestamp: 2 });
		const later = append(session, "Future source must not enter the earlier processing context");
		const source = (await session.resolveRequirementsEvidence(session.getRequirementsSource(current)!.key))!;
		expect(session.getLeafId()).toBe(later);
		expect(source.context.map(message => message.role)).toEqual(["compactionSummary", "user", "user"]);
		expect(source.units[0].text).toBe("Whole current pre-expansion instruction");
		expect(source.contextIndex).toBeUndefined();
		expect(source.referents.map(reference => [reference.source.original.entryId, reference.units[0].text, reference.contextIndex])).toEqual([[kept, "Whole retained pre-expansion instruction", undefined]]);
		expect(source.unavailableContext).toEqual([]);
		session.branch(kept);
		const sibling = append(session, "Another branch must not enter the addressed source context");
		const historical = await session.resolveRequirementsEvidence(source.source.key, source.source);
		expect(historical?.context).toEqual(source.context);
		expect(session.getLeafId()).toBe(sibling);
		const addressed = await session.resolveRequirementsEvidence(session.getRequirementsSource(distant)!.key, undefined, { context: false });
		expect(addressed?.units[0].text).toBe("Complete distant authored instruction, never truncate this original.");
	});
	test("later assistant and human entries preserve original A while reset still revokes it", async () => {
		const session = manager();
		for (let index = 0; index < 512; index++) append(session, `history ${index}`);
		const id = append(session, "current exact original");
		const source = session.getRequirementsSource(id)!;
		const resolving = session.resolveRequirementsEvidence(source.key, source);
		await Bun.sleep(0);
		session.appendMessage(createAssistantMessage("unrelated later response"));
		const resolved = await resolving;
		expect(resolved?.units[0].text).toBe("current exact original");
		expect(resolved?.context).toHaveLength(513);
		const catalogVersion = session.getRequirementsSourceVersion();
		const priorHuman = session.resolveRequirementsEvidence(source.key, source);
		await Bun.sleep(0);
		const laterHuman = append(session, "new independent human request");
		expect(session.getRequirementsSourceVersion()).not.toBe(catalogVersion);
		expect(session.getRequirementsSource(laterHuman)!.key).not.toBe(source.key);
		expect((await priorHuman)?.units[0].text).toBe("current exact original");
		const abandoned = session.resolveRequirementsEvidence(source.key, source);
		await Bun.sleep(0);
		session.appendResetBoundary();
		expect(await abandoned).toBeUndefined();
	});

	test("a concurrent journal append is not a missing source, but a changed original still fails the fence", async () => {
		using temp = TempDir.createSync("requirements-journal-read-race-");
		class PausedStorage extends FileSessionStorage {
			pause?: { entered: () => void; release: Promise<void> };
			override async readText(path: string): Promise<string> {
				const text = await super.readText(path);
				const pause = this.pause;
				this.pause = undefined;
				if (pause) { pause.entered(); await pause.release; }
				return text;
			}
		}
		const storage = new PausedStorage();
		const session = SessionManager.create(temp.path(), temp.path(), storage);
		try {
			const id = append(session, "exact committed original");
			await session.ensureOnDisk();
			await session.flush();
			const source = session.getRequirementsSource(id)!;
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			storage.pause = { entered: entered.resolve, release: release.promise };
			const resolving = session.resolveRequirementsEvidence(source.key, source, { context: false });
			await entered.promise;
			session.appendMessage(createAssistantMessage("concurrent response"));
			await session.flush();
			release.resolve();
			expect((await resolving)?.units[0].text).toBe("exact committed original");
			const changedEntered = Promise.withResolvers<void>();
			const changedRelease = Promise.withResolvers<void>();
			storage.pause = { entered: changedEntered.resolve, release: changedRelease.promise };
			const changed = session.resolveRequirementsEvidence(source.key, source, { context: false });
			await changedEntered.promise;
			const entry = session.getEntry(id)!;
			if (entry.type !== "message" || entry.message.role !== "user") throw new Error("Missing original");
			entry.message.content = "authored replacement";
			await session.rewriteEntries();
			changedRelease.resolve();
			expect(await changed).toBeUndefined();
		} finally { await session.close(); }
	});
});
