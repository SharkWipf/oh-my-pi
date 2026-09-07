import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { Agent, type StreamFn } from "@oh-my-pi/pi-agent-core";
import { type AssistantMessage, type Context, createAssistantMessageEventStream } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { TempDir } from "@oh-my-pi/pi-utils";
import { ModelRegistry } from "../src/config/model-registry";
import { Settings } from "../src/config/settings";
import { AgentSession } from "../src/session/agent-session";
import type { AuthStorage } from "../src/session/auth-storage";
import {
	packPreservedUserMessageClassifications,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
} from "../src/session/preserved-message-settings";
import { readPreservedUserMessageClassificationMasks } from "../src/session/preserved-messages";
import { SessionManager } from "../src/session/session-manager";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

const model = getBundledModel("openai", "gpt-4o")!;
function reply(text: string): AssistantMessage {
	return {
		role: "assistant", content: [{ type: "text", text }], api: model.api, provider: model.provider,
		model: model.id, stopReason: "stop", timestamp: 1,
		usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	};
}

// A real stream boundary whose results deliberately ignore abort: late provider
// replies must be rejected by session ownership, not merely by cooperative I/O.
class ControlledProvider {
	readonly requests: { context: Context; signal?: AbortSignal; finish(text?: string): void }[] = [];
	readonly stream: StreamFn = (_model, context, options) => {
		const stream = createAssistantMessageEventStream();
		let finished = false;
		this.requests.push({ context, signal: options?.signal,
			finish(text = "<labels>00000000000</labels>") {
				if (finished) return;
				finished = true;
				stream.push({ type: "done", reason: "stop", message: reply(text) });
			},
		});
		return stream;
	};
	forSource(text: string) {
		const request = this.requests.find(request => request.context.messages.some(message =>
			Array.isArray(message.content) && message.content.some(part => part.type === "text" && part.text === text)));
		if (!request) throw new Error(`No classifier request for ${text}`);
		return request;
	}
}

async function until(predicate: () => boolean): Promise<void> {
	const deadline = Date.now() + 3000;
	while (!predicate()) {
		if (Date.now() >= deadline) throw new Error("Classifier state did not settle");
		await new Promise<void>(resolve => setImmediate(resolve));
	}
}

describe("message classifier session jobs", () => {
	let dir: TempDir;
	let auth: AuthStorage;
	let registry: ModelRegistry;
	let provider: ControlledProvider;
	let manager: SessionManager;
	let session: AgentSession;
	const sessions: AgentSession[] = [];
	const managers: SessionManager[] = [];

	function createSession(source: SessionManager, live = false): AgentSession {
		const settings = Settings.isolated({
			"compaction.enabled": false,
			"compaction.keepUserMessages": true,
			"compaction.keepUserMessagesLlm": live,
			"compaction.keepUserMessagesLlmModel": `${model.provider}/${model.id}`,
		});
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			stream.push({ type: "done", reason: "stop", message: reply("Synthetic main response") });
			return stream;
		};
		const created = new AgentSession({
			agent: new Agent({ initialState: { model, systemPrompt: [], tools: [], messages: [] }, streamFn, getApiKey: () => "synthetic-key" }),
			sessionManager: source, settings, modelRegistry: registry, sideStreamFn: provider.stream,
		});
		sessions.push(created);
		return created;
	}
	function user(text: string): string {
		return manager.appendMessage({ role: "user", content: text, timestamp: 1 });
	}
	async function facts(source = manager) {
		const masks = await readPreservedUserMessageClassificationMasks(source.getBranch(), { isCurrent: () => true });
		if (!masks) throw new Error("Uninterrupted fact read unexpectedly canceled");
		return masks;
	}
	function save(id: string, mask: number) {
		manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, packPreservedUserMessageClassifications([{ id, mask }]));
	}
	function job(id: string) {
		const status = session.getMessageClassificationStatus().jobs.find(job => job.id === id);
		if (!status) throw new Error(`Missing classifier job ${id}`);
		return status;
	}
	async function settled(id: string) { await until(() => job(id).state !== "running"); }
	async function reopen() {
		await manager.flush();
		const reopened = await SessionManager.open(manager.getSessionFile()!, dir.path(), undefined, { suppressBreadcrumb: true });
		managers.push(reopened);
		return reopened;
	}

	beforeEach(() => {
		dir = TempDir.createSync("message-classifier-jobs-");
		auth = createInMemoryAuthStorage();
		auth.setRuntimeApiKey(model.provider, "synthetic-key");
		registry = new ModelRegistry(auth, dir.join("models.yml"));
		provider = new ControlledProvider();
		manager = SessionManager.create(dir.path(), dir.path());
		session = createSession(manager);
	});
	afterEach(async () => {
		for (const session of sessions.splice(0)) await session.dispose();
		for (const request of provider.requests) request.finish();
		for (const manager of managers.splice(0)) await manager.close();
		auth.close();
		await dir.remove();
	});

	it("persists valid eleven-bit zero as success, retaining prior facts through failed and pending reclassification", async () => {
		const id = user("synthetic prior fact");
		save(id, 1 << 10);
		const failed = await session.startMessageClassification(id);
		await until(() => provider.requests.length === 1);
		expect((await facts()).get(id)).toBe(1 << 10);
		expect(session.getMessageClassificationRowStatus(id)?.state).toBe("running");
		provider.requests[0]!.finish("<labels>0000000000</labels>");
		await settled(failed);
		expect(job(failed).failed).toBe(1);
		expect(session.getMessageClassificationRowStatus(id)?.state).toBe("failed");
		expect((await facts(await reopen())).get(id)).toBe(1 << 10);

		const retry = await session.startMessageClassification(id);
		await until(() => provider.requests.length === 2);
		expect((await facts()).get(id)).toBe(1 << 10);
		provider.requests[1]!.finish();
		await settled(retry);
		expect(job(retry).state).toBe("completed");
		expect(job(retry).saved).toBe(1);
		expect((await facts(await reopen())).get(id)).toBe(0);
	});

	it("backfills only missing facts including unknown versions, accepts >32 workers, and reports each saved row", async () => {
		const known = user("synthetic known zero");
		const unknown = user("synthetic future version");
		const missing = user("synthetic missing fact");
		save(known, 0);
		manager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, { v: 99, c: [unknown, 1024] });
		const savedRows: string[] = [];
		const unsubscribe = session.subscribeMessageClassification((status, affected) => {
			for (const id of affected) if (status.rows.some(row => row.entryId === id && row.state === "saved")) savedRows.push(id);
		});
		const backfill = await session.startMessageClassificationBackfill(33);
		await until(() => provider.requests.length === 2);
		provider.forSource("synthetic future version").finish("<labels>00000000001</labels>");
		await until(() => savedRows.includes(unknown));
		expect(job(backfill).state).toBe("running");
		expect((await facts(await reopen())).get(unknown)).toBe(1024);
		provider.forSource("synthetic missing fact").finish();
		await settled(backfill);
		expect(new Set(savedRows)).toEqual(new Set([unknown, missing]));
		expect(job(backfill).saved).toBe(2);
		expect(provider.requests).toHaveLength(2);
		unsubscribe();
		const complete = await session.startMessageClassificationBackfill(100);
		await settled(complete);
		expect(job(complete).saved).toBe(0);
		expect(provider.requests).toHaveLength(2);
	});

	it("continues serial backfill after its menu unsubscribes", async () => {
		const first = user("synthetic serial one");
		const second = user("synthetic serial two");
		let notifications = 0;
		const unsubscribe = session.subscribeMessageClassification(() => notifications++);
		const backfill = await session.startMessageClassificationBackfill(1);
		await until(() => provider.requests.length === 1);
		unsubscribe();
		const closedAt = notifications;
		provider.requests[0]!.finish();
		await until(() => provider.requests.length === 2);
		provider.requests[1]!.finish();
		await settled(backfill);
		expect(notifications).toBe(closedAt);
		expect(await facts(await reopen())).toEqual(new Map([[first, 0], [second, 0]]));
	});

	it("canceling backfill neither cancels selected nor live work and never saves its late response", async () => {
		const selected = user("synthetic selected");
		const canceled = user("synthetic canceled backfill");
		const selectedJob = await session.startMessageClassification(selected);
		await until(() => provider.requests.length === 1);
		const backfill = await session.startMessageClassificationBackfill(1);
		await until(() => provider.requests.length === 2);
		session.settings.override("compaction.keepUserMessagesLlm", true);
		await session.prompt("synthetic live source");
		await until(() => provider.requests.length === 3);
		const liveJob = session.getMessageClassificationStatus().jobs.find(job => job.kind === "live")!;
		expect(liveJob.state).toBe("running");
		session.cancelMessageClassification(backfill);
		expect(job(backfill).state).toBe("canceled");
		expect(session.getMessageClassificationRowStatus(canceled)?.state).toBe("canceled");
		expect(provider.forSource("synthetic canceled backfill").signal?.aborted).toBe(true);
		expect(provider.forSource("synthetic selected").signal?.aborted).toBe(false);
		expect(provider.forSource("synthetic live source").signal?.aborted).toBe(false);
		for (const request of provider.requests) request.finish();
		await settled(selectedJob);
		await settled(liveJob.id);
		await until(() => job(backfill).running === 0);
		const persisted = await facts(await reopen());
		expect(persisted.get(selected)).toBe(0);
		expect(persisted.has(canceled)).toBe(false);
		expect(persisted.size).toBe(2);
	});

	it("suppresses a result when source content is rewritten after the request starts", async () => {
		const id = user("synthetic original source");
		const started = await session.startMessageClassification(id);
		await until(() => provider.requests.length === 1);
		const entry = manager.getEntry(id)!;
		if (entry.type !== "message" || entry.message.role !== "user") throw new Error("Missing fixture source");
		entry.message.content = "synthetic rewritten source";
		await manager.rewriteEntries();
		provider.requests[0]!.finish();
		await settled(started);
		expect(job(started).state).toBe("failed");
		expect((await facts(await reopen())).has(id)).toBe(false);
	});

	it("never appends a late sibling result or steals the active branch leaf", async () => {
		const root = user("synthetic common ancestor");
		const sibling = user("synthetic old sibling");
		const started = await session.startMessageClassification(sibling);
		await until(() => provider.requests.length === 1);
		manager.branch(root);
		const active = user("synthetic active sibling");
		provider.requests[0]!.finish();
		await settled(started);
		expect(job(started).state).toBe("interrupted");
		expect(manager.getLeafId()).toBe(active);
		const persisted = await reopen();
		expect(persisted.getLeafId()).toBe(active);
		expect(persisted.getEntries().filter(entry => entry.type === "custom" && entry.customType === USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE)).toEqual([]);
	});

	it("clear does not await blocked classification, rejects its late result, and excludes pre-clear sources from backfill", async () => {
		const old = user("synthetic pre-clear source");
		const started = await session.startMessageClassification(old);
		await until(() => provider.requests.length === 1);
		expect(await session.resetSessionContext()).toBeDefined();
		expect(job(started).state).toBe("interrupted");
		const current = user("synthetic post-clear source");
		provider.requests[0]!.finish();
		const backfill = await session.startMessageClassificationBackfill(2);
		await until(() => provider.requests.length === 2);
		provider.forSource("synthetic post-clear source").finish();
		await settled(backfill);
		expect(await facts(await reopen())).toEqual(new Map([[current, 0]]));
	});

	it("dispose is nonblocking and reload preserves facts without restarting interrupted jobs", async () => {
		const saved = user("synthetic durable source");
		const pending = user("synthetic interrupted source");
		save(saved, 3);
		const started = await session.startMessageClassification(pending);
		await until(() => provider.requests.length === 1);
		const file = manager.getSessionFile()!;
		await session.dispose();
		expect(job(started).state).toBe("interrupted");
		provider.requests[0]!.finish();
		await until(() => job(started).running === 0);
		const reopened = await SessionManager.open(file, dir.path(), undefined, { suppressBreadcrumb: true });
		managers.push(reopened);
		const resumed = createSession(reopened, true);
		await new Promise<void>(resolve => setImmediate(resolve));
		expect(resumed.getMessageClassificationStatus()).toEqual({ jobs: [], rows: [] });
		expect(provider.requests).toHaveLength(1);
		expect(await facts(reopened)).toEqual(new Map([[saved, 3]]));
		const resumedJob = await resumed.startMessageClassificationBackfill(1);
		await until(() => provider.requests.length === 2);
		provider.requests[1]!.finish();
		await until(() => resumed.getMessageClassificationStatus().jobs.find(job => job.id === resumedJob)?.state === "completed");
		expect(await facts(reopened)).toEqual(new Map([[saved, 3], [pending, 0]]));
	});

	it("exposes an unavailable model rather than silently accepting missing classifications", async () => {
		const id = user("synthetic unavailable model source");
		session.settings.override("compaction.keepUserMessagesLlmModel", "nonexistent-provider/nonexistent-classifier");
		const availability = await session.getMessageClassificationAvailability();
		expect(availability.available).toBe(false);
		expect(typeof availability.reason).toBe("string");
		await expect(session.startMessageClassification(id)).rejects.toThrow();
		expect(provider.requests).toEqual([]);
		expect((await facts()).has(id)).toBe(false);
	});
});
