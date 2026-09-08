import { afterEach, describe, expect, it } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { CompactionCancelledError } from "@oh-my-pi/pi-agent-core/compaction";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { getOriginalSourceMessage } from "../src/session/messages";
import { SessionMaintenance, type SessionMaintenanceHost } from "../src/session/session-maintenance";
import { SessionManager } from "../src/session/session-manager";
import { MemorySessionStorage, type SessionStorageWriter } from "../src/session/session-storage";

class PublicationStorage extends MemorySessionStorage {
	gate: PromiseWithResolvers<void> | undefined;
	entered = Promise.withResolvers<void>();
	failCompaction = false;
	skipDrains = 0;

	override openWriter(...args: Parameters<MemorySessionStorage["openWriter"]>): SessionStorageWriter {
		const writer = super.openWriter(...args);
		return {
			append: line => writer.append(line),
			appendSync: line => {
				if (this.failCompaction && JSON.parse(line).type === "compaction") {
					this.failCompaction = false;
					throw new Error("compaction storage unavailable");
				}
				writer.appendSync!(line);
			},
			flush: () => writer.flush(),
			flushSync: () => writer.flushSync?.(),
			isOpen: () => writer.isOpen(),
			close: () => writer.close(),
			getError: () => writer.getError(),
		};
	}

	override async drain(): Promise<void> {
		if (this.skipDrains > 0) this.skipDrains--;
		else if (this.gate) {
			const gate = this.gate;
			this.entered.resolve();
			await gate.promise;
			if (this.gate === gate) this.gate = undefined;
		}
		await super.drain();
	}

	deferPublication(): void {
		// Each fixture gates one publication or one explicit recovery.
		this.gate = Promise.withResolvers<void>();
	}
}



const managers: SessionManager[] = [];
afterEach(async () => {
	await Promise.all(managers.splice(0).map(manager => manager.close().catch(() => undefined)));
});

function fixture() {
	const storage = new PublicationStorage();
	const manager = SessionManager.create("/compaction-publication", "/compaction-publication/sessions", storage);
	managers.push(manager);
	const model = getBundledModel("anthropic", "claude-sonnet-4-5")!;
	const first = manager.appendMessage({ role: "user", content: "old source ".repeat(2_000), timestamp: 1 });
	manager.appendMessage({ role: "user", content: "retained source", timestamp: 2 });
	const agent = new Agent({ initialState: { model, messages: manager.buildSessionContext().messages, tools: [] } });
	const settings = Settings.isolated({ "compaction.keepRecentTokens": 1, "compaction.methodOrder": ["soft"], "requirements.enabled": false });

	let ownership: unknown = {};
	let policy: unknown = {};
	let generate = async () => ({ document: "frozen handoff summary" });
	const effects: string[] = [];
	const host = {
		agent, sessionManager: manager, settings,
		model: () => model,
		sessionId: () => manager.getSessionId(),
		compactionOwnership: () => ownership,
		compactionPolicyIdentity: () => policy,
		compactionSourceSelection: async () => ({ originalSourceMessage: getOriginalSourceMessage }),

		isDisposed: () => false,
		isGeneratingHandoff: () => false,
		generateHandoffDocument: () => generate(),
		buildDisplaySessionContext: () => manager.buildSessionContext(),
		nonMessageTokenSource: () => ({}),
		rebaseAfterCompaction: () => effects.push("rebase"),
		resetPlanReference: () => effects.push("plan"),
		resetAdvisorRuntimes: () => effects.push("advisor"),
		syncTodoPhasesFromBranch: () => effects.push("todo"),
		closeCodexProviderSessionsForHistoryRewrite: () => effects.push("provider"),
		extensionRunner: { emit: async () => effects.push("session_compact") },
	} as unknown as SessionMaintenanceHost;
	const maintenance = new SessionMaintenance(host);
	return {
		storage, manager, agent, maintenance, effects, first,
		setGenerate: (fn: typeof generate) => { generate = fn; },
		changePolicy: () => { policy = {}; },
		switchBranch: () => {
			ownership = {};
			manager.branch(first);
			manager.appendMessage({ role: "user", content: "branch B", timestamp: 3 });
			agent.replaceMessages(manager.buildSessionContext().messages);
		},
		clear: () => {
			ownership = {};
			manager.appendResetBoundary();
			manager.appendMessage({ role: "user", content: "after clear", timestamp: 4 });
			agent.replaceMessages(manager.buildSessionContext().messages);
		},
	};
}

function compactions(manager: SessionManager) {
	return manager.getEntries().filter(entry => entry.type === "compaction");
}

describe("compaction durable publication", () => {
	it("persists source IDs before generation, waits for publication, and keeps policy-only changes plus a valid suffix", async () => {
		const f = fixture();
		const original = f.agent.state.messages;
		f.setGenerate(async () => {
			const persisted = await f.storage.readText(f.manager.getSessionFile()!);
			expect(persisted).toContain(f.first);
			f.storage.deferPublication();
			return { document: "frozen handoff summary" };
		});
		const run = f.maintenance.handoff();
		await f.storage.entered.promise;
		expect(compactions(f.manager)).toHaveLength(1);
		expect(f.agent.state.messages).toBe(original);
		expect(f.effects).toEqual([]);
		f.changePolicy();
		f.manager.appendMessage({ role: "user", content: "valid appended suffix", timestamp: 3 });
		f.storage.gate!.resolve();
		expect(await run).toEqual({ document: "frozen handoff summary" });
		expect(f.agent.state.messages.at(-1)).toMatchObject({ role: "user", content: "valid appended suffix" });
		expect(f.effects).toEqual(["rebase", "plan", "advisor", "todo", "provider", "session_compact"]);
		const reopened = await SessionManager.open(f.manager.getSessionFile()!, undefined, f.storage);
		managers.push(reopened);
		expect(compactions(reopened).map(entry => entry.id)).toEqual(compactions(f.manager).map(entry => entry.id));
		expect(reopened.buildSessionContext().messages).toEqual(f.agent.state.messages);
	});

	for (const transition of ["switchBranch", "clear"] as const) {
		it(`never installs branch A after ${transition} while publication is pending`, async () => {
			const f = fixture();
			f.setGenerate(async () => {
				f.storage.deferPublication();
				return { document: "A summary" };
			});
			const run = f.maintenance.handoff();
			await f.storage.entered.promise;
			const committedId = compactions(f.manager)[0].id;
			f[transition]();
			const otherContext = f.agent.state.messages;
			f.storage.gate!.resolve();
			expect(await run).toBeUndefined();
			expect(f.agent.state.messages).toBe(otherContext);
			expect(f.effects).toEqual([]);
			expect(compactions(f.manager).map(entry => entry.id)).toEqual([committedId]);
		});
	}

	it("recovers the same retained event after a nonthrowing append failure without regenerating", async () => {
		const f = fixture();
		let generations = 0;
		f.setGenerate(async () => {
			generations++;
			f.storage.failCompaction = true;
			return { document: "retained failed-write result" };
		});
		const original = f.agent.state.messages;
		await expect(f.maintenance.handoff()).rejects.toThrow("compaction storage unavailable");
		const entry = compactions(f.manager)[0];
		expect(f.agent.state.messages).toBe(original);
		expect(f.effects).toEqual([]);
		f.changePolicy();
		expect(await f.maintenance.recoverCompactionPersistence()).toBe(true);
		expect(generations).toBe(1);
		expect(compactions(f.manager).map(value => value.id)).toEqual([entry.id]);
		expect(f.effects.filter(value => value === "session_compact")).toEqual(["session_compact"]);
		expect(await f.maintenance.recoverCompactionPersistence()).toBe(false);
		const reopened = await SessionManager.open(f.manager.getSessionFile()!, undefined, f.storage);
		managers.push(reopened);
		expect(compactions(reopened).map(value => [value.id, value.summary])).toEqual([[entry.id, entry.summary]]);
	});

	it("suppresses origin installation after a branch switch during explicit persistence recovery", async () => {
		const f = fixture();
		f.setGenerate(async () => {
			f.storage.failCompaction = true;
			return { document: "origin recovery" };
		});
		await expect(f.maintenance.handoff()).rejects.toThrow("compaction storage unavailable");
		const committedId = compactions(f.manager)[0].id;
		f.storage.deferPublication();
		f.storage.skipDrains = 1; // Let the repair close/drain finish; gate its required final flush.
		const recovery = f.maintenance.recoverCompactionPersistence();
		await f.storage.entered.promise;
		f.switchBranch();
		const otherContext = f.agent.state.messages;
		f.storage.gate!.resolve();
		expect(await recovery).toBe(false);
		expect(f.agent.state.messages).toBe(otherContext);
		expect(f.effects).toEqual([]);
		expect(compactions(f.manager).map(entry => entry.id)).toEqual([committedId]);
	});

	it("rejects policy or in-place source changes during generation before publishing any event", async () => {
		for (const mutate of ["policy", "source"] as const) {
			const f = fixture();
			f.setGenerate(async () => {
				if (mutate === "policy") f.changePolicy();
				else {
					const entry = f.manager.getEntry(f.first)!;
					if (entry.type === "message" && entry.message.role === "user") entry.message.content = "rewritten source";
				}
				return { document: "stale result" };
			});
			await expect(f.maintenance.handoff()).rejects.toBeInstanceOf(CompactionCancelledError);
			expect(compactions(f.manager)).toEqual([]);
			expect(f.effects).toEqual([]);
		}
	});
});
