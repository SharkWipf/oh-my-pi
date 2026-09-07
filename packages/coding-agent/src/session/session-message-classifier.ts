import type { StreamFn } from "@oh-my-pi/pi-agent-core";
import type { Context, Model, SimpleStreamOptions } from "@oh-my-pi/pi-ai";
import { logger, untilAborted } from "@oh-my-pi/pi-utils";
import type { ModelRegistry } from "../config/model-registry";
import { resolveModelOverride } from "../config/model-resolver";
import type { Settings } from "../config/settings";
import {
	buildPreservedUserMessageClassifierInputFromLookup,
	classifyPreservedUserMessage,
	preservedUserMessageClassifierInputsEqual,
	type PreservedUserMessageClassifierInput,
} from "./preserve-user-messages-classifier";
import { packPreservedUserMessageClassifications, USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE } from "./preserved-message-settings";
import type { SessionEntry } from "./session-entries";
import type { SessionManager } from "./session-manager";

export interface MessageClassificationAvailability {
	available: boolean;
	model?: string;
	reason?: string;
}
export interface MessageClassificationRowStatus {
	entryId: string;
	jobId: string;
	state: "queued" | "running" | "saved" | "failed" | "canceled" | "interrupted";
	error?: string;
}
export interface MessageClassificationJobStatus {
	id: string;
	kind: "live" | "selected" | "backfill";
	state: "running" | "completed" | "failed" | "canceled" | "interrupted";
	sessionId: string;
	anchorId: string | null;
	model: string;
	workers: number;
	queued: number;
	running: number;
	completed: number;
	saved: number;
	failed: number;
	error?: string;
}
export interface MessageClassificationStatus {
	jobs: readonly MessageClassificationJobStatus[];
	rows: readonly MessageClassificationRowStatus[];
}
export type MessageClassificationListener = (status: MessageClassificationStatus, affectedIds: readonly string[]) => void;
export interface SessionMessageClassifierHost {
	sessionManager: SessionManager;
	settings: Settings;
	modelRegistry: ModelRegistry;
	generation(): number;
	isDisposed(): boolean;
	isStreaming(): boolean;
	sideStreamFn: StreamFn;
	prepareOptions(options: SimpleStreamOptions, model: Model): SimpleStreamOptions;
	obfuscate(context: Context): Context;
	readMasks(entries: readonly SessionEntry[], isCurrent: () => boolean): Promise<ReadonlyMap<string, number> | undefined>;
}
interface Job {
	targetId?: string;
	startPromise?: Promise<string>;
	epoch: number;
	validationLeafId: string | null;
	status: MessageClassificationJobStatus;
	generation: number;
	boundaryId: string | null;
	controller: AbortController;
	model: Model;
	iterator: Iterator<ClassifierTarget> | AsyncIterator<ClassifierTarget>;
	sharedResults: Set<Promise<void>>;
	finishedScanning: boolean;
}
interface Work {
	job: Job;
	entryId: string;
	controller: AbortController;
	input: PreservedUserMessageClassifierInput;
	completion: { promise: Promise<boolean>; resolve: (saved: boolean) => void };
}
interface ClassifierTarget { entryId: string; input: PreservedUserMessageClassifierInput }
interface ClassifierScope { entries: SessionEntry[]; boundaryId: string | null; anchorId: string | null; sessionId: string; generation: number; epoch: number }

/** Ephemeral side work owned by AgentSession; successful category facts belong to SessionManager. */
export class SessionMessageClassifier {
	readonly #host: SessionMessageClassifierHost;
	readonly #jobs = new Map<string, Job>();
	readonly #rows = new Map<string, MessageClassificationRowStatus>();
	readonly #pending = new Map<string, Work>();
	readonly #starts = new Map<string, Promise<string>>();
	readonly #listeners = new Set<MessageClassificationListener>();
	readonly #live = new Set<string>();
	#liveActive = false;
	#epoch = 0;
	#paused: { promise: Promise<void>; resolve: () => void } | undefined;

	constructor(host: SessionMessageClassifierHost) { this.#host = host; }
	getStatus(options?: { includeRows?: boolean }): MessageClassificationStatus {
		return {
			jobs: Array.from(this.#jobs.values(), job => ({ ...job.status })),
			rows: options?.includeRows === false ? [] : Array.from(this.#rows.values(), row => ({ ...row })),
		};
	}
	getRowStatus(id: string): MessageClassificationRowStatus | undefined {
		const row = this.#rows.get(id);
		return row ? { ...row } : undefined;
	}
	subscribe(listener: MessageClassificationListener): () => void {
		this.#listeners.add(listener);
		return () => this.#listeners.delete(listener);
	}
	#notify(ids: readonly string[] = []): void {
		if (!this.#listeners.size) return;
		const status: MessageClassificationStatus = {
			jobs: Array.from(this.#jobs.values(), job => ({ ...job.status })),
			rows: ids.flatMap(id => { const row = this.#rows.get(id); return row ? [{ ...row }] : []; }),
		};
		for (const listener of this.#listeners) {
			try { listener(status, ids); }
			catch (error) { logger.error("Classifier status listener failed", { error: String(error) }); }
		}
	}
	#resolveModel(): Model {
		const selector = this.#host.settings.get("compaction.keepUserMessagesLlmModel") || "@tiny";
		const { model } = resolveModelOverride([selector], this.#host.modelRegistry, this.#host.settings);
		if (!model) throw new Error(`Classifier model ${selector} is unavailable. Configure a supported classifier model.`);
		return model;
	}
	async getAvailability(): Promise<MessageClassificationAvailability> {
		try {
			const model = this.#resolveModel();
			const key = await this.#host.modelRegistry.getApiKey(model, this.#host.sessionManager.getSessionId());
			if (!key) throw new Error(`Classifier credentials unavailable for ${model.provider}. Configure Model.`);
			return { available: true, model: `${model.provider}/${model.id}` };
		} catch (error) { return { available: false, reason: String(error instanceof Error ? error.message : error) }; }
	}
	async #scope(targetId?: string): Promise<ClassifierScope> {
		const manager = this.#host.sessionManager;
		const scope: ClassifierScope = { entries: [], boundaryId: null, anchorId: manager.getLeafId(),
			sessionId: manager.getSessionId(), generation: this.#host.generation(), epoch: this.#epoch };
		let cursor = scope.anchorId;
		let deadline = performance.now() + 4;
		while (cursor !== null) {
			if (performance.now() >= deadline) {
				await new Promise<void>(resolve => setImmediate(resolve));
				if (scope.epoch !== this.#epoch || scope.sessionId !== manager.getSessionId() || scope.generation !== this.#host.generation()) throw new Error("Classifier source scan interrupted.");
				deadline = performance.now() + 4;
			}
			const entry = manager.getEntry(cursor);
			if (!entry) throw new Error("Classifier source ancestry is unavailable.");
			if (entry.type === "reset_boundary") { scope.boundaryId = entry.id; break; }
			if (targetId === undefined) scope.entries.push(entry);
			else if (entry.id === targetId) { scope.entries.push(entry); break; }
			cursor = entry.parentId;
		}
		if (targetId !== undefined && scope.entries.length === 0) throw new Error("Classifier target is not in the active post-clear branch.");
		if (targetId === undefined) scope.entries.reverse();
		return scope;
	}
	#input(entryId: string): PreservedUserMessageClassifierInput | undefined {
		return buildPreservedUserMessageClassifierInputFromLookup(entryId, id => this.#host.sessionManager.getEntry(id));
	}
	#current(job: Job): boolean {
		if (job.epoch !== this.#epoch || job.controller.signal.aborted || this.#host.isDisposed() || job.generation !== this.#host.generation() ||
			job.status.sessionId !== this.#host.sessionManager.getSessionId()) return false;
		const leafId = this.#host.sessionManager.getLeafId();
		let cursor = leafId;
		while (cursor !== job.validationLeafId) {
			if (cursor === null) return false;
			const entry = this.#host.sessionManager.getEntry(cursor);
			if (!entry || entry.type === "reset_boundary") return false;
			cursor = entry.parentId;
		}
		job.validationLeafId = leafId;
		return true;
	}
	async #ready(job: Job): Promise<boolean> {
		if (this.#paused && !job.controller.signal.aborted) {
			const paused = this.#paused.promise;
			await new Promise<void>(resolve => {
				const abort = () => resolve();
				job.controller.signal.addEventListener("abort", abort, { once: true });
				void paused.then(() => { job.controller.signal.removeEventListener("abort", abort); resolve(); });
			});
		}
		if (this.#current(job)) return true;
		this.#cancel(job, "interrupted", "Session, branch, reset, or source scope changed; resume missing classifications explicitly.");
		return false;
	}

	start(entryId: string, kind: "selected" | "live" = "selected"): Promise<string> {
		const starting = this.#starts.get(entryId);
		if (starting) return starting;
		const existing = this.#pending.get(entryId);
		if (existing && this.#current(existing.job)) return Promise.resolve(existing.job.status.id);
		const start = (async () => {
			const scope = await this.#scope(entryId);
			const input = this.#input(entryId);
			if (!input) throw new Error("Only real user source messages can be classified.");
			return this.#start(kind, 1, scope, () => [{ entryId, input }][Symbol.iterator](), entryId);
		})();
		this.#starts.set(entryId, start);
		void start.catch(() => { if (this.#starts.get(entryId) === start) this.#starts.delete(entryId); });
		return start;
	}
	async startBackfill(workers: number): Promise<string> {
		if (!Number.isSafeInteger(workers) || workers <= 0) throw new Error("Classifier workers must be a positive integer.");
		const scope = await this.#scope();
		const host = this.#host;
		async function* missing(isCurrent: () => boolean): AsyncGenerator<ClassifierTarget> {
			const masks = await host.readMasks(scope.entries, isCurrent);
			if (!masks || !isCurrent()) return;
			let deadline = performance.now() + 4;
			for (const entry of scope.entries) {
				if (performance.now() >= deadline) {
					await new Promise<void>(resolve => setImmediate(resolve));
					if (!isCurrent()) return;
					deadline = performance.now() + 4;
				}
				if (entry.type !== "message" || entry.message.role !== "user" || masks.has(entry.id)) continue;
				const input = buildPreservedUserMessageClassifierInputFromLookup(entry.id, id => host.sessionManager.getEntry(id));
				if (input) yield { entryId: entry.id, input };
			}
		}
		return this.#start("backfill", workers, scope, missing);
	}
	async #start(kind: Job["status"]["kind"], workers: number, scope: ClassifierScope, createIterator: (isCurrent: () => boolean) => Job["iterator"], targetId?: string): Promise<string> {
		if (this.#host.isDisposed()) throw new Error("Session disposed.");
		const model = this.#resolveModel();
		const job: Job = {
			targetId,
			startPromise: targetId ? this.#starts.get(targetId) : undefined,
			sharedResults: new Set(),
			validationLeafId: scope.anchorId, epoch: scope.epoch,
			status: { id: Bun.randomUUIDv7(), kind, state: "running", sessionId: scope.sessionId,
				anchorId: scope.anchorId, model: `${model.provider}/${model.id}`, workers,
				queued: 0, running: 0, completed: 0, saved: 0, failed: 0 },
			generation: scope.generation, boundaryId: scope.boundaryId, controller: new AbortController(), model, iterator: createIterator(() => this.#current(job)), finishedScanning: false,
		};
		const apiKey = await this.#host.modelRegistry.getApiKey(model, job.status.sessionId, { signal: job.controller.signal });
		if (!this.#current(job)) throw new Error("Classifier launch interrupted by a session or branch transition.");
		if (!apiKey) throw new Error(`Classifier credentials unavailable for ${model.provider}. Configure Model.`);
		// A new explicit/live attempt supersedes the prior settled status, not its durable facts.
		for (const [id, previous] of this.#jobs) {
			if (previous.status.kind === kind && previous.status.state !== "running") this.#jobs.delete(id);
		}
		this.#jobs.set(job.status.id, job);
		this.#notify();
		void this.#launch(job);
		return job.status.id;
	}
	async #launch(job: Job): Promise<void> {
		try {
			await this.#host.sessionManager.ensureOnDisk();
			if (!(await this.#ready(job))) return;
			await this.#host.sessionManager.flush();
			if (!(await this.#ready(job))) return;
			// A worker owns at most one projected message. The scanner never expands the entire history into requests.
			const workers: Promise<void>[] = [];
			for (let i = 0; i < job.status.workers; i++) {
				const work = await this.#next(job);
				if (!work) break;
				workers.push(this.#worker(job, work));
			}
			await Promise.all(workers);
			await Promise.all(job.sharedResults);
			if (job.status.state === "running") job.status.state = job.status.failed ? "failed" : "completed";
		} catch (error) {
			if (job.status.state === "running") { job.status.state = "failed"; job.status.error = String(error instanceof Error ? error.message : error); }
		} finally {
			if (job.targetId && this.#starts.get(job.targetId) === job.startPromise) this.#starts.delete(job.targetId);
			job.startPromise = undefined;
			if (job.status.kind === "live") { this.#liveActive = false; this.drainLive(); }
			this.#notify();
		}
	}
	async #next(job: Job): Promise<Work | undefined> {
		if (job.controller.signal.aborted || job.finishedScanning) return undefined;
		for (;;) {
			const next = await job.iterator.next();
			if (!(await this.#ready(job))) return undefined;
			if (!this.#current(job)) { this.#cancel(job, "interrupted", "Classifier source scope changed before queueing."); return undefined; }
			if (next.done) { job.finishedScanning = true; return undefined; }
			const shared = this.#pending.get(next.value.entryId);
			if (shared) {
				job.status.queued++;
				const observed = untilAborted(job.controller.signal, shared.completion.promise).then(saved => {
					if (!this.#current(job)) return;
					if (saved && preservedUserMessageClassifierInputsEqual(shared.input, this.#input(shared.entryId))) job.status.saved++;
					else job.status.failed++;
				}).catch(() => { if (!job.controller.signal.aborted) job.status.failed++; }).finally(() => {
					job.sharedResults.delete(observed);
					job.status.queued--; job.status.completed++; this.#notify();
				});
				job.sharedResults.add(observed);
				continue;
			}
			const work: Work = { job, ...next.value, controller: new AbortController(), completion: Promise.withResolvers<boolean>() };
			this.#pending.set(work.entryId, work);
			job.status.queued++;
			this.#rows.set(work.entryId, { entryId: work.entryId, jobId: job.status.id, state: "queued" });
			this.#notify([work.entryId]);
			return work;
		}
	}
	async #worker(job: Job, initial: Work): Promise<void> {
		let work: Work | undefined = initial;
		while (work && await this.#ready(job)) {
			await this.#run(work);
			if (!(await this.#ready(job))) break;
			// Yield to input/UI even when the provider completes synchronously.
			await new Promise<void>(resolve => setImmediate(resolve));
			if (!(await this.#ready(job))) break;
			work = await this.#next(job);
		}
	}
	async #run(work: Work): Promise<void> {
		const { job, entryId, input } = work;
		const row = this.#rows.get(entryId)!;
		const signal = AbortSignal.any([job.controller.signal, work.controller.signal]);
		job.status.queued--; job.status.running++;
		row.state = "running";
		this.#notify([entryId]);
		try {
			signal.throwIfAborted();
			if (!preservedUserMessageClassifierInputsEqual(input, this.#input(entryId))) throw new Error("Classifier source/context changed; retry with current input.");
			const requestSessionId = Bun.randomUUIDv7();
			const mask = await classifyPreservedUserMessage(input, {
				model: job.model,
				complete: async (context, maxTokens) => {
					if (!(await this.#ready(job))) throw new Error("Classifier interrupted.");
					signal.throwIfAborted();
					if (!this.#current(job) || this.#paused) throw new Error("Classifier ownership changed before request.");
					const options = this.#host.prepareOptions({
						apiKey: this.#host.modelRegistry.resolver(job.model, job.status.sessionId),
						sessionId: requestSessionId, promptCacheKey: `${job.status.sessionId}:message-classifier`,
						preferWebsockets: false, initiatorOverride: "agent", signal, maxTokens,
					}, job.model);
					const stream = await untilAborted(signal, async () => this.#host.sideStreamFn(job.model, this.#host.obfuscate(context), options));
					return untilAborted(signal, stream.result());
				},
			});
			if (!(await this.#ready(job))) return;
			signal.throwIfAborted();
			if (!preservedUserMessageClassifierInputsEqual(input, this.#input(entryId))) throw new Error("Classifier source/context changed; retry with current input.");
			if (!this.#current(job) || this.#paused) throw new Error("Classifier ownership changed before publication.");
			// No await between final owner/input validation and append: never write onto an off-branch resume leaf.
			this.#host.sessionManager.appendCustomEntry(USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE, packPreservedUserMessageClassifications([{ id: entryId, mask }]));
			await this.#host.sessionManager.flush();
			if (!(await this.#ready(job))) return;
			if (!this.#current(job) || this.#paused || !preservedUserMessageClassifierInputsEqual(input, this.#input(entryId))) { row.state = "interrupted"; return; }
			row.state = "saved";
			job.status.saved++;
		} catch (error) {
			if (!signal.aborted) {
				row.state = "failed";
				row.error = String(error instanceof Error ? error.message : error);
				job.status.failed++;
			} else if (!job.controller.signal.aborted) {
				row.state = "interrupted"; job.status.failed++;
			}
		} finally {
			job.status.running--; job.status.completed++;
			work.completion.resolve(row.state === "saved");
			if (this.#pending.get(entryId) === work) this.#pending.delete(entryId);
			this.#notify([entryId]);
			if (row.state === "saved" && this.#rows.get(entryId) === row) this.#rows.delete(entryId);
		}
	}
	cancel(id: string): void { const job = this.#jobs.get(id); if (job) this.#cancel(job, "canceled", "Canceled; saved facts remain active. Resume missing classifications explicitly."); }
	#cancel(job: Job, state: "canceled" | "interrupted", reason: string): void {
		if (job.status.state !== "running") return;
		job.status.state = state; job.status.error = reason;
		job.controller.abort(new Error(reason));
		if (job.targetId && this.#starts.get(job.targetId) === job.startPromise) this.#starts.delete(job.targetId);
		void job.iterator.return?.(); job.finishedScanning = true;
		const ids: string[] = [];
		for (const [id, work] of this.#pending) {
			if (work.job !== job) continue;
			const row = this.#rows.get(id)!;
			if (row.state === "queued") job.status.queued--;
			row.state = state; row.error = reason;
			work.completion.resolve(false);
			this.#pending.delete(id); ids.push(id);
		}
		this.#notify(ids);
	}
	pause(): void { this.#paused ??= Promise.withResolvers<void>(); }
	resume(): void {
		const paused = this.#paused; this.#paused = undefined; paused?.resolve();
		for (const job of this.#jobs.values()) if (!this.#current(job)) this.#cancel(job, "interrupted", "Session or branch changed; resume missing classifications explicitly.");
		this.drainLive();
	}
	interrupt(reason: string): void {
		this.#epoch++;
		this.#live.clear();
		this.#starts.clear();
		for (const job of this.#jobs.values()) this.#cancel(job, "interrupted", reason);
	}
	interruptInputs(ids: readonly string[]): void {
		for (const id of ids) {
			this.#starts.delete(id);
			const work = this.#pending.get(id);
			if (!work) continue;
			work.controller.abort(new Error("Classifier source/context changed; retry with current input."));
			const row = this.#rows.get(id)!;
			row.state = "interrupted"; row.error = "Source/context changed; explicit retry required.";
			work.completion.resolve(false);
			this.#pending.delete(id);
		}
		this.#notify(ids);
	}
	dispose(): void { this.interrupt("Session disposed; restart requires explicit missing-only backfill."); this.#paused?.resolve(); this.#paused = undefined; this.#listeners.clear(); }
	enqueueLive(entryId: string): void {
		if (!this.#host.settings.get("compaction.keepUserMessages") || !this.#host.settings.get("compaction.keepUserMessagesLlm")) return;
		this.#live.add(entryId);
		this.drainLive();
	}
	drainLive(): void {
		if (this.#liveActive || this.#paused || this.#host.isDisposed() || this.#host.isStreaming()) return;
		const next = this.#live.values().next();
		if (next.done) return;
		const entryId = next.value;
		this.#live.delete(entryId);
		this.#liveActive = true;
		void this.start(entryId, "live").then(jobId => {
			if (this.#jobs.get(jobId)?.status.kind !== "live") { this.#liveActive = false; this.drainLive(); }
		}).catch(error => {
			this.#rows.set(entryId, { entryId, jobId: "", state: "failed", error: String(error instanceof Error ? error.message : error) });
			this.#notify([entryId]);
			this.#liveActive = false; this.drainLive();
		});
	}
}
