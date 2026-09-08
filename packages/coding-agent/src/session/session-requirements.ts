import { randomUUID } from "node:crypto";
import type { Context, Model } from "@oh-my-pi/pi-ai";
import { prompt } from "@oh-my-pi/pi-utils";
import type { ModelRegistry } from "../config/model-registry";
import type { Settings } from "../config/settings";
import recallTemplate from "../prompts/requirements-recall.md" with { type: "text" };
import {
	admitRequirementsCandidates,
	createRequirementsBatch,
	extractRequirementsBatch,
	REQUIREMENTS_FORMAT,
	type RequirementsEvidencePackage,
	requirementsModelStatus,
	RequirementsPipelineError,
	type RequirementsModelStatus,
	requirementsReviewAccepted,
	reviewRequirementsCandidates,
	verifyRequirementsEvidence,
} from "../requirements/pipeline";
import {
	type ResolvedRequirementsSource,
	resolveRequirementsSource,
} from "../requirements/source-capture";
import { RequirementsStore } from "../requirements/store";
import type {
	RequirementsApplicable,
	RequirementsAuthority,
	RequirementsBatch,
	RequirementsCoverageSummary,
	RequirementsReview,
	RequirementsScope,
	RequirementsSnapshot,
	RequirementsSource,
} from "../requirements/types";
import type { AgentStorage } from "./agent-storage";
import type { SessionManager } from "./session-manager";

export interface SessionRequirementsHost {
	sessionManager: SessionManager;
	agentStorage: AgentStorage | null;
	settings: Settings;
	modelRegistry: ModelRegistry;
	getContext(): Context;
	getModel(): Model | undefined;
	/** Record a durable, host-attested operator decision; NEVER invoke the primary model. */
	promptOperatorSource(text: string, targetRevisionIds?: string[], options?: { literal?: boolean }): Promise<string>;
	isDisposed(): boolean;
}
export type RequirementsRecoveryMode = "off" | "bypass" | "clean";
export interface RequirementsCallReceipt {
	revisionIds: string[];
	publicationRevision: number;
	generation: number;
	coverageComplete: boolean;
	bypass: RequirementsRecoveryMode;
	tokens: number | null;
	tokenProvenance: "estimated" | "unknown";
	signature: string;
	phase?: "prepared" | "sent" | "refused";
	capacity?: {
		contextWindow: number;
		reserveTokens: number;
		irreducibleTokens: number | null;
		provenance: "estimated" | "unknown";
	};
}
export interface RequirementsApplicableSnapshot extends RequirementsApplicable {
	/** Entire durable ledger; not branch applicability or proof of freshly verified cold bytes. */
	ledgerCoverage: RequirementsCoverageSummary;
	pendingSources: RequirementsSource[];
	publicationRevision: number;
	generation: number;
	enabled: boolean;
	bypass: RequirementsRecoveryMode;
	signature: string;
}
export interface RequirementsStatus {
	enabled: boolean;
	bypass: RequirementsRecoveryMode;
	running: boolean;
	sourceCatalog: "unobserved" | "observing" | "last-observed";
	models: Record<"extractor" | "evidence" | "sanity", RequirementsModelStatus>;
	snapshot: RequirementsSnapshot;
	applicable: RequirementsApplicableSnapshot;
	activeTokens: number | null;
	tokenProvenance: "estimated" | "unknown";
	receipt?: RequirementsCallReceipt;
	error?: string;
}
export type RequirementsOperatorAction =
	| { kind: "inspect"; revisionId: string }
	| { kind: "retry"; sourceKey?: string }
	| { kind: "evidence"; sourceKey: string; referentKeys: string[] }
	| { kind: "gap"; sourceKey: string; reason: string }
	| { kind: "quarantine"; revisionIds: string[]; reason: string }
	| { kind: "restore"; revisionIds: string[]; literalUnitId?: string }
	| { kind: "clear"; scope: "session" | "project" | "global" | "all"; reason: string }
	| { kind: "decision"; text: string; targetRevisionIds?: string[] }
	| { kind: "literal-adopt"; sourceKey: string; unitId: string; scope: RequirementsScope };

export function composeProviderRequirements(
	context: Context,
	snapshot: RequirementsApplicableSnapshot,
	countTokens?: (text: string) => number,
): { context: Context; receipt: RequirementsCallReceipt } {
	const enabled = snapshot.enabled && snapshot.bypass === "off";
	const active = enabled ? snapshot.active : [];
	const ledgerHoles =
		snapshot.ledgerCoverage.total - snapshot.ledgerCoverage.referenceOnly - snapshot.ledgerCoverage.byState.complete;
	const text =
		enabled && (active.length || snapshot.conflicts.length || ledgerHoles)
			? prompt.render(recallTemplate, {
				memory: {
					requirements: active.map(revision => ({
						id: revision.requirementId,
						revision: revision.id,
						statement: revision.statement,
						scope: revision.scope,
						source: revision.sourceKey,
					})),
					conflicts: snapshot.conflicts.map(group =>
						group.map(revision => ({
							id: revision.id,
							source: revision.sourceKey,
							lifecycle: revision.lifecycle,
							kind: revision.kind,
							...(revision.lifecycle === "accepted" && revision.kind !== "withdraw"
								? { statement: revision.statement }
								: {}),
						})),
					),
					coverage: {
						scope: "entire durable requirements ledger; not branch-local coverage",
						...snapshot.ledgerCoverage,
						originals: "Cold source availability is last-observed; consumed requirement evidence is freshly checked",
						ledger: "/memory requirements coverage",
						source: "/memory requirements source <source-key>",
						retry: "/memory requirements retry [source-key]",
					},
					pendingSources: snapshot.pendingSources.map(source => ({
						source: source.key,
						state: source.state,
						reason: source.reason,
					})),
				},
			})
			: "";
	const base = context.systemPrompt ?? [];
	const parts = text ? [...base, text] : base;
	const estimated = countTokens ? countTokens(text) : undefined;
	const tokens = typeof estimated === "number" && Number.isFinite(estimated) && estimated >= 0 ? estimated : null;
	return {
		context: { ...context, systemPrompt: parts },
		receipt: {
			revisionIds: active.map(revision => revision.id),
			publicationRevision: snapshot.publicationRevision,
			generation: snapshot.generation,
			coverageComplete: enabled && ledgerHoles === 0 && snapshot.conflicts.length === 0,
			bypass: snapshot.bypass,
			tokens,
			tokenProvenance: tokens === null ? "unknown" : "estimated",
			signature: snapshot.signature,
			phase: "prepared",
		},
	};
}

/** Owns sources, jobs and operator actions independently of the ordinary memory backend and UI lifetime. */
export class SessionRequirements {
	readonly #volatile: RequirementsStore;
	readonly #unsubscribeSettings: () => void;
	#sourceKeys = new Set<string>();
	readonly #pendingLive = new Map<string, { entryId: string; integrity: string }>();
	#pendingGeneration = 0;
	#pendingSessionId: string | undefined;
	#pendingSnapshot: Readonly<{ sessionId: string; epoch: number; generation: number; entryIds: readonly string[] }> | undefined;
	#epoch = 0;
	#bypass: RequirementsRecoveryMode = "off";
	#controller = new AbortController();
	#authorizedOwnerSessionId: string | undefined;
	#run: Promise<void> | undefined;
	#disposed = false;
	#error: string | undefined;
	#receipt: RequirementsCallReceipt | undefined;
	#lifecycle = 0;
	#observing: Promise<void> | undefined;
	readonly #extraEvidence = new Map<string, string[]>();
	#hasObserved = false;
	#applicableCache: RequirementsApplicableSnapshot | undefined;
	#composition:
		| {
				base: string[] | undefined;
				signature: string;
				parts: string[];
				receipt: RequirementsCallReceipt;
				activeTokens: number | null;
				countTokens?: (text: string) => number;
		  }
		| undefined;
	#observedVersion: string | undefined;

	constructor(readonly host: SessionRequirementsHost) {
		this.#volatile = new RequirementsStore();
		this.#unsubscribeSettings = host.settings.onEffectiveChange((path, value) => {
			if (path !== "requirements.enabled") return;
			if (!value) { this.cancelPending("Requirements disabled"); this.releasePendingLive(); }
			else
				void this.observeCommittedSources().catch(error => {
					this.#error = error instanceof Error ? error.message : String(error);
				});
		});
	}
	get #storage(): AgentStorage | RequirementsStore {
		return this.host.sessionManager.getSessionFile() && this.host.agentStorage
			? this.host.agentStorage
			: this.#volatile;
	}
	#snapshot(): RequirementsSnapshot {
		return this.#storage.getRequirementsSnapshot();
	}
	#consumptionSnapshot() {
		this.#syncEpoch();
		const sessionId = this.host.sessionManager.getSessionId();
		return this.#storage.getRequirementsConsumptionSnapshot({
			projectId: this.host.sessionManager.getCwd(),
			sessionId,
			epoch: this.#epoch,
			branchId: `${sessionId}:${this.#epoch}:${this.#lifecycle}`,
		});
	}
	get #enabled(): boolean {
		return this.host.settings.get("requirements.enabled") && !this.#disposed && !this.host.isDisposed();
	}
	#authority(snapshot: { state: { generation: number } }): RequirementsAuthority {
		return {
			ownerSessionId: this.host.sessionManager.getSessionId(),
			branchId: `${this.host.sessionManager.getSessionId()}:${this.#epoch}:${this.#lifecycle}`,
			epoch: this.#epoch,
			generation: snapshot.state.generation,
		};
	}
	#assertAvailable(signal = this.#controller.signal): void {
		signal.throwIfAborted();
		if (!this.#enabled) throw new Error("Requirements disabled; accepted data remains inspectable");
		if (this.#bypass !== "off") throw new Error("Requirements publication suspended by memory recovery mode");
	}
	async observeCommittedSources(explicit = false): Promise<void> {
		if (this.#disposed || this.host.isDisposed() || (!explicit && (!this.#enabled || this.#bypass !== "off"))) return;
		if (this.#observing) {
			await this.#observing;
			return this.observeCommittedSources(explicit);
		}
		const manager = this.host.sessionManager;
		const version = manager.getRequirementsSourceVersion();
		let lifecycle = this.#lifecycle;
		const current = () =>
			!this.#disposed &&
			!this.host.isDisposed() &&
			lifecycle === this.#lifecycle &&
			version === manager.getRequirementsSourceVersion();
		const work = (async () => {
			// Yield before any catalog/snapshot work, including synchronous in-memory journals.
			await new Promise<void>(resolve => setImmediate(resolve));
			if (!current()) return;
			if (this.#observedVersion === version) {
				if (this.#enabled && this.#bypass === "off") await this.#reconcileApplicableIntegrity();
				return;
			}
			await manager.ensureOnDisk();
			await manager.flush();
			if (!current()) return;
			const keys = new Set<string>();
			const changed: RequirementsSource[] = [];
			let sliceStarted = performance.now();
			const intake = () => {
				if (!changed.length) return;
				this.#storage.intakeRequirementsSources(changed);
				changed.length = 0;
			};
			const iterator = manager.iterateRequirementsSources();
			let epoch = this.#epoch;
			try {
				while (true) {
					const next = await iterator.next();
					if (!current()) return;
					if (next.done) {
						epoch = next.value;
						break;
					}
					const { source } = next.value;
					keys.add(source.key);
					const prior = this.#storage.getRequirementsSourceMetadata(source.key);
					if (!prior) changed.push(source);
					else if (source.locators.some(locator => !prior.locators.some(current =>
						current.sessionId === locator.sessionId && current.entryId === locator.entryId && current.journalPath === locator.journalPath
					))) {
						const retained = this.#storage.getRequirementsSource(source.key)!;
						changed.push({ ...retained, locators: source.locators });
					}
					// A scheduling quantum, not a source count/retention limit: drain every descriptor.
					if (performance.now() - sliceStarted >= 8) {
						intake();
						await new Promise<void>(resolve => setImmediate(resolve));
						if (!current()) return;
						sliceStarted = performance.now();
					}
				}
			} finally {
				await iterator.return(epoch);
			}
			intake();
			if (!current()) return;
			let removed = false;
			for (const key of this.#sourceKeys) {
				if (!keys.has(key)) {
					removed = true;
					break;
				}
			}
			if (epoch !== this.#epoch || removed) {
				this.cancelPending("Active source ancestry or context epoch changed");
				lifecycle = this.#lifecycle;
				this.releasePendingLive();
			}
			this.#epoch = epoch;
			this.#sourceKeys = keys;
			// Catalog observations are not byte proof. Select fresh retained decision/job evidence
			// only after every catalog await, including publications from another SQLite owner.
			await this.#reconcileApplicableIntegrity();
			if (!current()) return;
			this.#hasObserved = true;
			this.#observedVersion = version;
		})();
		this.#observing = work;
		try {
			await work;
		} catch (error) {
			this.#error = String(error);
			throw error;
		} finally {
			if (this.#observing === work) this.#observing = undefined;
		}
		if (
			!this.#disposed &&
			!this.host.isDisposed() &&
			(explicit || (this.#enabled && this.#bypass === "off")) &&
			this.#observedVersion !== manager.getRequirementsSourceVersion()
		)
			return this.observeCommittedSources(explicit);
	}
	async #reconcileApplicableIntegrity(): Promise<void> {
		await this.#reconcileIntegrity();
	}
	#consumedEvidence(): RequirementsSource[] {
		// Fresh applicable/restorable heads and authorized jobs, including every actual evidence dependency.
		return this.#consumptionSnapshot().sources;
	}
	async #reconcileIntegrity(sources = this.#consumedEvidence()): Promise<Record<string, string>> {
		const observations = await this.host.sessionManager.observeRequirementsEvidence(sources);
		const verified: Record<string, string> = {};
		for (const observation of observations)
			if (observation.integrity) verified[observation.key] = observation.integrity;
		this.#storage.reconcileRequirementsSources(observations);
		return verified;
	}
	async #reconcileReviewedEvidence(input: RequirementsEvidencePackage, batch?: RequirementsBatch): Promise<Record<string, string>> {
		// Explicit restore/publication consumes its frozen input even when routine recall excludes its scope.
		const sources = new Map(this.#consumedEvidence().map(source => [source.key, source]));
		for (const resolved of [input.source, ...input.references]) sources.set(resolved.source.key, resolved.source);
		for (const key of Object.keys(batch?.readSourceIntegrities ?? {})) {
			const source = this.#storage.getRequirementsSource(key);
			if (source) sources.set(key, source);
		}
		return this.#reconcileIntegrity([...sources.values()]);
	}
	async inspectSource(sourceKey: string, context = false): Promise<ResolvedRequirementsSource> {
		let descriptor = this.#storage.getRequirementsSource(sourceKey);
		const resolved = await resolveRequirementsSource(this.host.sessionManager, sourceKey, descriptor, { context });
		if (!resolved || resolved.source.state === "orphaned") {
			if (descriptor) this.#storage.reconcileRequirementsSources([{ key: sourceKey, integrity: null }]);
			throw new Error(`Original requirements source unavailable: ${sourceKey}; derived records are not replacement evidence`);
		}
		if (descriptor && descriptor.units.length && descriptor.integrity !== resolved.source.integrity)
			await this.#reconcileIntegrity([descriptor]);
		if (descriptor?.authorityGeneration !== undefined) resolved.source.authorityGeneration = descriptor.authorityGeneration;
		this.#storage.intakeRequirementsSource(resolved.source);
		descriptor = this.#storage.getRequirementsSource(sourceKey);
		if (descriptor?.integrity === resolved.source.integrity) resolved.source = descriptor;
		return resolved;
	}

	#syncEpoch(): void {
		const epoch = this.host.sessionManager.getRequirementsEpoch();
		if (epoch === this.#epoch) return;
		this.cancelPending("Session applicability epoch changed");
		this.releasePendingLive();
		this.#sourceKeys.clear();
		this.#epoch = epoch;
	}

	currentSignature(): string {
		if (this.#enabled && this.#bypass === "off") this.#syncEpoch();
		if (!this.#enabled || this.#bypass !== "off") return `${this.host.sessionManager.getSessionId()}:${this.#lifecycle}:${this.#enabled}:${this.#bypass}`;
		const state = this.#storage.getRequirementsState();
		return `${this.host.sessionManager.getSessionId()}:${this.#epoch}:${this.#lifecycle}:${state.generation}:${state.publicationRevision}:${this.#enabled}:${this.#bypass}:${this.host.sessionManager.getRequirementsSourceVersion()}`;
	}

	async refreshCurrentEvidence(): Promise<void> {
		if (!this.#enabled || this.#bypass !== "off") return;
		if (!this.#hasObserved && !this.#observing) void this.observeCommittedSources().catch(error => { this.#error = String(error); });
		await this.#reconcileApplicableIntegrity();
	}

	prepareFragment(countTokens?: (text: string) => number): { fragment: string; receipt: RequirementsCallReceipt } {
		if (!this.#enabled || this.#bypass !== "off") {
			const receipt: RequirementsCallReceipt = { revisionIds: [], publicationRevision: 0, generation: this.#lifecycle, coverageComplete: false, bypass: this.#bypass, tokens: 0, tokenProvenance: "estimated", signature: this.currentSignature(), phase: "prepared" };
			this.recordCallReceipt(receipt);
			return { fragment: "", receipt };
		}
		const composed = this.composeProviderContext({ messages: [] }, countTokens);
		this.recordCallReceipt(composed.receipt);
		return { fragment: composed.context.systemPrompt?.join("\n\n") ?? "", receipt: composed.receipt };
	}

	snapshotApplicable(): RequirementsApplicableSnapshot {
		const signature = this.currentSignature();
		if (this.#applicableCache?.signature === signature) return this.#applicableCache;
		const snapshot = this.#consumptionSnapshot();
		const pendingSources = [...this.#pendingLive.keys()].flatMap(key => {
			const source = this.#storage.getRequirementsSource(key);
			return source && source.state !== "complete" && source.state !== "gap" ? [source] : [];
		});
		return this.#applicableCache = {
			...snapshot.applicable, ledgerCoverage: snapshot.coverage, pendingSources,
			publicationRevision: snapshot.state.publicationRevision, generation: snapshot.state.generation,
			enabled: this.#enabled, bypass: this.#bypass, signature,
		};
	}
	composeProviderContext(
		context: Context,
		countTokens?: (text: string) => number,
	): { context: Context; receipt: RequirementsCallReceipt } {
		const snapshot = this.snapshotApplicable();
		const cached = this.#composition;
		if (
			cached &&
			cached.base === context.systemPrompt &&
			cached.signature === snapshot.signature &&
			cached.countTokens === countTokens
		)
			return { context: { ...context, systemPrompt: cached.parts }, receipt: structuredClone(cached.receipt) };
		const composed = composeProviderRequirements(context, snapshot, countTokens);
		if (this.#observedVersion !== this.host.sessionManager.getRequirementsSourceVersion())
			composed.receipt.coverageComplete = false;
		const estimated = countTokens?.(
			JSON.stringify(
				snapshot.active.map(revision => ({
					id: revision.requirementId,
					revision: revision.id,
					statement: revision.statement,
					scope: revision.scope,
					source: revision.sourceKey,
				})),
			),
		);
		const activeTokens =
			typeof estimated === "number" && Number.isFinite(estimated) && estimated >= 0 ? estimated : null;
		this.#composition = {
			base: context.systemPrompt,
			signature: snapshot.signature,
			parts: composed.context.systemPrompt!,
			receipt: composed.receipt,
			activeTokens,
			countTokens,
		};
		return composed;
	}

	async acceptDelivered(entryId: string): Promise<void> {
		if (!this.#enabled || this.#bypass !== "off") return;
		const lifecycle = this.#lifecycle;
		await this.host.sessionManager.ensureOnDisk();
		await this.host.sessionManager.flush();
		if (lifecycle !== this.#lifecycle || !this.#enabled || this.#bypass !== "off") return;
		const source = this.host.sessionManager.getRequirementsSource(entryId);
		if (!source || source.referenceOnly || source.state !== "pending") return;
		const sessionId = this.host.sessionManager.getSessionId();
		if (this.#pendingSessionId !== sessionId) this.releasePendingLive();
		this.#pendingSessionId = sessionId;
		source.authorityGeneration = this.#storage.getRequirementsState().generation;
		this.#storage.intakeRequirementsSource(source);
		this.#sourceKeys.add(source.key);
		this.#pendingLive.set(source.key, { entryId, integrity: source.integrity });
		this.#pendingGeneration++;
		void this.processPending(source.key).catch(error => { this.#error = String(error); });
	}

	pendingLiveSnapshot(): Readonly<{sessionId: string; epoch: number; generation: number; entryIds: readonly string[]}> {
		const sessionId = this.host.sessionManager.getSessionId();
		if (this.#pendingSessionId !== undefined && this.#pendingSessionId !== sessionId) { this.releasePendingLive(); this.#pendingSessionId = sessionId; }
		if (this.#pendingSnapshot?.sessionId === sessionId && this.#pendingSnapshot.epoch === this.#epoch && this.#pendingSnapshot.generation === this.#pendingGeneration) return this.#pendingSnapshot;
		return this.#pendingSnapshot = Object.freeze({ sessionId, epoch: this.#epoch, generation: this.#pendingGeneration, entryIds: Object.freeze([...this.#pendingLive.values()].map(value => value.entryId)) });
	}
	status(): Omit<RequirementsStatus, "snapshot">;
	status(options: { includeLedger: true }): RequirementsStatus;
	status(options?: { includeLedger: true }): Omit<RequirementsStatus, "snapshot"> & { snapshot?: RequirementsSnapshot } {
		const applicable = this.snapshotApplicable();
		const activeTokens =
			this.#composition?.signature === applicable.signature ? this.#composition.activeTokens : null;
		return {
			enabled: this.#enabled,
			bypass: this.#bypass,
			running: !!this.#run,
			sourceCatalog: this.#observedVersion === this.host.sessionManager.getRequirementsSourceVersion()
				? "last-observed" : this.#observing ? "observing" : "unobserved",
			models: requirementsModelStatus(this.host),
			...(options?.includeLedger ? { snapshot: this.#snapshot() } : {}),
			applicable,
			activeTokens,
			tokenProvenance: activeTokens === null ? "unknown" : "estimated",
			receipt: this.#receipt ? structuredClone(this.#receipt) : undefined,
			error: this.#error,
		};
	}
	recordDispatch(): void {
		if (this.#receipt?.phase === "prepared") this.#receipt = { ...this.#receipt, phase: "sent" };
	}
	recordCallReceipt(receipt: RequirementsCallReceipt): void {
		this.#receipt = structuredClone(receipt);
	}
	async #evidencePackage(sourceKey: string, restoringRevisionId?: string): Promise<RequirementsEvidencePackage> {
		const source = await this.inspectSource(sourceKey, true);
		if (source.source.referenceOnly)
			throw new Error("Referents are evidence only; process the human source that adopted them");
		const snapshot = this.#consumptionSnapshot();
		source.source.adoptedUnitIds = restoringRevisionId
			? undefined
			: this.#storage.getRequirementsSource(sourceKey)?.adoptedUnitIds;
		const applicable = snapshot.applicable;
		const applicableRevisionIds = new Set(snapshot.applicableRevisionIds);
		// Include quarantined targets for restore and competing heads for explicit operator resolution.
		const restoring = snapshot.revisions.find(revision => revision.id === restoringRevisionId);
		const active = [
			...new Map(
				[
					...applicable.active,
					...applicable.conflicts.flat(),
					...snapshot.revisions.filter(
						revision =>
							applicableRevisionIds.has(revision.id) ||
							revision.id === restoringRevisionId ||
							restoring?.predecessorRevisionIds.includes(revision.id) ||
							source.operatorTargetRevisionIds?.includes(revision.id),
					),
				].map(revision => [revision.id, revision]),
			).values(),
		];
		const references: ResolvedRequirementsSource[] = [];
		const previous = this.#storage.getRequirementsLatestBatch(sourceKey);
		const keys = new Set([
			...Object.keys(previous?.readSourceIntegrities ?? {}),
			...(this.#extraEvidence.get(sourceKey) ?? []),
			...active.flatMap(revision => [
				revision.sourceKey,
				...revision.evidence.map(span => span.sourceKey),
				...(revision.referents ?? []).map(span => span.sourceKey),
				...(revision.relations ?? []).flatMap(relation => [
					relation.predecessorSourceKey, relation.successorSourceKey,
					...relation.evidence.map(span => span.sourceKey),
				]),
			]),
		]);
		for (const key of keys) if (key !== sourceKey) references.push(await this.inspectSource(key));
		return {
			source,
			references,
			active,
			applicableRevisionIds: snapshot.applicableRevisionIds,
			projectId: this.host.sessionManager.getCwd(),
			authority: this.#authority(snapshot),
			publicationRevision: snapshot.state.publicationRevision,
			readHeads: snapshot.readHeads,
		};
	}
	#recordReferencedSources(batch: RequirementsBatch, input: RequirementsEvidencePackage): void {
		const seen = new Set<string>();
		const visit = (resolved: ResolvedRequirementsSource) => {
			const { source } = resolved;
			if (seen.has(source.key)) return;
			seen.add(source.key);
			// Keep selected provenance in the existing ledger, never the whole context catalog.
			if (source.referenceOnly && batch.readSourceIntegrities[source.key] === source.integrity)
				this.#storage.intakeRequirementsSource(source);
			for (const referent of resolved.referents) visit(referent);
		};
		visit(input.source);
		for (const reference of input.references) visit(reference);
	}
	#sourceInScope(source: RequirementsSource): boolean {
		const manager = this.host.sessionManager;
		return (
			(source.locators.some(locator => locator.sessionId === manager.getSessionId()) &&
				manager.isRequirementsSourceApplicable(source)) ||
			this.#consumptionSnapshot().sources.some(current => current.key === source.key)
		);
	}
	async processPending(sourceKey?: string): Promise<void> {
		const signal = this.#controller.signal;
		this.#assertAvailable(signal);
		if (this.#run) {
			await this.#run;
			this.#assertAvailable(signal);
			return this.processPending(sourceKey);
		}
		const run = (async () => {
			if (!sourceKey) await this.observeCommittedSources();
			this.#assertAvailable(signal);
			const selected = sourceKey ? this.#storage.getRequirementsSource(sourceKey) : undefined;
			if (sourceKey && !selected) throw new Error(`Unknown source: ${sourceKey}`);
			if (selected?.referenceOnly)
				throw new Error("Referents are evidence only; retry the human source that adopted them");
			// Resolve exact local ancestry or current dependencies without waiting for historical indexing.
			if (selected && !this.#sourceInScope(selected))
				throw new Error("Foreign requirements source is not associated with the current scope");
			// Indexed current input, not historical unknown-origin descriptors, drives automatic work.
			const candidates = selected ? [selected] : this.#storage.getRequirementsPendingSources();
			const pending = candidates.filter(source =>
				!source.referenceOnly && source.state === "pending" &&
				(sourceKey === source.key || this.#sourceKeys.has(source.key)),
			);
			for (const source of pending) {
				signal.throwIfAborted();
				this.#assertAvailable();
				let processingIntegrity = source.integrity;
				try {
					const input = await this.#evidencePackage(source.key);
					this.#assertAvailable(signal);
					processingIntegrity = input.source.source.integrity;
					this.#storage.authorizeRequirementsOwner(input.authority);
					this.#authorizedOwnerSessionId = input.authority.ownerSessionId;
					this.#storage.saveRequirementsBatch(createRequirementsBatch(input));
					const batch = await extractRequirementsBatch(this.host, input, signal);
					signal.throwIfAborted();
					this.#assertAvailable();
					this.#recordReferencedSources(batch, input);
					batch.id = this.#storage.saveRequirementsBatch(batch);
					if (!requirementsReviewAccepted(batch.review))
						throw new Error(batch.reason ?? "Independent review rejected or uncertain; no publication");
					const verified = await this.#reconcileReviewedEvidence(input, batch);
					signal.throwIfAborted();
					this.#assertAvailable();
					const result = this.#storage.publishRequirementsBatch(batch.id, input.authority, verified);
					if (result.status !== "accepted" && result.status !== "already-accepted")
						throw new Error(result.reason ?? result.status);
					if (this.#pendingLive.delete(source.key)) this.#pendingGeneration++;
					this.#error = undefined;
				} catch (error) {
					if (signal.aborted) throw error;
					this.#error = error instanceof Error ? error.message : String(error);
					const current = this.#storage.getRequirementsSource(source.key);
					if (current && current.state !== "complete" && current.integrity === processingIntegrity)
						this.#storage.setRequirementsSourceDisposition(
							source.key,
							processingIntegrity,
							error instanceof RequirementsPipelineError && error.unsupported ? "unsupported" : "failed",
							this.#error,
						);
				}
			}
		})();
		this.#run = run;
		try {
			await run;
		} finally {
			if (this.#run === run) this.#run = undefined;
		}
	}
	cancelPending(reason = "Requirements owner lifecycle changed"): void {
		this.#lifecycle++;
		this.#controller.abort(new Error(reason));
		this.#controller = new AbortController();
		if (this.#authorizedOwnerSessionId !== undefined) {
			this.#storage.invalidateRequirementsOwner(this.#authorizedOwnerSessionId);
			this.#authorizedOwnerSessionId = undefined;
		}
	}
	setRecoveryMode(mode: RequirementsRecoveryMode): void {
		if (this.#bypass !== "off" && mode === "off")
			throw new Error("Memory recovery bypass is sticky for this run; restart explicitly to resume memory");
		this.#bypass = mode;
		if (mode !== "off") { this.cancelPending(`Memory ${mode} recovery`); this.releasePendingLive(); }
	}
	releasePendingLive(): void {
		this.#pendingLive.clear();
		this.#pendingGeneration++;
	}

	dispose(): void {
		this.#disposed = true;
		this.#unsubscribeSettings();
		this.cancelPending("Requirements owner disposed");
		this.releasePendingLive();
	}
	async applyOperatorAction(action: RequirementsOperatorAction): Promise<unknown> {
		const signal = this.#controller.signal;
		const actor = `operator:${this.host.sessionManager.getSessionId()}`;
		if (action.kind === "inspect") {
			const revision = this.#storage.getRequirementsRevision(action.revisionId);
			if (!revision) throw new Error(`Unknown requirements revision: ${action.revisionId}`);
			return structuredClone(revision);
		}
		if (action.kind === "quarantine") {
			this.cancelPending("Operator quarantined requirements");
			return this.#storage.quarantineRequirements(action.revisionIds, actor, action.reason);
		}
		if (action.kind === "clear") {
			this.cancelPending("Operator cleared requirements");
			const kinds: RequirementsScope["kind"][] = action.scope === "all" ? ["session", "task", "project", "global"] : action.scope === "session" ? ["session", "task"] : [action.scope];
			const result = kinds.map(kind => this.#storage.clearRequirements({ kind, sessionId: this.host.sessionManager.getSessionId(), epoch: this.#epoch, projectId: this.host.sessionManager.getCwd() }, actor, action.reason));
			this.releasePendingLive();
			return result;
		}
		if (action.kind === "decision") {
			if (!action.text.trim()) throw new Error("Operator decision must not be empty");
			for (const id of action.targetRevisionIds ?? [])
				if (!this.#storage.getRequirementsRevision(id)) throw new Error(`Unknown operator target revision: ${id}`);
			await this.host.promptOperatorSource(action.text, action.targetRevisionIds);
			return this.status();
		}
		if (action.kind === "retry") {
			if (!action.sourceKey) await this.observeCommittedSources(true);
			this.#assertAvailable(signal);
			const sources = action.sourceKey
				? [this.#storage.getRequirementsSource(action.sourceKey)].filter((source): source is RequirementsSource => !!source)
				: this.#storage.getRequirementsPendingSources();
			for (const source of sources) {
				if (source.referenceOnly || source.state === "complete") continue;
				if (!this.#sourceInScope(source)) continue;
				const resolved = await this.inspectSource(source.key);
				this.#assertAvailable(signal);
				this.#storage.setRequirementsSourceDisposition(source.key, resolved.source.integrity, "pending", "Operator requested original-position retry/backfill");
				await this.processPending(source.key);
				this.#assertAvailable(signal);
			}
			return this.status();
		}
		// Exact actions consume addressed sources/revisions, not the unrelated historical catalog.
		signal.throwIfAborted();
		if (action.kind === "gap") {
			const source = this.#storage.getRequirementsSource(action.sourceKey);
			if (!source) throw new Error(`Unknown requirements source: ${action.sourceKey}`);
			this.#storage.recordRequirementsGap(source.key, source.integrity, actor, action.reason);
			if (this.#pendingLive.delete(source.key)) this.#pendingGeneration++;
			return this.status();
		}
		if (action.kind === "evidence") {
			this.#assertAvailable(signal);
			const source = this.#storage.getRequirementsSource(action.sourceKey);
			if (!source) throw new Error(`Unknown requirements source: ${action.sourceKey}`);
			if (!this.#sourceInScope(source))
				throw new Error("Foreign requirements source is not associated with the current scope");
			for (const key of action.referentKeys) {
				await this.inspectSource(key);
				this.#assertAvailable(signal);
			}
			this.#extraEvidence.set(action.sourceKey, [...action.referentKeys]);
			this.#storage.setRequirementsSourceDisposition(
				source.key,
				source.integrity,
				"pending",
				"Operator supplied host-resolved evidence",
			);
			await this.processPending(action.sourceKey);
			this.#assertAvailable(signal);
			return this.status();
		}
		this.#assertAvailable(signal);
		if (action.kind === "literal-adopt") {
			const selected = await this.inspectSource(action.sourceKey);
			this.#assertAvailable(signal);
			const selectedUnit = selected.units.find(unit => unit.id === action.unitId);
			if (selectedUnit?.text === undefined || !selectedUnit.text.trim())
				throw new Error("Literal adoption requires a complete original text unit");
			const adopted = selected.source.referenceOnly || selected.source.origin.kind !== "human";
			let sourceKey = action.sourceKey;
			if (adopted) {
				const entryId = await this.host.promptOperatorSource(selectedUnit.text, undefined, { literal: true });
				this.#assertAvailable(signal);
				const source = this.host.sessionManager.getRequirementsSource(entryId);
				if (!source) throw new Error("Accepted operator adoption source is unavailable");
				sourceKey = source.key;
				source.authorityGeneration = this.#storage.getRequirementsState().generation;
				this.#storage.intakeRequirementsSource(source);
				this.#sourceKeys.add(sourceKey);
				this.#extraEvidence.set(sourceKey, [action.sourceKey]);
			}
			const input = await this.#evidencePackage(sourceKey);
			this.#assertAvailable(signal);
			const unit = adopted ? input.source.units.find(unit => unit.text === selectedUnit.text) : selectedUnit;
			if (!unit) throw new Error("Accepted operator adoption does not preserve the selected complete unit");
			const candidate = admitRequirementsCandidates(
				{
					version: REQUIREMENTS_FORMAT,
					sourceKey: input.source.source.key,
					sourceIntegrity: input.source.source.integrity,
					manifest: input.source.source.units.map(({ id, sha256, byteLength }) => ({ id, sha256, byteLength })),
					disposition: "complete",
					unresolved: [],
					operations: [
						{
							id: randomUUID(),
							kind: "add",
							statement: unit.text,
							scope: action.scope,
							evidence: [
								{
									sourceKey: input.source.source.key,
									integrity: input.source.source.integrity,
									unitId: unit.id,
								},
							],
							referents: adopted ? [{ sourceKey: selected.source.key, integrity: selected.source.integrity, unitId: selectedUnit.id }] : [],
							predecessorRevisionIds: [],
						},
					],
				},
				input,
			);
			const literalAcceptance = {
				actor,
				sourceKey: input.source.source.key,
				integrity: input.source.source.integrity,
				unitIds: [unit.id],
			};
			this.#storage.authorizeRequirementsOwner(input.authority);
			this.#authorizedOwnerSessionId = input.authority.ownerSessionId;
			const review = await reviewRequirementsCandidates(this.host, input, candidate, signal, literalAcceptance);
			this.#assertAvailable(signal);
			const batch: RequirementsBatch = {
				id: randomUUID(),
				sourceKey: input.source.source.key,
				sourceIntegrity: input.source.source.integrity,
				extractionVersion: REQUIREMENTS_FORMAT,
				reviewRevision: `${REQUIREMENTS_FORMAT}:${input.publicationRevision}:literal:${unit.id}`,
				manifest: input.source.source.units,
				readHeads: input.readHeads,
				readSourceIntegrities: Object.fromEntries([input.source, ...input.references].map(resolved => [resolved.source.key, resolved.source.integrity])),
				authority: input.authority,
				operationIds: candidate.map(operation => operation.id),
				operations: candidate.map(({ id: _id, ...op }) => op),
				review,
				status: requirementsReviewAccepted(review) ? "reviewed" : "rejected",
			};
			batch.id = this.#storage.saveRequirementsBatch(batch);
			const verified = await this.#reconcileReviewedEvidence(input, batch);
			this.#assertAvailable(signal);
			const result = this.#storage.publishRequirementsBatch(batch.id, input.authority, verified);
			if (this.#storage.getRequirementsSource(action.sourceKey)?.state === "complete" && this.#pendingLive.delete(action.sourceKey)) this.#pendingGeneration++;
			return result;
		}
		if (action.kind === "restore") {
			const restored: string[] = [];
			for (const id of action.revisionIds) {
				const revision = this.#storage.getRequirementsRevision(id);
				if (!revision || revision.lifecycle !== "quarantined")
					throw new Error(`Revision is not quarantined: ${id}`);
				const input = await this.#evidencePackage(revision.sourceKey, revision.id);
				this.#assertAvailable(signal);
				if (input.source.source.integrity !== revision.sourceIntegrity)
					throw new Error("Restore evidence changed; original decision remains suspended");
				for (const span of [...revision.evidence, ...(revision.referents ?? [])])
					verifyRequirementsEvidence(span, input);
				const candidates = admitRequirementsCandidates(
					{
						version: REQUIREMENTS_FORMAT,
						sourceKey: input.source.source.key,
						sourceIntegrity: input.source.source.integrity,
						manifest: input.source.source.units.map(({ id, sha256, byteLength }) => ({ id, sha256, byteLength })),
						disposition: "complete",
						unresolved: [],
						operations: [
							{
								id: randomUUID(),
								kind: revision.kind,
								...(revision.kind !== "add" ? { requirementId: revision.requirementId } : {}),
								statement: revision.statement,
								scope: revision.scope,
								evidence: revision.evidence,
								referents: revision.referents,
								predecessorRevisionIds: revision.predecessorRevisionIds,
							},
						],
					},
					input,
				);
				const literal: RequirementsReview["literalAcceptance"] = action.literalUnitId
					? {
							actor,
							sourceKey: revision.sourceKey,
							integrity: revision.sourceIntegrity,
							unitIds: [action.literalUnitId],
						}
					: undefined;
				const review = await reviewRequirementsCandidates(this.host, input, candidates, signal, literal);
				this.#assertAvailable(signal);
				const verified = await this.#reconcileReviewedEvidence(input);
				this.#assertAvailable(signal);
				const receipt = {
					id: randomUUID(),
					actor,
					revisionIds: [id],
					operationIds: candidates.map(operation => operation.id),
					generation: input.authority.generation,
					publicationRevision: input.publicationRevision,
					readHeads: input.readHeads,
					sourceIntegrities: verified,
					review,
				};
				if (!this.#storage.restoreRequirements(receipt))
					throw new Error(
						review.sanity?.reason ??
							review.evidence?.reason ??
							"Fresh restoration validation unavailable, rejected, uncertain or stale; revision remains suspended",
					);
				restored.push(id);
			}
			return restored;
		}
	}
}
