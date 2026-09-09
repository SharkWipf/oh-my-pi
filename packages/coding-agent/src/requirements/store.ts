import { createHash, randomUUID } from "node:crypto";
import type {
	RequirementsApplicable,
	RequirementsAuthority,
	RequirementsBatch,
	RequirementsConsumptionContext,
	RequirementsConsumptionSnapshot,
	RequirementsCoverageSummary,
	RequirementsEvidence,
	RequirementsObservation,
	RequirementsOperation,
	RequirementsPublicationResult,
	RequirementsRestoreReceipt,
	RequirementsRevision,
	RequirementsRelation,
	RequirementsScope,
	RequirementsSnapshot,
	RequirementsSource,
	RequirementsSourceMetadata,
	RequirementsSourceState,
} from "./types";

export function createEmptyRequirementsSnapshot(): RequirementsSnapshot {
	return {
		state: { publicationRevision: 0, generation: 0, owners: {}, restoreReceipts: [], restoreReviews: {} },
		sources: [],
		batches: [],
		revisions: [],
	};
}
export function createEmptyRequirementsCoverageSummary(): RequirementsCoverageSummary {
	return {
		total: 0,
		referenceOnly: 0,
		byState: { pending: 0, complete: 0, unsupported: 0, failed: 0, gap: 0, orphaned: 0 },
	};
}

function addOperationSourceKeys(operation: RequirementsOperation, keys: Set<string>): void {
	for (const span of operation.evidence) keys.add(span.sourceKey);
	for (const span of operation.referents ?? []) keys.add(span.sourceKey);
	for (const relation of operation.relations ?? []) {
		keys.add(relation.predecessorSourceKey);
		keys.add(relation.successorSourceKey);
		for (const span of relation.evidence) keys.add(span.sourceKey);
	}
}

/** Verify only current scoped authority and live work; historical catalog access is explicit. */
export function getRequirementsConsumedSources(
	snapshot: RequirementsSnapshot,
	context: RequirementsConsumptionContext,
	readSource: (key: string) => RequirementsSource | undefined,
): Pick<RequirementsConsumptionSnapshot, "sources" | "applicable" | "applicableRevisionIds"> {
	const heads = getApplicableRequirementHeads(snapshot, context, readSource);
	const keys = new Set<string>();
	const revisions = new Map(snapshot.revisions.map(revision => [revision.id, revision]));
	const revisionIds = new Set<string>();
	const batchIds = new Set<string>();
	const operationSources = (operation: RequirementsOperation) => {
		addOperationSourceKeys(operation, keys);
		for (const id of operation.predecessorRevisionIds) revisionIds.add(id);
	};
	for (const group of heads) {
		if (!group.some(revision => revision.lifecycle !== "historical")) continue;
		for (const revision of group) {
			revisionIds.add(revision.id);
			batchIds.add(revision.batchId);
		}
	}
	// A non-head suppressed only by a current source relation may become a head after
	// evidence changes. Explicit predecessor IDs, unlike that relation, cannot disappear.
	const scoped = snapshot.revisions.filter(revision => requirementsScopeApplies(revision, context));
	const explicitPredecessors = new Set(scoped.flatMap(revision => revision.predecessorRevisionIds));
	const headIds = new Set(heads.flatMap(group => group.map(revision => revision.id)));
	const dynamicLineages = new Set(scoped.filter(revision => revision.lifecycle !== "historical" &&
		!headIds.has(revision.id) && !explicitPredecessors.has(revision.id)).map(revision => revision.requirementId));
	for (const revision of scoped) if (dynamicLineages.has(revision.requirementId)) {
		revisionIds.add(revision.id);
		batchIds.add(revision.batchId);
	}
	const batchSources = (batch: RequirementsBatch) => {
		keys.add(batch.sourceKey);
		for (const key of Object.keys(batch.readSourceIntegrities)) keys.add(key);
		if (batch.review.literalAcceptance) keys.add(batch.review.literalAcceptance.sourceKey);
		for (const operation of batch.operations) operationSources(operation);
	};
	const batchesById = new Map(snapshot.batches.map(batch => [batch.id, batch]));
	for (const batch of snapshot.batches) {
		const authority = batch.authority;
		const live = (batch.status === "pending" || batch.status === "reviewed") &&
			authority.ownerSessionId === context.sessionId && authority.branchId === context.branchId &&
			authority.epoch === context.epoch && authority.generation === snapshot.state.generation &&
			sameAuthority(snapshot.state.owners[context.sessionId], authority) &&
			(!context.sourceKeys || context.sourceKeys.has(batch.sourceKey));
		if (!live && !batchIds.has(batch.id)) continue;
		batchSources(batch);
	}
	for (const id of revisionIds) {
		const revision = revisions.get(id);
		if (!revision) continue;
		keys.add(revision.sourceKey);
		operationSources(revision);
	}
	// Source relations can be established by a different accepted decision. Follow only
	// relation authority reachable from this operation's consumed source ancestry.
	const relationOwners = new Map<string, RequirementsRevision[]>();
	for (const revision of snapshot.revisions) {
		if (revision.lifecycle === "historical") continue;
		for (const relation of revision.relations ?? []) {
			const owners = relationOwners.get(relation.successorSourceKey) ?? [];
			owners.push(revision);
			relationOwners.set(relation.successorSourceKey, owners);
		}
	}
	const sources: RequirementsSource[] = [];
	const visitedRevisions = new Set(revisionIds);
	const pendingRevisions: RequirementsRevision[] = [];
	const addRevision = (revision: RequirementsRevision) => {
		pendingRevisions.push(revision);
		while (pendingRevisions.length) {
			const current = pendingRevisions.pop()!;
			if (visitedRevisions.has(current.id)) continue;
			visitedRevisions.add(current.id);
			keys.add(current.sourceKey);
			operationSources(current);
			const batch = batchesById.get(current.batchId);
			if (batch) batchSources(batch);
			for (const id of current.predecessorRevisionIds) {
				const predecessor = revisions.get(id);
				if (predecessor) pendingRevisions.push(predecessor);
			}
		}
	};
	// Set iteration visits newly added ancestors and terminates even on corrupt cycles.
	for (const key of keys) {
		const source = readSource(key);
		if (!source) continue;
		sources.push(source);
		if (source.parentKey) keys.add(source.parentKey);
		for (const revision of relationOwners.get(key) ?? []) addRevision(revision);
	}
	// Historical head gaps remain inspectable ledger metadata, not freshly consumed bytes.
	const gapKeys = new Set<string>();
	for (const group of heads) for (const revision of group) {
		gapKeys.add(revision.sourceKey);
		addOperationSourceKeys(revision, gapKeys);
	}
	const coverageGaps = new Map(sources.filter(source => !source.referenceOnly && source.state !== "complete").map(source => [source.key, source]));
	for (const key of gapKeys) {
		const source = readSource(key);
		if (!source) continue;
		if (!source.referenceOnly && source.state !== "complete") coverageGaps.set(key, source);
		if (source.parentKey) gapKeys.add(source.parentKey);
	}
	return {
		sources,
		applicableRevisionIds: heads.filter(group => group.length === 1 && group[0].lifecycle === "accepted").map(group => group[0].id),
		applicable: {
			...applicableHeads(heads),
			coverageGaps: [...coverageGaps.values()],
		},
	};
}

export function requirementsSourceKey(original: { journalId: string; entryId: string }): string {
	return JSON.stringify([original.journalId, original.entryId]);
}
function sameAuthority(a: RequirementsAuthority | undefined, b: RequirementsAuthority): boolean {
	return (
		!!a &&
		a.ownerSessionId === b.ownerSessionId &&
		a.branchId === b.branchId &&
		a.epoch === b.epoch &&
		a.generation === b.generation
	);
}
function sourceAt(sources: ReadonlyMap<string, RequirementsSource>, key: string): RequirementsSource {
	const source = sources.get(key);
	if (!source) throw new Error(`Unknown requirements source: ${key}`);
	return source;
}
function validateSpan(sources: ReadonlyMap<string, RequirementsSource>, span: RequirementsEvidence): void {
	const source = sourceAt(sources, span.sourceKey);
	const unit = source.units.find(item => item.id === span.unitId);
	if (
		source.integrity !== span.integrity ||
		source.integrityAvailable === false ||
		source.state === "orphaned" ||
		!unit ||
		unit.unsupportedReason ||
		"start" in span || "end" in span
	) {
		throw new Error("Evidence is unavailable, changed, or not a complete host-issued unit");
	}
}
function validateScope(scope: RequirementsScope): void {
	if (
		!["global", "project", "session", "task"].includes(scope.kind) ||
		(scope.kind === "project" && !scope.projectId) ||
		((scope.kind === "session" || scope.kind === "task") && (!scope.sessionId || !Number.isSafeInteger(scope.epoch)))
	) {
		throw new Error("Requirement scope lacks its concrete identity");
	}
}
function validateReview(
	sources: ReadonlyMap<string, RequirementsSource>,
	review: RequirementsBatch["review"],
	operations: RequirementsBatch["operations"],
	operationIds: readonly string[],
	sourceKey: string,
	revisions: readonly RequirementsRevision[],
	readHeads: Record<string, string[]>,
): void {
	if (!Array.isArray(operationIds) || operationIds.length !== operations.length || new Set(operationIds).size !== operationIds.length || operationIds.some(id => !id.trim()))
		throw new Error("Reviewed candidate identities do not match the atomic proposal");
	const byId = new Map(operationIds.map((id, index) => [id, operations[index]]));
	const accepted = (outcome: RequirementsBatch["review"]["sanity"], label: string) => {
		if (outcome?.outcome !== "accepted" || !outcome.model || !outcome.format || !Array.isArray(outcome.candidates) || outcome.candidates.length !== operationIds.length)
			throw new Error(`${label} requires complete independent candidate acceptance`);
		const seen = new Set<string>();
		for (const candidate of outcome.candidates) {
			if (seen.has(candidate.id) || !byId.has(candidate.id) || candidate.decision !== "pass" || !candidate.reason.trim()) throw new Error(`${label} candidate decision is absent, duplicated, rejected or uncertain`);
			seen.add(candidate.id);
		}
	};
	accepted(review.sanity, "Candidate-only sanity");
	const source = sourceAt(sources, sourceKey);
	if (!review.literalAcceptance) {
		accepted(review.evidence, "Independent evidence review");
		if (!Array.isArray(review.evidence?.obligations)) throw new Error("Independent obligation inventory is absent");
		const known = new Map(revisions.map(revision => [revision.id, revision]));
		const covered = new Set<string>(), seen = new Set<string>();
		for (const obligation of review.evidence.obligations) {
			if (!obligation.id || seen.has(obligation.id) || obligation.decision !== "pass" || !obligation.reason.trim() || !obligation.statement.trim() || !obligation.sourceUnitIds.length || obligation.sourceUnitIds.some(id => !source.units.some(unit => unit.id === id && !unit.unsupportedReason))) throw new Error("Independent obligation is invalid, uncovered, rejected or uncertain");
			seen.add(obligation.id);
			if (!obligation.operationIds.length && !obligation.applicableRevisionIds.length) throw new Error("Independent obligation has no candidate or applicable-head coverage");
			if (obligation.adoptedUnitIds.some(id => !source.adoptedUnitIds?.includes(id) || !obligation.sourceUnitIds.includes(id))) throw new Error("Obligation claims unaccepted literal coverage");
			for (const id of obligation.operationIds) {
				const operation = byId.get(id);
				if (!operation || operation.kind !== obligation.kind || (operation.kind !== "add" && operation.requirementId !== obligation.requirementId) || obligation.predecessorRevisionIds.some(id => !operation.predecessorRevisionIds.includes(id)) || !operation.evidence.some(unit => unit.sourceKey === sourceKey && obligation.sourceUnitIds.includes(unit.unitId))) throw new Error("Candidate does not perform the independently inventoried obligation");
				covered.add(id);
			}
			for (const id of obligation.applicableRevisionIds) {
				const revision = known.get(id), heads = revision && readHeads[revision.requirementId];
				if (!revision || revision.lifecycle !== "accepted" || (revision.availability && revision.availability !== "available") || heads?.length !== 1 || heads[0] !== id || (obligation.kind === "withdraw") !== (revision.kind === "withdraw") || (obligation.kind !== "add" && (revision.requirementId !== obligation.requirementId || obligation.predecessorRevisionIds.includes(id)))) throw new Error("Obligation does not resolve to a current applicable accepted head");
			}
		}
		if (operationIds.some(id => !covered.has(id))) throw new Error("Candidate lacks independently inventoried source-obligation coverage");
		return;
	}
	const literal = review.literalAcceptance;
	if (!literal.actor || sourceKey !== literal.sourceKey || source.integrity !== literal.integrity || !literal.unitIds.length || operations.length !== literal.unitIds.length)
		throw new Error("Literal acceptance requires complete current units");
	const adopted = new Set<string>();
	for (const operation of operations) {
		if (operation.evidence.length !== 1) throw new Error("Literal acceptance requires one complete unit per operation");
		const span = operation.evidence[0], unit = source.units.find(item => item.id === span.unitId);
		if (!unit || span.sourceKey !== source.key || span.integrity !== source.integrity || !literal.unitIds.includes(unit.id) || adopted.has(unit.id) || unit.byteLength !== Buffer.byteLength(operation.statement) || unit.sha256 !== createHash("sha256").update(operation.statement).digest("hex"))
			throw new Error("Literal acceptance is not a complete source unit");
		adopted.add(unit.id);
	}
}

/** Pure synchronous domain mutations; AgentStorage supplies the private SQLite transaction. */
class RequirementsReducer {
	readonly #sources = new Map<string, RequirementsSource>();
	constructor(readonly snapshot: RequirementsSnapshot = createEmptyRequirementsSnapshot(), readonly readSource?: (key: string) => RequirementsSource | undefined, readonly readRelations?: (key: string) => RequirementsRelation[]) {
		for (const source of snapshot.sources) this.#sources.set(source.key, source);
	}
	invalidateRequirementsOwner(ownerSessionId: string): void {
		delete this.snapshot.state.owners[ownerSessionId];
	}
	authorizeRequirementsOwner(authority: RequirementsAuthority): void {
		if (authority.generation !== this.snapshot.state.generation) throw new Error("Stale requirements generation");
		this.snapshot.state.owners[authority.ownerSessionId] = { ...authority };
	}
	intakeRequirementsSources(sources: readonly RequirementsSource[]): void {
		for (const source of sources) this.intakeRequirementsSource(source);
	}
	intakeRequirementsSource(source: RequirementsSource): void {
		if (source.key !== requirementsSourceKey(source.original))
			throw new Error("Source key does not match original journal and entry");
		if (source.state === "complete" || source.state === "gap" || source.adoptedUnitIds?.length)
			throw new Error("Intake cannot certify source disposition");
		if (
			!source.integrity ||
			new Set(source.units.map(unit => unit.id)).size !== source.units.length ||
			source.units.some(unit => !Number.isSafeInteger(unit.byteLength) || unit.byteLength < 0 || !unit.sha256)
		)
			throw new Error("Invalid full-unit manifest");
		const existing = this.#sources.get(source.key);
		if (existing) {
			const metadataOnly = source.units.length === 0 && source.integrityAvailable === false;
			if (!metadataOnly && (existing.integrity !== source.integrity || existing.integrityAvailable === false))
				this.reconcileRequirementsSources([{ key: source.key, integrity: source.integrity, units: source.units }]);
			if (!metadataOnly) {
				existing.units = structuredClone(source.units);
				existing.origin = structuredClone(source.origin);
				if (source.state === "unsupported" && existing.state !== "complete" && existing.state !== "gap") { existing.state = source.state; existing.reason = source.reason; }
			}
			for (const locator of source.locators)
				if (!existing.locators.some(item => JSON.stringify(item) === JSON.stringify(locator)))
					existing.locators.push(locator);
			if (source.authorityGeneration !== undefined && existing.authorityGeneration === undefined) existing.authorityGeneration = source.authorityGeneration;
			return;
		}
		const captured = structuredClone(source);
		this.snapshot.sources.push(captured);
		this.#sources.set(captured.key, captured);
	}
	setRequirementsSourceDisposition(
		key: string,
		integrity: string,
		state: Exclude<RequirementsSourceState, "complete" | "gap">,
		reason: string,
	): void {
		const source = sourceAt(this.#sources, key);
		if (source.integrity !== integrity) throw new Error("Stale source disposition");
		if (source.state === "complete")
			throw new Error("Accepted disposition can only change through source reconciliation");
		source.state = state;
		source.reason = reason;
	}
	recordRequirementsGap(key: string, integrity: string, actor: string, reason: string): void {
		const source = sourceAt(this.#sources, key);
		if (!actor || !reason || source.referenceOnly || source.integrity !== integrity || source.state === "complete")
			throw new Error("Gap requires an explicit current-source operator disposition");
		source.gap = { actor, integrity, reason };
		source.state = "gap";
		source.reason = reason;
		this.snapshot.state.publicationRevision++;
	}
	saveRequirementsBatch(batch: RequirementsBatch): string {
		if (batch.status === "accepted") throw new Error("Only publication can accept a batch");
		const source = sourceAt(this.#sources, batch.sourceKey);
		if (source.referenceOnly) throw new Error("Referents cannot authorize a requirements proposal");
		if (source.integrity !== batch.sourceIntegrity) throw new Error("Batch source changed");
		const existing = this.snapshot.batches.find(
			item =>
				item.id === batch.id ||
				(item.sourceKey === batch.sourceKey &&
					item.sourceIntegrity === batch.sourceIntegrity &&
					item.extractionVersion === batch.extractionVersion &&
					item.reviewRevision === batch.reviewRevision),
		);
		if (
			existing &&
			(existing.sourceKey !== batch.sourceKey ||
				existing.sourceIntegrity !== batch.sourceIntegrity ||
				existing.extractionVersion !== batch.extractionVersion ||
				existing.reviewRevision !== batch.reviewRevision)
		)
			throw new Error("Batch ID belongs to different frozen work");
		if (existing?.status === "accepted") return existing.id;
		this.snapshot.batches = this.snapshot.batches.filter(
			item => item.status === "accepted" || item.sourceKey !== batch.sourceKey,
		);
		this.snapshot.batches.push(structuredClone({ ...batch, id: existing?.id ?? batch.id }));
		return existing?.id ?? batch.id;
	}
	publishRequirementsBatch(
		batchId: string,
		authority: RequirementsAuthority,
		verifiedIntegrities: Record<string, string>,
	): RequirementsPublicationResult {
		const { snapshot } = this;
		const batch = snapshot.batches.find(item => item.id === batchId);
		const result = (
			status: RequirementsPublicationResult["status"],
			reason?: string,
		): RequirementsPublicationResult => {
			if (batch && batch.status !== "accepted") {
				if (status === "stale" || status === "rejected") batch.status = status;
				batch.reason = reason;
			}
			return {
				status,
				reason,
				revisionIds: snapshot.revisions.filter(item => item.batchId === batchId).map(item => item.id),
				publicationRevision: snapshot.state.publicationRevision,
			};
		};
		if (!batch) return result("rejected", "Unknown batch");
		if (batch.status === "accepted") return result("already-accepted");
		if (
			!sameAuthority(snapshot.state.owners[authority.ownerSessionId], authority) ||
			!sameAuthority(batch.authority, authority) ||
			authority.generation !== snapshot.state.generation
		)
			return result("stale", "Owner branch, epoch, or generation changed");
		const source = sourceAt(this.#sources, batch.sourceKey);
		if (source.referenceOnly) return result("rejected", "Referents cannot authorize a requirements proposal");
		if (
			source.integrityAvailable === false ||
			source.integrity !== batch.sourceIntegrity ||
			verifiedIntegrities[source.key] !== source.integrity ||
			source.state === "orphaned"
		)
			return result("stale", "Original source no longer resolves unchanged");
		if (source.state === "complete") return result("rejected", "Source already has an accepted atomic proposal");
		for (const operation of batch.operations)
			if (snapshot.state.tombstones?.some(item => sameScope(item.scope, operation.scope) && (source.authorityGeneration ?? -1) < item.generation))
				return result("rejected", "Source predates an explicit scoped clear; new accepted operator input is required");
		if (!readHeadsMatch(snapshot, batch.readHeads))
			return result("stale", "Reviewed requirement heads changed; isolated relationship review required");
		if (batch.status !== "reviewed") return result("waiting", "Independent reviews unfinished");
		try {
			if (JSON.stringify(batch.manifest) !== JSON.stringify(source.units))
				throw new Error("Frozen full-source manifest changed");
			if (!batch.review.literalAcceptance && source.units.some(unit => unit.unsupportedReason))
				throw new Error("Source contains unprocessed unsupported units");
			const targets = batch.operations.flatMap(operation =>
				operation.requirementId ? [operation.requirementId] : [],
			);
			if (new Set(targets).size !== targets.length)
				throw new Error("Atomic proposal contains competing operations for one requirement");
			for (const [key, integrity] of Object.entries(batch.readSourceIntegrities))
				if (sourceAt(this.#sources, key).integrity !== integrity || verifiedIntegrities[key] !== integrity)
					throw new Error("Reviewed source integrity changed or was not verified");
			validateReview(this.#sources, batch.review, batch.operations, batch.operationIds, source.key, snapshot.revisions, batch.readHeads);
			if (batch.review.literalAcceptance?.unitIds.some(id => source.adoptedUnitIds?.includes(id)))
				throw new Error("Complete source unit already adopted");
			for (const operation of batch.operations) {
				validateScope(operation.scope);
				if (
					!["add", "change", "withdraw"].includes(operation.kind) ||
					!operation.statement.trim() ||
					!operation.evidence.length
				)
					throw new Error("Invalid requirement operation");
				if (
					source.origin.kind !== "human" ||
					!operation.evidence.some(span => span.sourceKey === source.key)
				)
					throw new Error("Operation lacks host-attested operator source authority");
				if (
					!batch.review.literalAcceptance &&
					!operation.evidence.some(
						span => span.sourceKey === source.key && !source.adoptedUnitIds?.includes(span.unitId),
					)
				)
					throw new Error("Operation only repeats an already adopted source unit");
				if (operation.kind === "add" && operation.requirementId)
					throw new Error("New requirement IDs are host assigned at publication");
				if (
					operation.kind !== "add" &&
					!snapshot.revisions.some(item => item.requirementId === operation.requirementId)
				)
					throw new Error("Correction has no host-resolved requirement target");
				for (const span of [
					...operation.evidence,
					...(operation.referents ?? []),
					...(operation.relations ?? []).flatMap(relation => relation.evidence),
				]) {
					validateSpan(this.#sources, span);
					if (
						verifiedIntegrities[span.sourceKey] !== span.integrity ||
						batch.readSourceIntegrities[span.sourceKey] !== span.integrity
					)
						throw new Error("Evidence was not part of the verified frozen review input");
				}
				for (const id of operation.predecessorRevisionIds) {
					const predecessor = snapshot.revisions.find(item => item.id === id);
					if (
						!predecessor ||
						predecessor.requirementId !== operation.requirementId ||
						predecessor.lifecycle !== "accepted" ||
						!batch.readHeads[predecessor.requirementId]?.includes(predecessor.id)
					)
						throw new Error("Unresolved predecessor revision");
					if (!hasSourceRelation(snapshot, predecessor.sourceKey, source.key, operation.relations, this.readSource, this.readRelations))
						throw new Error("Predecessor lacks host-supported source authority");
				}
				for (const relation of operation.relations ?? []) {
					if (!relation.evidence.length || relation.predecessorSourceKey === relation.successorSourceKey)
						throw new Error("Unsupported source relation");
					for (const key of [relation.predecessorSourceKey, relation.successorSourceKey]) {
						const endpoint = sourceAt(this.#sources, key);
						if (
							endpoint.referenceOnly ||
							endpoint.integrityAvailable === false ||
							endpoint.state === "orphaned" ||
							verifiedIntegrities[key] !== endpoint.integrity ||
							batch.readSourceIntegrities[key] !== endpoint.integrity
						)
							throw new Error("Source relation endpoint was not verified in frozen evidence");
					}
					if (
						hasSourceRelation(
							snapshot,
							relation.successorSourceKey,
							relation.predecessorSourceKey,
							batch.operations.flatMap(item => item.relations ?? []),
							this.readSource,
							this.readRelations,
						)
					)
						throw new Error("Cyclic source relation");
				}
			}
		} catch (error) {
			return result("rejected", error instanceof Error ? error.message : String(error));
		}
		const publicationRevision = ++snapshot.state.publicationRevision;
		for (const operation of batch.operations)
			snapshot.revisions.push({
				...structuredClone(operation),
				id: randomUUID(),
				requirementId: operation.requirementId ?? randomUUID(),
				sourceKey: source.key,
				sourceIntegrity: source.integrity,
				batchId,
				publicationRevision,
				lifecycle: "accepted",
			});
		batch.status = "accepted";
		if (batch.review.literalAcceptance) {
			source.adoptedUnitIds = [...(source.adoptedUnitIds ?? []), ...batch.review.literalAcceptance.unitIds];
		}
		const complete =
			!batch.review.literalAcceptance || source.units.every(unit => source.adoptedUnitIds?.includes(unit.id));
		source.state = complete ? "complete" : source.gap ? "gap" : "pending";
		if (complete) {
			delete source.gap;
			delete source.reason;
		} else source.reason = "Literal unit accepted; remaining source units require coverage review";
		return result("accepted");
	}
	reconcileRequirementsSources(observations: RequirementsObservation[]): void {
		let changed = false;
		const affected = new Set<string>();
		for (const observation of observations) {
			const source = this.#sources.get(observation.key);
			if (!source) continue;
			if (observation.locators) source.locators = structuredClone(observation.locators);
			if (observation.integrity === source.integrity && source.integrityAvailable !== false) continue;
			if (observation.integrity === null && source.integrityAvailable === false) continue;
			changed = true; affected.add(source.key);
			source.integrityAvailable = observation.integrity !== null;
			if (observation.integrity && observation.integrity !== source.integrity) { source.integrity = observation.integrity; delete source.adoptedUnitIds; }
			if (observation.units) source.units = structuredClone(observation.units);
			if (source.gap?.integrity !== source.integrity) delete source.gap;
			source.state = source.gap ? "gap" : observation.integrity ? "pending" : "orphaned";
			source.reason = observation.integrity ? "Original available; current coverage must be checked" : "Original source unavailable";
		}
		const availability = (key: string, integrity: string): RequirementsRevision["availability"] => {
			const source = this.#sources.get(key);
			return !source || source.integrityAvailable === false ? "unavailable" : source.integrity !== integrity ? "changed" : "available";
		};
		for (const revision of this.snapshot.revisions) {
			const evidence = [...revision.evidence, ...(revision.referents ?? []), ...(revision.relations ?? []).flatMap(item => item.evidence)];
			if (!affected.has(revision.sourceKey) && !evidence.some(item => affected.has(item.sourceKey))) continue;
			const states = [availability(revision.sourceKey, revision.sourceIntegrity), ...evidence.map(item => availability(item.sourceKey, item.integrity))];
			revision.availability = states.includes("unavailable") ? "unavailable" : states.includes("changed") ? "changed" : "available";
			// Original availability never rewrites explicit operator lifecycle.
			const dependent = this.#sources.get(revision.sourceKey);
			if (dependent?.state === "complete" && revision.availability !== "available") { dependent.state = "pending"; dependent.reason = "Referenced original evidence is unavailable or changed"; }
		}
		for (const batch of this.snapshot.batches) {
			if (!affected.has(batch.sourceKey) && !Object.keys(batch.readSourceIntegrities).some(key => affected.has(key))) continue;
			if (batch.status !== "accepted") { batch.status = "stale"; batch.reason = "Frozen source evidence changed or became unavailable"; continue; }
			const source = this.#sources.get(batch.sourceKey);
			if (source && availability(source.key, batch.sourceIntegrity) === "available" && (!batch.review.literalAcceptance || source.units.every(unit => source.adoptedUnitIds?.includes(unit.id))) && Object.entries(batch.readSourceIntegrities).every(([key, integrity]) => availability(key, integrity) === "available")) { source.state = "complete"; delete source.reason; }
		}
		if (changed) this.snapshot.state.publicationRevision++;
	}
	quarantineRequirements(revisionIds: string[], actor: string, reason: string): number {
		if (!actor || !reason) throw new Error("Quarantine requires actor and reason");
		const revisions = revisionIds.map(id => {
			const revision = this.snapshot.revisions.find(item => item.id === id);
			if (!revision) throw new Error(`Unknown revision: ${id}`);
			return revision;
		});
		const generation = ++this.snapshot.state.generation;
		for (const revision of revisions)
			if (revision.lifecycle === "accepted") {
				revision.lifecycle = "quarantined";
				revision.quarantine = { actor, reason, generation };
			}
		this.snapshot.state.publicationRevision++;
		return generation;
	}
	restoreRequirements(receipt: RequirementsRestoreReceipt): boolean {
		if (this.snapshot.state.restoreReceipts.some(item => item.id === receipt.id)) return true;
		if (!receipt.actor || !receipt.revisionIds.length) return false;
		const revisions = receipt.revisionIds.map(id => this.snapshot.revisions.find(item => item.id === id));
		try {
			if (
				receipt.generation !== this.snapshot.state.generation ||
				!readHeadsMatch(this.snapshot, receipt.readHeads)
			)
				throw new Error("Restore authority or publication baseline changed");
			for (const revision of revisions) {
				if (
					!revision ||
					revision.lifecycle !== "quarantined" ||
					sourceAt(this.#sources, revision.sourceKey).integrity !== revision.sourceIntegrity ||
					receipt.sourceIntegrities[revision.sourceKey] !== revision.sourceIntegrity
				)
					throw new Error("Suspended revision or original evidence is not current");
				validateScope(revision.scope);
				for (const span of [
					...revision.evidence,
					...(revision.referents ?? []),
					...(revision.relations ?? []).flatMap(relation => relation.evidence),
				]) {
					validateSpan(this.#sources, span);
					if (receipt.sourceIntegrities[span.sourceKey] !== span.integrity)
						throw new Error("Restore evidence was not verified");
				}
			}
			validateReview(this.#sources, receipt.review, revisions as RequirementsRevision[], receipt.operationIds, revisions[0]!.sourceKey, this.snapshot.revisions, receipt.readHeads);
		} catch (error) {
			for (const revision of revisions)
				if (revision)
					this.snapshot.state.restoreReviews[revision.id] = {
						receipt: structuredClone(receipt),
						reason: error instanceof Error ? error.message : String(error),
					};
			return false;
		}
		for (const revision of revisions)
			if (revision) {
				revision.lifecycle = "accepted";
				revision.availability = "available";
				delete revision.quarantine;
			}
		this.snapshot.state.restoreReceipts.push(structuredClone(receipt));
		this.snapshot.state.publicationRevision++;
		return true;
	}
}

function hasSourceRelation(
	snapshot: RequirementsSnapshot,
	predecessor: string,
	successor: string,
	extra: RequirementsRevision["relations"] = [],
	readSource: (key: string) => RequirementsSource | undefined = key => snapshot.sources.find(item => item.key === key),
	readRelations?: (key: string) => RequirementsRelation[],
): boolean {
	const relations = [
		...snapshot.revisions.filter(item => item.lifecycle === "accepted" && (!item.availability || item.availability === "available")).flatMap(item => item.relations ?? []),
		...(extra ?? []),
	];
	const seen = new Set<string>();
	const pending = [successor];
	while (pending.length) {
		const key = pending.pop()!;
		if (seen.has(key)) continue;
		seen.add(key);
		const source = readSource(key);
		if (!source || source.integrityAvailable === false || source.state === "orphaned") continue;
		if (key === predecessor) return true;
		const parent = source.parentKey ? readSource(source.parentKey) : undefined;
		if (parent && parent.ownerSessionId === source.ownerSessionId && parent.epoch === source.epoch)
			pending.push(parent.key);
		for (const relation of relations)
			if (relation.successorSourceKey === key) pending.push(relation.predecessorSourceKey);
		for (const relation of readRelations?.(key) ?? []) pending.push(relation.predecessorSourceKey);
	}
	return false;
}

function requirementsScopeApplies(
	revision: RequirementsRevision,
	context: Omit<RequirementsConsumptionContext, "branchId">,
): boolean {
	const scope = revision.scope;
	return (
		scope.kind === "global" ||
		(scope.kind === "project" && scope.projectId === context.projectId) ||
		((scope.kind === "session" || scope.kind === "task") &&
			scope.sessionId === context.sessionId && scope.epoch === context.epoch &&
			(!context.sourceKeys || context.sourceKeys.has(revision.sourceKey)))
	);
}

function getApplicableRequirementHeads(
	snapshot: RequirementsSnapshot,
	context: Omit<RequirementsConsumptionContext, "branchId">,
	readSource: (key: string) => RequirementsSource | undefined,
): RequirementsRevision[][] {
	const groups = new Map<string, RequirementsRevision[]>();
	for (const revision of snapshot.revisions) {
		if (!requirementsScopeApplies(revision, context)) continue;
		const group = groups.get(revision.requirementId) ?? [];
		group.push(revision);
		groups.set(revision.requirementId, group);
	}
	const heads: RequirementsRevision[][] = [];
	for (const group of groups.values()) {
		// Suspended/historical successors suppress old heads: quarantine is not rollback.
		heads.push(group.length === 1 ? group : group.filter(revision =>
			!group.some(other => other.id !== revision.id &&
				(other.predecessorRevisionIds.includes(revision.id) ||
					hasSourceRelation(snapshot, revision.sourceKey, other.sourceKey, other.relations, readSource))),
		));
	}
	return heads;
}

function applicableHeads(heads: RequirementsRevision[][]): Pick<RequirementsApplicable, "active" | "conflicts"> {
	const active: RequirementsRevision[] = [];
	const conflicts: RequirementsRevision[][] = [];
	for (const group of heads) {
		if (group.length > 1) conflicts.push(group);
		else if (group.length === 1 && group[0].lifecycle === "accepted" && (!group[0].availability || group[0].availability === "available") && group[0].kind !== "withdraw")
			active.push(group[0]);
	}
	return { active, conflicts };
}

export function getApplicableRequirements(
	snapshot: RequirementsSnapshot,
	context: { projectId?: string; sessionId: string; epoch: number; sourceKeys?: ReadonlySet<string> },
): RequirementsApplicable {
	const associatedSources = new Set<string>();
	for (const revision of snapshot.revisions) {
		if (!requirementsScopeApplies(revision, context)) continue;
		associatedSources.add(revision.sourceKey);
		for (const span of [...revision.evidence, ...(revision.referents ?? [])]) associatedSources.add(span.sourceKey);
		for (const relation of revision.relations ?? []) {
			associatedSources.add(relation.predecessorSourceKey);
			associatedSources.add(relation.successorSourceKey);
			for (const span of relation.evidence) associatedSources.add(span.sourceKey);
		}
	}
	const sourcesByKey = new Map(snapshot.sources.map(source => [source.key, source]));
	for (const key of associatedSources) {
		const parent = sourcesByKey.get(key)?.parentKey;
		if (parent) associatedSources.add(parent);
	}
	return {
		...applicableHeads(getApplicableRequirementHeads(snapshot, context, key => sourcesByKey.get(key))),
		coverageGaps: snapshot.sources.filter(source => !source.referenceOnly && source.state !== "complete" &&
			(!context.sourceKeys || context.sourceKeys.has(source.key) || associatedSources.has(source.key))),
	};
}

/** Storage-owner queries; every normal transaction supplies a bounded affected-row set. */
export interface RequirementsBackend {
	transaction<T>(run: () => T): T;
	state(ownerSessionId?: string, receiptId?: string, revisionIds?: readonly string[], scopes?: readonly RequirementsScope[]): RequirementsSnapshot["state"];
	writeState(before: RequirementsSnapshot["state"], after: RequirementsSnapshot["state"]): void;
	source(key: string): RequirementsSource | undefined;
	batch(id: string): RequirementsBatch | undefined;
	revision(id: string): RequirementsRevision | undefined;
	revisionIdsForBatch(batchId: string): string[];
	relationsForSuccessor(key: string): RequirementsRelation[];
	putSource(source: RequirementsSource): void;
	putBatch(batch: RequirementsBatch): void;
	putRevision(revision: RequirementsRevision): void;
	deleteBatch(id: string): void;
	batchIdsForSource(key: string): string[];
	dependentIds(key: string, kind: "revision" | "batch"): string[];
	headIds(context?: RequirementsConsumptionContext, requirementId?: string): string[];
	setHeads(requirementId: string, revisions: RequirementsRevision[]): void;
	pending(context?: RequirementsConsumptionContext): RequirementsSource[];
	coverage(keys?: ReadonlySet<string>): RequirementsCoverageSummary;
	snapshot(): RequirementsSnapshot;
}
function sameScope(a: RequirementsScope, b: RequirementsScope): boolean {
	if (a.kind !== b.kind) return false;
	if (a.kind === "global") return true;
	if (a.kind === "project") return a.projectId === b.projectId;
	return a.sessionId === b.sessionId && a.epoch === b.epoch;
}
function readHeadsMatch(snapshot: RequirementsSnapshot, readHeads: Record<string, string[]>): boolean {
	if (!readHeads || typeof readHeads !== "object") return false;
	for (const [id, expected] of Object.entries(readHeads)) {
		if (!Array.isArray(expected)) return false;
		const rows = snapshot.revisions.filter(item => item.requirementId === id);
		const heads = rows.filter(row => !rows.some(other => other.id !== row.id && sameScope(other.scope, row.scope) &&
			(other.predecessorRevisionIds.includes(row.id) || hasSourceRelation(snapshot, row.sourceKey, other.sourceKey, other.relations))));
		if (heads.length !== expected.length || heads.some(row => !expected.includes(row.id))) return false;
	}
	return true;
}
interface RequirementsWork {
	sourceKeys?: Iterable<string>;
	batchIds?: Iterable<string>;
	revisionIds?: Iterable<string>;
	requirementIds?: Iterable<string>;
	ownerSessionId?: string;
	receiptId?: string;
	dependencyKeys?: Iterable<string>;
	scopes?: readonly RequirementsScope[];
}

/** Requirements-specific ephemeral backend. No SQLite owner or ordinary memory is opened. */
class InMemoryRequirementsBackend implements RequirementsBackend {
	readonly #sources = new Map<string, RequirementsSource>();
	readonly #batches = new Map<string, RequirementsBatch>();
	readonly #revisions = new Map<string, RequirementsRevision>();
	readonly #batchRevisions = new Map<string, Set<string>>();
	readonly #relationRevisions = new Map<string, Set<string>>();
	readonly #heads = new Map<string, Set<string>>();
	readonly #scopeHeads = new Map<string, Set<string>>();
	readonly #batchSources = new Map<string, Set<string>>();
	readonly #dependencies = new Map<string, Set<string>>();
	readonly #pending = new Map<string, Set<string>>();
	readonly #coverage = createEmptyRequirementsCoverageSummary();
	#state: RequirementsSnapshot["state"];
	constructor(snapshot = createEmptyRequirementsSnapshot()) {
		this.#state = structuredClone(snapshot.state);
		for (const source of snapshot.sources) this.putSource(source);
		for (const batch of snapshot.batches) this.putBatch(batch);
		const groups = new Map<string, RequirementsRevision[]>();
		for (const revision of snapshot.revisions) {
			this.putRevision(revision);
			const rows = groups.get(revision.requirementId) ?? [];
			rows.push(revision); groups.set(revision.requirementId, rows);
		}
		for (const [id, rows] of groups) this.setHeads(id, rows.filter(row => !rows.some(other => other.id !== row.id && sameScope(other.scope, row.scope) &&
			(other.predecessorRevisionIds.includes(row.id) || hasSourceRelation(snapshot, row.sourceKey, other.sourceKey, other.relations)))));
	}
	transaction<T>(run: () => T): T { return run(); }
	state(ownerSessionId?: string, receiptId?: string, revisionIds: readonly string[] = [], scopes: readonly RequirementsScope[] = []): RequirementsSnapshot["state"] {
		return structuredClone({ publicationRevision: this.#state.publicationRevision, generation: this.#state.generation,
			owners: ownerSessionId && this.#state.owners[ownerSessionId] ? { [ownerSessionId]: this.#state.owners[ownerSessionId] } : {},
			restoreReceipts: receiptId ? this.#state.restoreReceipts.filter(item => item.id === receiptId) : [],
			restoreReviews: Object.fromEntries(revisionIds.filter(id => this.#state.restoreReviews[id]).map(id => [id, this.#state.restoreReviews[id]])),
			tombstones: scopes.flatMap(scope => (this.#state.tombstones ?? []).filter(item => sameScope(item.scope, scope))),
		});
	}
	writeState(before: RequirementsSnapshot["state"], after: RequirementsSnapshot["state"]): void {
		this.#state.publicationRevision = after.publicationRevision; this.#state.generation = after.generation;
		this.#state.tombstones ??= [];
		for (const tombstone of after.tombstones ?? []) {
			const index = this.#state.tombstones.findIndex(item => sameScope(item.scope, tombstone.scope));
			if (index < 0) this.#state.tombstones.push(structuredClone(tombstone));
			else this.#state.tombstones[index] = structuredClone(tombstone);
		}
		for (const id of Object.keys(before.owners)) if (!after.owners[id]) delete this.#state.owners[id];
		Object.assign(this.#state.owners, structuredClone(after.owners));
		for (const receipt of after.restoreReceipts) if (!this.#state.restoreReceipts.some(item => item.id === receipt.id)) this.#state.restoreReceipts.push(structuredClone(receipt));
		Object.assign(this.#state.restoreReviews, structuredClone(after.restoreReviews));
	}
	source(key: string) { const row = this.#sources.get(key); return row ? structuredClone(row) : undefined; }
	batch(id: string) { const row = this.#batches.get(id); return row ? structuredClone(row) : undefined; }
	revision(id: string) { const row = this.#revisions.get(id); return row ? structuredClone(row) : undefined; }
	#index(index: Map<string, Set<string>>, key: string, id: string, add: boolean): void {
		if (!add) { index.get(key)?.delete(id); return; }
		let ids = index.get(key); if (!ids) index.set(key, ids = new Set()); ids.add(id);
	}
	revisionIdsForBatch(batchId: string): string[] { return [...(this.#batchRevisions.get(batchId) ?? [])]; }
	putSource(source: RequirementsSource): void {
		const old = this.#sources.get(source.key);
		if (old) { if (old.referenceOnly) this.#coverage.referenceOnly--; else this.#coverage.byState[old.state]--; this.#index(this.#pending, old.ownerSessionId, old.key, false); }
		else this.#coverage.total++;
		if (source.referenceOnly) this.#coverage.referenceOnly++; else this.#coverage.byState[source.state]++;
		this.#sources.set(source.key, structuredClone(source));
		if (!source.referenceOnly && source.state !== "complete" && source.origin.kind === "human") this.#index(this.#pending, source.ownerSessionId, source.key, true);
	}
	#edges(row: RequirementsBatch | RequirementsRevision, add: boolean): void {
		const keys = new Set([row.sourceKey]);
		if ("operations" in row) { for (const key of Object.keys(row.readSourceIntegrities)) keys.add(key); for (const operation of row.operations) addOperationSourceKeys(operation, keys); }
		else addOperationSourceKeys(row, keys);
		for (const key of keys) this.#index(this.#dependencies, ("operations" in row ? "batch" : "revision") + key, row.id, add);
	}
	putBatch(batch: RequirementsBatch): void { const old = this.#batches.get(batch.id); if (old) this.#edges(old, false); this.#batches.set(batch.id, structuredClone(batch)); this.#index(this.#batchSources, batch.sourceKey, batch.id, true); this.#edges(batch, true); }
	putRevision(revision: RequirementsRevision): void {
		const old = this.#revisions.get(revision.id);
		if (old) { this.#edges(old, false); this.#index(this.#batchRevisions, old.batchId, old.id, false); for (const relation of old.relations ?? []) this.#index(this.#relationRevisions, relation.successorSourceKey, old.id, false); }
		this.#revisions.set(revision.id, structuredClone(revision)); this.#index(this.#batchRevisions, revision.batchId, revision.id, true); this.#edges(revision, true);
		if (revision.lifecycle === "accepted" && (!revision.availability || revision.availability === "available")) for (const relation of revision.relations ?? []) this.#index(this.#relationRevisions, relation.successorSourceKey, revision.id, true);
	}
	relationsForSuccessor(key: string): RequirementsRelation[] {
		return structuredClone([...(this.#relationRevisions.get(key) ?? [])].flatMap(id => this.#revisions.get(id)!.relations?.filter(relation => relation.successorSourceKey === key) ?? []));
	}
	deleteBatch(id: string): void { const old = this.#batches.get(id); if (!old) return; this.#edges(old, false); this.#index(this.#batchSources, old.sourceKey, id, false); this.#batches.delete(id); }
	batchIdsForSource(key: string) { return [...(this.#batchSources.get(key) ?? [])]; }
	dependentIds(key: string, kind: "revision" | "batch") { return [...(this.#dependencies.get(kind + key) ?? [])]; }
	#scope(scope: RequirementsScope): string { return JSON.stringify([scope.kind, scope.projectId ?? null, scope.sessionId ?? null, scope.epoch ?? null]); }
	headIds(context?: RequirementsConsumptionContext, requirementId?: string): string[] {
		if (requirementId) return [...(this.#heads.get(requirementId) ?? [])];
		if (!context) return [...this.#heads.values()].flatMap(ids => [...ids]);
		const scopes: RequirementsScope[] = [{ kind: "global" }, { kind: "project", projectId: context.projectId }, { kind: "session", sessionId: context.sessionId, epoch: context.epoch }, { kind: "task", sessionId: context.sessionId, epoch: context.epoch }];
		return scopes.flatMap(scope => [...(this.#scopeHeads.get(this.#scope(scope)) ?? [])]).filter(id => requirementsScopeApplies(this.#revisions.get(id)!, context));
	}
	setHeads(requirementId: string, rows: RequirementsRevision[]): void {
		for (const id of this.#heads.get(requirementId) ?? []) { const row = this.#revisions.get(id)!; this.#index(this.#scopeHeads, this.#scope(row.scope), id, false); }
		this.#heads.set(requirementId, new Set(rows.map(row => row.id)));
		for (const row of rows) this.#index(this.#scopeHeads, this.#scope(row.scope), row.id, true);
	}
	pending(context?: RequirementsConsumptionContext): RequirementsSource[] {
		const ids = context ? this.#pending.get(context.sessionId) ?? [] : [...this.#pending.values()].flatMap(ids => [...ids]);
		return [...ids].map(id => this.source(id)!).filter(source => !context || (source.epoch === context.epoch && (!context.sourceKeys || context.sourceKeys.has(source.key))));
	}
	coverage(keys?: ReadonlySet<string>): RequirementsCoverageSummary {
		if (!keys) return structuredClone(this.#coverage);
		const summary = createEmptyRequirementsCoverageSummary();
		for (const key of keys) { const source = this.#sources.get(key); if (!source) continue; summary.total++; if (source.referenceOnly) summary.referenceOnly++; else summary.byState[source.state]++; }
		return summary;
	}
	snapshot(): RequirementsSnapshot { return structuredClone({ state: this.#state, sources: [...this.#sources.values()], batches: [...this.#batches.values()], revisions: [...this.#revisions.values()] }); }
}

export class RequirementsStore {
	readonly #backend: RequirementsBackend;
	constructor(backend: RequirementsBackend | RequirementsSnapshot = createEmptyRequirementsSnapshot()) {
		this.#backend = "transaction" in backend ? backend : new InMemoryRequirementsBackend(backend);
	}
	getRequirementsState() { return this.#backend.state(); }
	getRequirementsSnapshot() { return this.#backend.transaction(() => this.#backend.snapshot()); }
	getRequirementsSource(key: string) { return this.#backend.source(key); }
	getRequirementsBatch(id: string) { return this.#backend.batch(id); }
	getRequirementsLatestBatch(sourceKey: string) {
		const ids = this.#backend.batchIdsForSource(sourceKey);
		return ids.length ? this.#backend.batch(ids[ids.length - 1]) : undefined;
	}
	getRequirementsRevision(id: string) { return this.#backend.revision(id); }
	getRequirementsSourceMetadata(key: string): RequirementsSourceMetadata | undefined {
		const source = this.#backend.source(key); if (!source) return undefined;
		return { key, integrity: source.integrity, integrityAvailable: source.integrityAvailable, state: source.state, locators: source.locators };
	}
	getRequirementsCoverageSummary(keys?: ReadonlySet<string>) { return this.#backend.coverage(keys); }
	getRequirementsPendingSources(context?: RequirementsConsumptionContext) { return this.#backend.pending(context); }
	#load(work: RequirementsWork): RequirementsSnapshot {
		const sourceKeys = new Set(work.sourceKeys), batchIds = new Set(work.batchIds), revisionIds = new Set(work.revisionIds);
		for (const id of work.requirementIds ?? []) for (const head of this.#backend.headIds(undefined, id)) revisionIds.add(head);
		for (const key of work.dependencyKeys ?? []) {
			for (const id of this.#backend.dependentIds(key, "revision")) revisionIds.add(id);
			for (const id of this.#backend.dependentIds(key, "batch")) batchIds.add(id);
		}
		const snapshot = createEmptyRequirementsSnapshot();
		for (const id of batchIds) {
			const batch = this.#backend.batch(id); if (!batch) continue; snapshot.batches.push(batch); sourceKeys.add(batch.sourceKey);
			if (batch.status === "accepted") for (const revisionId of this.#backend.revisionIdsForBatch(batch.id)) revisionIds.add(revisionId);
			for (const key of Object.keys(batch.readSourceIntegrities)) sourceKeys.add(key);
			for (const requirementId of Object.keys(batch.readHeads ?? {})) for (const head of this.#backend.headIds(undefined, requirementId)) revisionIds.add(head);
			for (const operation of batch.operations) { addOperationSourceKeys(operation, sourceKeys); for (const predecessor of operation.predecessorRevisionIds) revisionIds.add(predecessor); }
		}
		for (const id of revisionIds) {
			const revision = this.#backend.revision(id); if (!revision) continue; snapshot.revisions.push(revision);
			sourceKeys.add(revision.sourceKey); addOperationSourceKeys(revision, sourceKeys);
		}
		for (const key of sourceKeys) {
			const source = this.#backend.source(key); if (!source) continue; snapshot.sources.push(source);
		}
		const scopes = [...(work.scopes ?? []), ...snapshot.batches.flatMap(batch => batch.operations.map(operation => operation.scope)), ...snapshot.revisions.map(revision => revision.scope)];
		snapshot.state = this.#backend.state(work.ownerSessionId, work.receiptId, [...revisionIds], scopes);
		return snapshot;
	}
	#mutate<T>(work: RequirementsWork, apply: (store: RequirementsReducer) => T): T {
		return this.#backend.transaction(() => {
			const snapshot = this.#load(work);
			const before = structuredClone(snapshot);
			const result = apply(new RequirementsReducer(snapshot, key => this.#backend.source(key), key => this.#backend.relationsForSuccessor(key)));
			const previousSources = new Map(before.sources.map(row => [row.key, JSON.stringify(row)]));
			const previousBatches = new Map(before.batches.map(row => [row.id, JSON.stringify(row)]));
			const previousRevisions = new Map(before.revisions.map(row => [row.id, JSON.stringify(row)]));
			for (const source of snapshot.sources) if (previousSources.get(source.key) !== JSON.stringify(source)) this.#backend.putSource(source);
			for (const batch of snapshot.batches) { if (previousBatches.get(batch.id) !== JSON.stringify(batch)) this.#backend.putBatch(batch); previousBatches.delete(batch.id); }
			for (const id of previousBatches.keys()) this.#backend.deleteBatch(id);
			const changedRequirements = new Set<string>();
			for (const revision of snapshot.revisions) if (previousRevisions.get(revision.id) !== JSON.stringify(revision)) { this.#backend.putRevision(revision); changedRequirements.add(revision.requirementId); }
			for (const id of changedRequirements) {
				const rows = new Map(this.#backend.headIds(undefined, id).map(head => [head, this.#backend.revision(head)!]));
				for (const revision of snapshot.revisions) if (revision.requirementId === id) rows.set(revision.id, revision);
				const candidates = [...rows.values()];
				this.#backend.setHeads(id, candidates.filter(row => !candidates.some(other => other.id !== row.id && sameScope(row.scope, other.scope) &&
					(other.predecessorRevisionIds.includes(row.id) || hasSourceRelation(snapshot, row.sourceKey, other.sourceKey, other.relations, key => this.#backend.source(key), key => this.#backend.relationsForSuccessor(key))))));
			}
			if (JSON.stringify(before.state) !== JSON.stringify(snapshot.state)) this.#backend.writeState(before.state, snapshot.state);
			return result;
		});
	}
	getRequirementsConsumptionSnapshot(context: RequirementsConsumptionContext): RequirementsConsumptionSnapshot {
		return this.#backend.transaction(() => {
			const headIds = this.#backend.headIds(context);
			const snapshot = this.#load({ revisionIds: headIds, ownerSessionId: context.sessionId });
			const heads = new Map<string, RequirementsRevision[]>();
			for (const id of headIds) { const row = snapshot.revisions.find(row => row.id === id)!; const group = heads.get(row.requirementId) ?? []; group.push(row); heads.set(row.requirementId, group); }
			const groups = [...heads.values()];
			return { ...snapshot, readHeads: Object.fromEntries([...heads].map(([id, rows]) => [id, rows.map(row => row.id)])),
				coverage: this.#backend.coverage(), pendingSources: [],
				applicable: { ...applicableHeads(groups), coverageGaps: snapshot.sources.filter(source => !source.referenceOnly && source.state !== "complete") },
				applicableRevisionIds: groups.filter(rows => rows.length === 1 && rows[0].lifecycle === "accepted" && (!rows[0].availability || rows[0].availability === "available")).map(rows => rows[0].id),
			};
		});
	}
	authorizeRequirementsOwner(authority: RequirementsAuthority): void { this.#mutate({ ownerSessionId: authority.ownerSessionId }, store => store.authorizeRequirementsOwner(authority)); }
	invalidateRequirementsOwner(ownerSessionId: string): void { this.#mutate({ ownerSessionId }, store => store.invalidateRequirementsOwner(ownerSessionId)); }
	intakeRequirementsSource(source: RequirementsSource): void { this.intakeRequirementsSources([source]); }
	intakeRequirementsSources(sources: readonly RequirementsSource[]): void {
		this.#backend.transaction(() => {
			const sourceKeys = new Set<string>(), dependencyKeys = new Set<string>();
			for (const source of sources) {
				sourceKeys.add(source.key);
				if (source.units.length === 0 && source.integrityAvailable === false) continue;
				const existing = this.#backend.source(source.key);
				// Unchanged bytes and locator/manifest hydration do not consume old batches
				// or their potentially journal-wide frozen read sets.
				if (existing && (existing.integrity !== source.integrity || existing.integrityAvailable === false))
					dependencyKeys.add(source.key);
			}
			this.#mutate({ sourceKeys, dependencyKeys }, store => store.intakeRequirementsSources(sources));
		});
	}
	setRequirementsSourceDisposition(key: string, integrity: string, state: Exclude<RequirementsSourceState, "complete" | "gap">, reason: string): void { this.#mutate({ sourceKeys: [key] }, store => store.setRequirementsSourceDisposition(key, integrity, state, reason)); }
	recordRequirementsGap(key: string, integrity: string, actor: string, reason: string): void { this.#mutate({ sourceKeys: [key] }, store => store.recordRequirementsGap(key, integrity, actor, reason)); }
	saveRequirementsBatch(batch: RequirementsBatch): string { return this.#mutate({ sourceKeys: [batch.sourceKey], batchIds: this.#backend.batchIdsForSource(batch.sourceKey) }, store => store.saveRequirementsBatch(batch)); }
	publishRequirementsBatch(batchId: string, authority: RequirementsAuthority, verifiedIntegrities: Record<string, string>): RequirementsPublicationResult { return this.#mutate({ batchIds: [batchId], ownerSessionId: authority.ownerSessionId }, store => store.publishRequirementsBatch(batchId, authority, verifiedIntegrities)); }
	reconcileRequirementsSources(observations: RequirementsObservation[]): void {
		const latest = new Map<string, RequirementsObservation>();
		for (const observation of observations) latest.set(observation.key, { ...latest.get(observation.key), ...observation });
		const changed = [...latest.values()].filter(observation => { const source = this.#backend.source(observation.key); return source && !(observation.integrity === source.integrity && source.integrityAvailable !== false && (!observation.locators || JSON.stringify(source.locators) === JSON.stringify(observation.locators))) && !(observation.integrity === null && source.integrityAvailable === false && !observation.locators); });
		if (changed.length) {
			const sourceKeys = changed.map(row => row.key);
			this.#mutate({ sourceKeys, dependencyKeys: sourceKeys }, store => store.reconcileRequirementsSources(changed));
		}
	}
	quarantineRequirements(revisionIds: string[], actor: string, reason: string): number { return this.#mutate({ revisionIds }, store => store.quarantineRequirements(revisionIds, actor, reason)); }
	restoreRequirements(receipt: RequirementsRestoreReceipt): boolean { return this.#mutate({ revisionIds: receipt.revisionIds, requirementIds: Object.keys(receipt.readHeads ?? {}), receiptId: receipt.id }, store => store.restoreRequirements(receipt)); }
	withdrawRequirements(revisionIds: string[], actor: string, reason: string): number {
		if (!actor || !reason) throw new Error("Withdrawal requires actor and reason");
		return this.#mutate({ revisionIds }, store => {
			const revisions = revisionIds.map(id => { const revision = store.snapshot.revisions.find(row => row.id === id); if (!revision) throw new Error(`Unknown revision: ${id}`); return revision; });
			const generation = ++store.snapshot.state.generation;
			for (const revision of revisions) { revision.lifecycle = "historical"; revision.withdrawal = { actor, reason, generation }; }
			store.snapshot.state.publicationRevision++;
			return generation;
		});
	}
	clearRequirements(scope: RequirementsScope, actor: string, reason: string): number {
		validateScope(scope); if (!actor || !reason) throw new Error("Scoped clear requires actor and reason");
		const context = { projectId: scope.projectId, sessionId: scope.sessionId ?? "", epoch: scope.epoch ?? 0, branchId: "" };
		const ids = this.#backend.headIds(context).filter(id => sameScope(this.#backend.revision(id)!.scope, scope));
		return this.#mutate({ revisionIds: ids, scopes: [scope] }, store => {
			const generation = ++store.snapshot.state.generation;
			for (const revision of store.snapshot.revisions) if (sameScope(revision.scope, scope)) { revision.lifecycle = "historical"; revision.quarantine = { actor, reason, generation }; }
			store.snapshot.state.tombstones ??= [];
			store.snapshot.state.tombstones = store.snapshot.state.tombstones.filter(item => !sameScope(item.scope, scope));
			store.snapshot.state.tombstones.push({ scope, actor, reason, generation }); store.snapshot.state.publicationRevision++;
			return generation;
		});
	}
}
