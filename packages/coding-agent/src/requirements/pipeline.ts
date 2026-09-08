import { createHash, randomUUID } from "node:crypto";
import { completeSimple, type ImageContent, type Model, retryTransientCompletion } from "@oh-my-pi/pi-ai";
import { clampThinkingLevelForModel } from "@oh-my-pi/pi-catalog/model-thinking";
import type { ModelRegistry } from "../config/model-registry";
import type { Settings } from "../config/settings";
import evidencePrompt from "../prompts/requirements-evidence.md" with { type: "text" };
import extractionPrompt from "../prompts/requirements-extraction.md" with { type: "text" };
import sanityPrompt from "../prompts/requirements-sanity.md" with { type: "text" };
import { resolveRoleModelFull } from "../session/role-models";
import type { SessionManager } from "../session/session-manager";
import type { ResolvedRequirementsSource } from "./source-capture";
import type {
	RequirementsAuthority,
	RequirementsBatch,
	RequirementsEvidence,
	RequirementsOperation,
	RequirementsReview,
	RequirementsRevision,
	RequirementsReviewOutcome,
	RequirementsScope,
} from "./types";

export const REQUIREMENTS_FORMAT = "requirements-v2-2";

export interface RequirementsPipelineHost {
	settings: Settings;
	modelRegistry: ModelRegistry;
	sessionManager: SessionManager;
	getModel(): Model | undefined;
}
export interface RequirementsEvidencePackage {
	source: ResolvedRequirementsSource;
	/** Complete host-resolved prior sources/referents, never model-selected excerpts. */
	references: ResolvedRequirementsSource[];
	/** Known revisions for targets/lineage, including inactive or quarantined records. */
	active: RequirementsRevision[];
	/** Host-selected settled accepted heads; withdrawal effects are not executable recall. */
	applicableRevisionIds: readonly string[];
	projectId: string;
	authority: RequirementsAuthority;
	publicationRevision: number;
	readHeads: Record<string, string[]>;
}
export interface RequirementsCandidate extends RequirementsOperation {
	id: string;
}
export interface RequirementsModelStatus {
	configured?: string;
	resolved?: string;
	reason?: string;
	capabilities?: string[];
}
export class RequirementsPipelineError extends Error {
	constructor(
		public readonly stage: "extractor" | "mechanical" | "evidence" | "sanity",
		message: string,
		public readonly unsupported = false,
	) {
		super(`${stage}: ${message}`);
	}
}
const roles = { extractor: "requirements", evidence: "requirementsEvidence", sanity: "requirementsSanity" } as const;
export function requirementsModelStatus(
	host: RequirementsPipelineHost,
): Record<keyof typeof roles, RequirementsModelStatus> {
	return Object.fromEntries(
		Object.entries(roles).map(([stage, role]) => {
			const configured = host.settings.getModelRole(role);
			const resolved = resolveRoleModelFull(host.settings, role, host.modelRegistry.getAll(), host.getModel());
			return [
				stage,
				{
					configured,
					resolved: resolved.model ? `${resolved.model.provider}/${resolved.model.id}` : undefined,
					capabilities: resolved.model?.input,
					reason: resolved.warning ?? (!resolved.model ? `Configure the ${role} model role` : undefined),
				},
			];
		}),
	) as Record<keyof typeof roles, RequirementsModelStatus>;
}

function object(value: unknown, label: string): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value))
		throw new RequirementsPipelineError("mechanical", `${label} must be an object`);
	return value as Record<string, unknown>;
}
function keys(value: Record<string, unknown>, allowed: string[], label: string): void {
	if (Object.keys(value).some(key => !allowed.includes(key)))
		throw new RequirementsPipelineError("mechanical", `${label} contains unsupported fields`);
}
function text(value: unknown, label: string): string {
	if (typeof value !== "string" || !value.trim())
		throw new RequirementsPipelineError("mechanical", `${label} must be nonempty text`);
	return value;
}
function array(value: unknown, label: string): unknown[] {
	if (!Array.isArray(value)) throw new RequirementsPipelineError("mechanical", `${label} must be an array`);
	return value;
}
function allSources(input: RequirementsEvidencePackage): ResolvedRequirementsSource[] {
	const found = new Map<string, ResolvedRequirementsSource>();
	const visit = (source: ResolvedRequirementsSource) => {
		if (found.has(source.source.key)) return;
		found.set(source.source.key, source);
		for (const reference of source.referents) visit(reference);
	};
	visit(input.source);
	for (const reference of input.references) visit(reference);
	return [...found.values()];
}
export function verifyRequirementsEvidence(value: unknown, input: RequirementsEvidencePackage): RequirementsEvidence {
	const span = object(value, "evidence");
	keys(span, ["sourceKey", "integrity", "unitId"], "evidence");
	const sourceKey = text(span.sourceKey, "sourceKey");
	const integrity = text(span.integrity, "integrity");
	const unitId = text(span.unitId, "unitId");
	const resolved = allSources(input).find(item => item.source.key === sourceKey);
	const descriptor = resolved?.source.units.find(unit => unit.id === unitId);
	const unit = resolved?.units.find(unit => unit.id === unitId);
	if (!resolved || resolved.source.integrity !== integrity || !descriptor || !unit || descriptor.unsupportedReason) {
		throw new RequirementsPipelineError("mechanical", "Unknown, stale or unsupported evidence unit");
	}
	const bytes =
		unit.text !== undefined
			? Buffer.from(unit.text, "utf8")
			: unit.image
				? Buffer.from(unit.image.data, "base64")
				: undefined;
	if (
		!bytes ||
		bytes.length !== descriptor.byteLength ||
		createHash("sha256").update(bytes).digest("hex") !== descriptor.sha256
	) {
		throw new RequirementsPipelineError("mechanical", "Original evidence bytes do not match the frozen manifest");
	}
	if ((unit.text !== undefined ? "text" : "image") !== descriptor.kind)
		throw new RequirementsPipelineError("mechanical", "Evidence kind differs from frozen manifest");
	return { sourceKey, integrity, unitId };
}
function scope(value: unknown, input: RequirementsEvidencePackage): RequirementsScope {
	const candidate = object(value, "scope");
	keys(candidate, ["kind", "projectId", "sessionId", "epoch"], "scope");
	if (candidate.kind === "global") {
		if (Object.keys(candidate).length !== 1)
			throw new RequirementsPipelineError("mechanical", "Global scope has extraneous identity");
		return { kind: "global" };
	}
	if (
		candidate.kind === "project" &&
		candidate.projectId === input.projectId &&
		candidate.sessionId === undefined &&
		candidate.epoch === undefined
	)
		return { kind: "project", projectId: input.projectId };
	if (
		(candidate.kind === "task" || candidate.kind === "session") &&
		candidate.sessionId === input.authority.ownerSessionId &&
		candidate.epoch === input.authority.epoch &&
		candidate.projectId === undefined
	) {
		return { kind: candidate.kind, sessionId: candidate.sessionId, epoch: candidate.epoch };
	}
	throw new RequirementsPipelineError("mechanical", "Scope identity does not match the initiating owner");
}
export function admitRequirementsCandidates(raw: unknown, input: RequirementsEvidencePackage): RequirementsCandidate[] {
	const envelope = object(raw, "extraction");
	keys(
		envelope,
		["version", "sourceKey", "sourceIntegrity", "manifest", "disposition", "unresolved", "operations"],
		"extraction",
	);
	const source = input.source.source;
	if (source.referenceOnly)
		throw new RequirementsPipelineError("mechanical", "Reference-only material lacks operator source authority");
	if (
		envelope.version !== REQUIREMENTS_FORMAT ||
		envelope.sourceKey !== source.key ||
		envelope.sourceIntegrity !== source.integrity
	)
		throw new RequirementsPipelineError("mechanical", "Extraction source/version mismatch");
	if (envelope.disposition !== "complete" || array(envelope.unresolved, "unresolved").length !== 0)
		throw new RequirementsPipelineError(
			"mechanical",
			`Incomplete extraction: ${JSON.stringify(envelope.unresolved)}`,
		);
	const manifest = array(envelope.manifest, "manifest");
	const seenUnits = new Set<string>();
	for (const entry of manifest) {
		const unit = object(entry, "manifest unit");
		keys(unit, ["id", "sha256", "byteLength"], "manifest unit");
		const original = source.units.find(item => item.id === unit.id);
		if (
			!original ||
			original.sha256 !== unit.sha256 ||
			original.byteLength !== unit.byteLength ||
			seenUnits.has(original.id)
		)
			throw new RequirementsPipelineError("mechanical", "Invented, duplicate or partial source manifest");
		seenUnits.add(original.id);
	}
	if (seenUnits.size !== source.units.length)
		throw new RequirementsPipelineError("mechanical", "Not every original source unit was presented");
	const seen = new Set<string>();
	return array(envelope.operations, "operations").map(rawOperation => {
		const op = object(rawOperation, "operation");
		keys(
			op,
			["id", "kind", "requirementId", "statement", "scope", "evidence", "referents", "predecessorRevisionIds"],
			"operation",
		);
		const id = text(op.id, "candidate id");
		if (seen.has(id)) throw new RequirementsPipelineError("mechanical", "Duplicate candidate id");
		seen.add(id);
		if (op.kind !== "add" && op.kind !== "change" && op.kind !== "withdraw")
			throw new RequirementsPipelineError("mechanical", "Unknown operation");
		const evidence = array(op.evidence, "evidence").map(span => verifyRequirementsEvidence(span, input));
		if (
			evidence.length &&
			evidence.every(span => span.sourceKey === source.key && source.adoptedUnitIds?.includes(span.unitId))
		)
			throw new RequirementsPipelineError(
				"mechanical",
				"Candidate duplicates an already operator-adopted complete unit",
			);
		if (!evidence.some(span => span.sourceKey === source.key) || source.origin.kind !== "human")
			throw new RequirementsPipelineError("mechanical", "Operation lacks host-attested operator source authority");
		const referents = array(op.referents ?? [], "referents").map(span => verifyRequirementsEvidence(span, input));
		const predecessors = array(op.predecessorRevisionIds, "predecessorRevisionIds").map(value =>
			text(value, "predecessor"),
		);
		if (new Set(predecessors).size !== predecessors.length)
			throw new RequirementsPipelineError("mechanical", "Duplicate predecessors");
		const requirementId = op.requirementId === undefined ? undefined : text(op.requirementId, "requirementId");
		if (op.kind === "add") {
			if (requirementId !== undefined || predecessors.length)
				throw new RequirementsPipelineError("mechanical", "Add cannot choose durable IDs or predecessors");
		} else if (
			!requirementId ||
			!predecessors.length ||
			predecessors.some(
				id => !input.active.some(revision => revision.id === id && revision.requirementId === requirementId),
			)
		) {
			throw new RequirementsPipelineError("mechanical", "Unknown correction/withdrawal target or predecessor");
		}
		const relations = predecessors
			.filter(id => input.source.operatorTargetRevisionIds?.includes(id))
			.map(id => {
				const predecessor = input.active.find(revision => revision.id === id)!;
				return {
					predecessorSourceKey: predecessor.sourceKey,
					successorSourceKey: source.key,
					evidence: [
						...evidence.filter(span => span.sourceKey === source.key),
						...predecessor.evidence.map(span => verifyRequirementsEvidence(span, input)),
					],
				};
			});
		return {
			id,
			kind: op.kind,
			...(requirementId ? { requirementId } : {}),
			statement: text(op.statement, "statement"),
			scope: scope(op.scope, input),
			evidence,
			referents,
			predecessorRevisionIds: predecessors,
			...(relations.length ? { relations } : {}),
		};
	});
}

/** Deliberately allowlisted: no source pointers, prior text, quote, target ID or producer metadata. */
export function projectRequirementsSanity(
	candidates: readonly RequirementsCandidate[],
): { id: string; operation: RequirementsOperation["kind"]; statement: string; scope: RequirementsScope }[] {
	return candidates.map(candidate => ({
		id: candidate.id,
		operation: candidate.kind,
		statement: candidate.statement,
		scope: {
			kind: candidate.scope.kind,
			...(candidate.scope.projectId !== undefined ? { projectId: candidate.scope.projectId } : {}),
			...(candidate.scope.sessionId !== undefined ? { sessionId: candidate.scope.sessionId } : {}),
			...(candidate.scope.epoch !== undefined ? { epoch: candidate.scope.epoch } : {}),
		},
	}));
}
function reviewResult(
	raw: unknown,
	candidates: readonly RequirementsCandidate[],
	input?: RequirementsEvidencePackage,
): Omit<RequirementsReviewOutcome, "model" | "format"> {
	const response = object(raw, "review");
	keys(response, input ? ["coverage", "reason", "obligations", "candidates"] : ["candidates"], "review");
	const decisions = array(response.candidates, "review candidates");
	if (decisions.length !== candidates.length)
		throw new RequirementsPipelineError("mechanical", "Reviewer omitted candidate decisions");
	const candidateIds = new Set(candidates.map(candidate => candidate.id));
	const seen = new Set<string>();
	let outcome: "accepted" | "rejected" | "failed" = "accepted";
	const reasons: string[] = [];
	const recorded: NonNullable<RequirementsReviewOutcome["obligations"]> = [];
	const candidateReviews: NonNullable<RequirementsReviewOutcome["candidates"]> = [];
	const consider = (decision: unknown, reason: unknown) => {
		if (!["pass", "reject", "uncertain"].includes(String(decision)))
			throw new RequirementsPipelineError("mechanical", "Invalid reviewer decision");
		const explanation = text(reason, "review reason");
		if (decision !== "pass") {
			if (decision === "reject" || outcome !== "rejected") outcome = decision === "reject" ? "rejected" : "failed";
			reasons.push(explanation);
		}
	};
	if (input) {
		consider(response.coverage, response.reason);
		const units = new Set(input.source.source.units.map(unit => unit.id));
		const known = new Map(input.active.map(revision => [revision.id, revision]));
		const applicable = new Set(input.applicableRevisionIds);
		const existing = new Set(input.active.filter(revision => applicable.has(revision.id) && revision.lifecycle === "accepted" && (!revision.availability || revision.availability === "available")).map(revision => revision.id));
		const operationsById = new Map(candidates.map(candidate => [candidate.id, candidate]));
		const knownIds = new Set(known.keys());
		const adopted = new Set(input.source.source.adoptedUnitIds ?? []);
		const obligations = new Set<string>();
		const identifiers = (value: unknown, allowed: ReadonlySet<string>, label: string): string[] => {
			const ids = array(value, label).map(value => text(value, label));
			if (new Set(ids).size !== ids.length || ids.some(id => !allowed.has(id)))
				throw new RequirementsPipelineError("mechanical", `Unknown or duplicate ${label}`);
			return ids;
		};
		for (const value of array(response.obligations, "source obligations")) {
			const obligation = object(value, "source obligation");
			keys(obligation, ["id", "kind", "requirementId", "predecessorRevisionIds", "statement", "sourceUnitIds", "operationIds", "applicableRevisionIds", "adoptedUnitIds", "decision", "reason"], "source obligation");
			const id = text(obligation.id, "obligation id");
			const statement = text(obligation.statement, "obligation statement");
			if (obligations.has(id)) throw new RequirementsPipelineError("mechanical", "Duplicate source obligation id");
			obligations.add(id);
			const kind = text(obligation.kind, "obligation kind");
			if (kind !== "add" && kind !== "change" && kind !== "withdraw")
				throw new RequirementsPipelineError("mechanical", "Unknown obligation operation kind");
			const target = obligation.requirementId === undefined ? undefined : text(obligation.requirementId, "obligation requirement id");
			const predecessors = identifiers(obligation.predecessorRevisionIds, knownIds, "obligation predecessor ids");
			if (kind === "add" ? target !== undefined || predecessors.length > 0 : !target || !predecessors.length || predecessors.some(id => known.get(id)!.requirementId !== target))
				throw new RequirementsPipelineError("mechanical", "Obligation target does not match its requested operation");
			const sourceUnitIds = identifiers(obligation.sourceUnitIds, units, "obligation source units");
			if (!sourceUnitIds.length) throw new RequirementsPipelineError("mechanical", "Source obligation lacks original units");
			const operations = identifiers(obligation.operationIds, candidateIds, "coverage operation ids");
			const revisions = identifiers(obligation.applicableRevisionIds, existing, "coverage applicable revision ids");
			const adoptedUnits = identifiers(obligation.adoptedUnitIds, adopted, "coverage adopted unit ids");
			if (adoptedUnits.some(unit => !sourceUnitIds.includes(unit)))
				throw new RequirementsPipelineError("mechanical", "Adopted coverage does not identify an obligation source unit");
			if (!operations.length && !revisions.length)
				consider("reject", `Uncovered source obligation ${id}: ${statement}`);
			consider(obligation.decision, obligation.reason);
			recorded.push({ id, kind, ...(target ? { requirementId: target } : {}), predecessorRevisionIds: predecessors, statement, sourceUnitIds, operationIds: operations, applicableRevisionIds: revisions, adoptedUnitIds: adoptedUnits, decision: obligation.decision as "pass" | "reject" | "uncertain", reason: String(obligation.reason) });
			for (const operationId of operations) {
				const operation = operationsById.get(operationId)!;
				if (!operation.evidence.some(unit => unit.sourceKey === input.source.source.key && sourceUnitIds.includes(unit.unitId)))
					consider("reject", `Coverage operation ${operationId} does not cite obligation ${id}`);
				if (operation.kind !== kind || operation.requirementId !== target || predecessors.some(id => !operation.predecessorRevisionIds.includes(id)))
					consider("reject", `Coverage operation ${operationId} does not perform obligation ${id}`);
			}
			for (const revisionId of revisions) {
				const revision = known.get(revisionId)!;
				if ((kind === "withdraw") !== (revision.kind === "withdraw") || (kind !== "add" && (revision.requirementId !== target || predecessors.includes(revision.id))))
					consider("reject", `Unchanged or unrelated revision ${revisionId} does not fulfill obligation ${id}`);
			}
		}
	}
	for (const entry of decisions) {
		const decision = object(entry, "candidate decision");
		keys(decision, ["id", "decision", "reason"], "candidate decision");
		const id = text(decision.id, "review id");
		if (seen.has(id) || !candidateIds.has(id))
			throw new RequirementsPipelineError("mechanical", "Invented or duplicate reviewer id");
		seen.add(id);
		consider(decision.decision, decision.reason);
		candidateReviews.push({ id, decision: decision.decision as "pass" | "reject" | "uncertain", reason: String(decision.reason) });
	}
	return { outcome, candidates: candidateReviews, ...(input ? { obligations: recorded } : {}), ...(reasons.length ? { reason: reasons.join("; ") } : {}) };
}
type CitedRequirementsEvidence = RequirementsEvidence & { text?: string; image?: ImageContent };
function citedEvidenceKey(span: RequirementsEvidence): string {
	return JSON.stringify([span.sourceKey, span.integrity, span.unitId]);
}
function citedEvidencePayload(
	input: RequirementsEvidencePackage,
	candidates: readonly RequirementsCandidate[],
): Map<string, CitedRequirementsEvidence> {
	const sources = new Map<string, ResolvedRequirementsSource>();
	for (const source of allSources(input)) sources.set(source.source.key, source);
	const cited = new Map<string, CitedRequirementsEvidence>();
	const append = (span: RequirementsEvidence) => {
		const identity = citedEvidenceKey(span);
		if (cited.has(identity)) return;
		verifyRequirementsEvidence(span, input);
		const unit = sources.get(span.sourceKey)!.units.find(unit => unit.id === span.unitId)!;
		if (unit.text !== undefined) {
			cited.set(identity, { ...span, text: unit.text });
		} else if (unit.image) cited.set(identity, { ...span, image: unit.image });
		else throw new RequirementsPipelineError("mechanical", "Original cited evidence is unavailable");
	};
	for (const candidate of candidates) {
		for (const span of candidate.evidence) append(span);
		for (const span of candidate.referents ?? []) append(span);
		for (const relation of candidate.relations ?? [])
			for (const span of relation.evidence) append(span);
	}
	return cited;
}
function contextualPayload(input: RequirementsEvidencePackage) {
	const applicable = new Set(input.applicableRevisionIds);
	const applicableRequirements: RequirementsRevision[] = [];
	const knownRequirements: RequirementsRevision[] = [];
	for (const revision of input.active)
		(applicable.has(revision.id) ? applicableRequirements : knownRequirements).push(revision);
	return {
		sourceKey: input.source.source.key,
		sourceIntegrity: input.source.source.integrity,
		projectId: input.projectId,
		owner: input.authority,
		sources: allSources(input).map(source =>
			source.context === input.source.context && source.contextIndex !== undefined
				? { descriptor: source.source, contextIndex: source.contextIndex }
				: { descriptor: source.source, units: source.units },
		),
		originalContext: input.source.context,
		unavailableContext: input.source.unavailableContext,
		applicableRequirements,
		knownRequirements,
		operatorTargetRevisionIds: input.source.operatorTargetRevisionIds,
	};
}
async function call(
	host: RequirementsPipelineHost,
	stage: keyof typeof roles,
	payload: unknown,
	input: RequirementsEvidencePackage,
	signal: AbortSignal,
	contextual: boolean,
): Promise<{ raw: unknown; model: string }> {
	const resolution = resolveRoleModelFull(host.settings, roles[stage], host.modelRegistry.getAll(), host.getModel());
	const model = resolution.model;
	if (!model) throw new RequirementsPipelineError(stage, resolution.warning ?? `Configure ${roles[stage]} model role`);
	const images: ImageContent[] = [];
	if (contextual) {
		for (const source of allSources(input))
			for (const descriptor of source.source.units) {
				if (descriptor.unsupportedReason)
					throw new RequirementsPipelineError(stage, descriptor.unsupportedReason, true);
				const unit = source.units.find(unit => unit.id === descriptor.id);
				if (!unit || (unit.text === undefined && !unit.image))
					throw new RequirementsPipelineError(stage, `Original unit ${descriptor.id} unavailable`, true);
				verifyRequirementsEvidence({ sourceKey: source.source.key, integrity: source.source.integrity, unitId: descriptor.id }, input);
			}
	}
	const imageIndexes = new Map<string, number>();
	const serialized = JSON.stringify(payload, (_key, value: unknown) => {
		if (
			!contextual ||
			!value ||
			typeof value !== "object" ||
			!("type" in value) ||
			value.type !== "image" ||
			!("data" in value) ||
			typeof value.data !== "string" ||
			!("mimeType" in value) ||
			typeof value.mimeType !== "string"
		)
			return value;
		let index = imageIndexes.get(value.data);
		if (index === undefined) {
			index = images.length;
			images.push({ type: "image", data: value.data, mimeType: value.mimeType });
			imageIndexes.set(value.data, index);
		}
		return { type: "image", mimeType: value.mimeType, suppliedImageIndex: index };
	});
	if (images.length && !model.input.includes("image"))
		throw new RequirementsPipelineError(
			stage,
			`${model.provider}/${model.id} lacks required vision capability`,
			true,
		);
	const owner = { sessionId: host.sessionManager.getSessionId(), parentId: host.sessionManager.getLeafId() };
	const prompt =
		stage === "extractor"
			? extractionPrompt
			: stage === "evidence"
				? evidencePrompt
				: sanityPrompt;
	const response = await retryTransientCompletion(async () => {
		signal.throwIfAborted();
		// Fresh identity on every attempt. Credential ownership is NOT conversation ownership.
		const identity = `requirements-${stage}-${randomUUID()}`;
		const result = await completeSimple(
			model,
			{
				systemPrompt: [prompt],
				messages: [
					{ role: "user", content: [{ type: "text", text: serialized }, ...images], timestamp: Date.now() },
				],
			},
			{
				apiKey: host.modelRegistry.resolver(model, owner.sessionId),
				signal,
				sessionId: identity,
				promptCacheKey: identity,
				cacheRetention: "none",
				...(resolution.thinkingLevel === "off"
					? { disableReasoning: true }
					: resolution.thinkingLevel &&
						  resolution.thinkingLevel !== "auto" &&
						  resolution.thinkingLevel !== "inherit"
						? { reasoning: clampThinkingLevelForModel(model, resolution.thinkingLevel) }
						: {}),
			},
		);
		host.sessionManager.appendModelUsage(
			{
				purpose: `requirements-${stage}`,
				role: roles[stage],
				api: result.api,
				provider: result.provider,
				model: result.model,
				usage: result.usage,
				stopReason: result.stopReason,
				errorMessage: result.errorMessage,
			},
			owner,
		);
		return result;
	});
	signal.throwIfAborted();
	if (response.stopReason !== "stop" || response.content.some(block => block.type === "toolCall"))
		throw new RequirementsPipelineError(
			stage,
			response.errorMessage ?? `Incomplete response (${response.stopReason})`,
		);
	const output = response.content
		.filter(block => block.type === "text")
		.map(block => block.text)
		.join("");
	try {
		return { raw: JSON.parse(output), model: `${model.provider}/${model.id}` };
	} catch {
		throw new RequirementsPipelineError(stage, "Malformed or truncated JSON; no prefix accepted");
	}
}
export async function reviewRequirementsCandidates(
	host: RequirementsPipelineHost,
	input: RequirementsEvidencePackage,
	candidates: RequirementsCandidate[],
	signal: AbortSignal,
	literalAcceptance?: RequirementsReview["literalAcceptance"],
): Promise<RequirementsReview> {
	const review: RequirementsReview = {};
	if (literalAcceptance) {
		const source = input.source;
		if (
			source.source.referenceOnly ||
			source.source.origin.kind !== "human" ||
			!literalAcceptance.actor ||
			literalAcceptance.sourceKey !== source.source.key ||
			literalAcceptance.integrity !== source.source.integrity ||
			literalAcceptance.unitIds.length !== 1 ||
			candidates.length !== 1
		)
			throw new RequirementsPipelineError("mechanical", "Invalid operator literal adoption");
		const unit = source.units.find(unit => unit.id === literalAcceptance.unitIds[0]);
		if (
			unit?.text === undefined ||
			candidates[0].statement !== unit.text ||
			candidates[0].evidence.length !== 1 ||
			candidates[0].evidence[0].unitId !== unit.id ||
			candidates[0].evidence[0].sourceKey !== source.source.key
		)
			throw new RequirementsPipelineError(
				"mechanical",
				"Literal exception requires unchanged complete source-unit text",
			);
		verifyRequirementsEvidence(candidates[0].evidence[0], input);
		review.literalAcceptance = literalAcceptance;
	} else {
		try {
			const cited = citedEvidencePayload(input, candidates);
			const result = await call(
				host,
				"evidence",
				{
					...contextualPayload(input),
					candidates,
					citedUnits: [...cited.values()],
					disposition: "complete",
				},
				input,
				signal,
				true,
			);
			review.evidence = {
				model: result.model,
				format: REQUIREMENTS_FORMAT,
				...reviewResult(result.raw, candidates, input),
			};
		} catch (error) {
			if (signal.aborted) throw error;
			review.evidence = {
				model: requirementsModelStatus(host).evidence.resolved ?? "unresolved:requirementsEvidence",
				format: REQUIREMENTS_FORMAT,
				outcome: "failed",
				reason: String(error),
			};
		}
		if (review.evidence.outcome !== "accepted") return review;
	}
	try {
		const result = await call(
			host,
			"sanity",
			{ candidates: projectRequirementsSanity(candidates) },
			input,
			signal,
			false,
		);
		review.sanity = {
			model: result.model,
			format: REQUIREMENTS_FORMAT,
			...reviewResult(result.raw, candidates),
		};
	} catch (error) {
		if (signal.aborted) throw error;
		review.sanity = {
			model: requirementsModelStatus(host).sanity.resolved ?? "unresolved:requirementsSanity",
			format: REQUIREMENTS_FORMAT,
			outcome: "failed",
			reason: String(error),
		};
	}
	return review;
}
export function requirementsReviewAccepted(review: RequirementsReview): boolean {
	return (
		!!(review.literalAcceptance || review.evidence?.outcome === "accepted") && review.sanity?.outcome === "accepted"
	);
}
/** Freeze selected evidence identities; bodies remain in original source storage. */
export function createRequirementsBatch(
	input: RequirementsEvidencePackage,
	operations: readonly RequirementsOperation[] = [],
	operationIds: readonly string[] = [],
): RequirementsBatch {
	if (input.source.source.referenceOnly)
		throw new RequirementsPipelineError("mechanical", "Reference-only material lacks operator source authority");
	if (operationIds.length !== operations.length || new Set(operationIds).size !== operationIds.length || operationIds.some(id => !id.trim()))
		throw new RequirementsPipelineError("mechanical", "Batch requires one unique ID per operation");
	const selected = new Set<string>();
	for (const operation of operations) {
		for (const span of operation.evidence) selected.add(span.sourceKey);
		for (const span of operation.referents ?? []) selected.add(span.sourceKey);
		for (const relation of operation.relations ?? []) {
			selected.add(relation.predecessorSourceKey);
			selected.add(relation.successorSourceKey);
			for (const span of relation.evidence) selected.add(span.sourceKey);
		}
	}
	return {
		id: randomUUID(),
		sourceKey: input.source.source.key,
		sourceIntegrity: input.source.source.integrity,
		extractionVersion: REQUIREMENTS_FORMAT,
		reviewRevision: `${REQUIREMENTS_FORMAT}:${input.publicationRevision}`,
		manifest: input.source.source.units,
		readHeads: input.readHeads,
		readSourceIntegrities: Object.fromEntries(
			allSources(input)
				.filter(source => !source.source.referenceOnly || selected.has(source.source.key))
				.map(source => [source.source.key, source.source.integrity]),
		),
		authority: input.authority,
		operations: [...operations],
		operationIds: [...operationIds],
		review: {},
		status: "pending",
	};
}
export async function extractRequirementsBatch(
	host: RequirementsPipelineHost,
	input: RequirementsEvidencePackage,
	signal: AbortSignal,
): Promise<RequirementsBatch> {
	if (input.source.source.referenceOnly)
		throw new RequirementsPipelineError("mechanical", "Reference-only material lacks operator source authority");
	const extraction = await call(host, "extractor", contextualPayload(input), input, signal, true);
	const candidates = admitRequirementsCandidates(extraction.raw, input);
	const operations = candidates.map(({ id: _id, ...operation }) => operation);
	const batch = createRequirementsBatch(input, operations, candidates.map(candidate => candidate.id));
	const review = await reviewRequirementsCandidates(host, input, candidates, signal);
	return {
		...batch,
		operations,
		review,
		status: requirementsReviewAccepted(review) ? "reviewed" : "rejected",
		reason: review.evidence?.reason ?? review.sanity?.reason,
	};
}
