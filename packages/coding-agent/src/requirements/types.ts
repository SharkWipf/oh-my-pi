export interface RequirementsOriginalSource {
	journalId: string;
	entryId: string;
}
export interface RequirementsLocator {
	/** Normal BlobStore envelope retained by an explicit dependent-evidence deletion choice. */
	retainedBlobHash?: string;
	sessionId: string;
	journalPath?: string;
	entryId: string;
}
export interface RequirementsUnit {
	id: string;
	kind: "text" | "attachment" | "image" | "audio" | "opaque";
	byteLength: number;
	sha256: string;
	locator?: string;
	unsupportedReason?: string;
}
export interface RequirementsOrigin {
	kind: "human" | "extension" | "tool" | "assistant" | "advisor" | "subagent" | "unknown";
	producerId?: string;
}
export type RequirementsSourceState = "pending" | "complete" | "unsupported" | "failed" | "gap" | "orphaned";
export interface RequirementsSource {
	/** Generation of actual accepted live delivery; historical backfill never advances it. */
	authorityGeneration?: number;
	/** Journal-backed nonhuman evidence; never source intake or publication authority. */
	referenceOnly?: true;
	key: string;
	original: RequirementsOriginalSource;
	locators: RequirementsLocator[];
	integrity: string;
	parentKey: string | null;
	/** False when a recorded coverage gap refers to currently unavailable original bytes. */
	integrityAvailable?: boolean;
	ownerSessionId: string;
	branchId: string;
	epoch: number;
	origin: RequirementsOrigin;
	units: RequirementsUnit[];
	durable: boolean;
	state: RequirementsSourceState;
	reason?: string;
	gap?: { actor: string; integrity: string; reason: string };
	/** Complete units explicitly adopted under this integrity; other units remain uncovered. */
	adoptedUnitIds?: string[];
}
/** Host-issued complete immutable source unit; excerpts are not evidence identities. */
export interface RequirementsEvidence {
	sourceKey: string;
	integrity: string;
	unitId: string;
}
export interface RequirementsScope {
	kind: "global" | "project" | "session" | "task";
	projectId?: string;
	sessionId?: string;
	epoch?: number;
}
/** Only the host evidence resolver constructs these, never the extractor. */
export interface RequirementsRelation {
	predecessorSourceKey: string;
	successorSourceKey: string;
	evidence: RequirementsEvidence[];
}
export interface RequirementsOperation {
	kind: "add" | "change" | "withdraw";
	requirementId?: string;
	statement: string;
	scope: RequirementsScope;
	evidence: RequirementsEvidence[];
	referents?: RequirementsEvidence[];
	predecessorRevisionIds: string[];
	relations?: RequirementsRelation[];
}
export interface RequirementsAuthority {
	ownerSessionId: string;
	branchId: string;
	epoch: number;
	generation: number;
}
export interface RequirementsReviewOutcome {
	candidates?: { id: string; decision: "pass" | "reject" | "uncertain"; reason: string }[];
	model: string;
	format: string;
	outcome: "accepted" | "rejected" | "failed";
	reason?: string;
	obligations?: {
		id: string;
		kind: RequirementsOperation["kind"];
		requirementId?: string;
		predecessorRevisionIds: string[];
		statement: string;
		sourceUnitIds: string[];
		operationIds: string[];
		applicableRevisionIds: string[];
		adoptedUnitIds: string[];
		decision: "pass" | "reject" | "uncertain";
		reason: string;
	}[];
}
export interface RequirementsReview {
	evidence?: RequirementsReviewOutcome;
	sanity?: RequirementsReviewOutcome;
	literalAcceptance?: { actor: string; sourceKey: string; integrity: string; unitIds: string[] };
}
export interface RequirementsBatch {
	operationIds: string[];
	id: string;
	sourceKey: string;
	sourceIntegrity: string;
	extractionVersion: string;
	reviewRevision: string;
	manifest: RequirementsUnit[];
	readHeads: Record<string, string[]>;
	readSourceIntegrities: Record<string, string>;
	authority: RequirementsAuthority;
	operations: RequirementsOperation[];
	review: RequirementsReview;
	status: "pending" | "reviewed" | "accepted" | "stale" | "rejected";
	reason?: string;
}
export interface RequirementsRevision extends RequirementsOperation {
	/** Exact operator withdrawal; never cleared by quarantine restore or evidence reappearance. */
	withdrawal?: { actor: string; reason: string; generation: number };
	id: string;
	requirementId: string;
	batchId: string;
	sourceKey: string;
	sourceIntegrity: string;
	publicationRevision: number;
	lifecycle: "accepted" | "quarantined" | "historical";
	availability?: "available" | "unavailable" | "changed";
	quarantine?: { actor: string; reason: string; generation: number };
}
export interface RequirementsRestoreReceipt {
	operationIds: string[];
	readHeads: Record<string, string[]>;
	id: string;
	actor: string;
	revisionIds: string[];
	generation: number;
	publicationRevision: number;
	sourceIntegrities: Record<string, string>;
	review: RequirementsReview;
}
export interface RequirementsState {
	tombstones?: { scope: RequirementsScope; actor: string; reason: string; generation: number }[];
	publicationRevision: number;
	generation: number;
	owners: Record<string, RequirementsAuthority>;
	restoreReceipts: RequirementsRestoreReceipt[];
	/** Latest unsuccessful restore per revision; acceptance does not erase prior rejection. */
	restoreReviews: Record<string, { receipt: RequirementsRestoreReceipt; reason: string }>;
}
export interface RequirementsSnapshot {
	state: RequirementsState;
	sources: RequirementsSource[];
	batches: RequirementsBatch[];
	revisions: RequirementsRevision[];
}
/** Intake metadata is a ledger comparison, never proof of current original bytes. */
export type RequirementsSourceMetadata = Pick<
	RequirementsSource,
	"key" | "integrity" | "integrityAvailable" | "state" | "locators"
>;
export interface RequirementsCoverageSummary {
	total: number;
	referenceOnly: number;
	/** Source dispositions only; referents do not represent source coverage. */
	byState: Record<RequirementsSourceState, number>;
}
export interface RequirementsConsumptionContext {
	projectId?: string;
	sessionId: string;
	epoch: number;
	sourceKeys?: ReadonlySet<string>;
	branchId: string;
}
/** One detached ledger read, not authorization to retry/backfill the full catalog. */
export interface RequirementsConsumptionSnapshot extends RequirementsSnapshot {
	readHeads: Record<string, string[]>;
	/** All catalog rows remain inspectable through the full snapshot/exact source API. */
	coverage: RequirementsCoverageSummary;
	/** Unresolved genuine input; callers apply active ancestry/epoch before protection. */
	pendingSources: RequirementsSource[];
	/** Scoped heads computed against the full revision metadata, not the narrowed evidence list. */
	applicable: RequirementsApplicable;
	/** Settled accepted heads, including fulfilled withdrawals; never conflicts or suspended predecessors. */
	applicableRevisionIds: string[];
}
export interface RequirementsObservation {
	key: string;
	integrity: string | null;
	locators?: RequirementsLocator[];
	units?: RequirementsUnit[];
}
export interface RequirementsPublicationResult {
	status: "accepted" | "already-accepted" | "waiting" | "stale" | "rejected";
	reason?: string;
	revisionIds: string[];
	publicationRevision: number;
}
export interface RequirementsApplicable {
	active: RequirementsRevision[];
	conflicts: RequirementsRevision[][];
	coverageGaps: RequirementsSource[];
}
