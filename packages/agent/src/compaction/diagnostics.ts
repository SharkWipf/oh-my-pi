/** A quantity describes its measurement domain; it is never a provider invoice by implication. */
export interface DiagnosticTokenQuantity {
	tokens: number | null;
	basis: "local-estimate" | "tokenizer" | "upper-bound" | "image-estimate" | "provider-reported" | "unknown";
	description: string;
}

export type ContextInventoryKind =
	| "system-prompt"
	| "tools"
	| "context"
	| "skills"
	| "summary"
	| "native"
	| "text"
	| "tool"
	| "frame"
	| "original-image"
	| "gap"
	| "retained"
	| "post-compaction";

/** One disjoint physical occurrence. Separated occurrences must not be coalesced. */
export interface ContextInventoryRow {
	location: string;
	kind: ContextInventoryKind;
	label: string;
	coverage: "source" | "aggregate" | "historical" | "unknown" | "fixed";
	sourceIds?: string[];
	/** Physical ownership intersecting this occurrence; a shared frame is still charged once. */
	contributions?: ("ordinary" | "selected-user" | "manual-nonuser")[];
	selectionReasons: string[];
	quantity: DiagnosticTokenQuantity;
	counts: { messages: number; blocks: number; frames: number; images: number; items?: number };
	/** UTF-8 bytes for text, base64 characters for encoded image payloads. */
	payloadSize?: { value: number; unit: "utf8-bytes" | "base64-characters" };
	controls: string[];
	note?: string;
}

/** Stored on the existing compaction event, together with the result it describes. */
export interface CompactionDiagnostics {
	snapshot: "recorded-at-compaction" | "current-reconstruction" | "OMP-prehook" | "observed-posthook";
	model: string;
	method: string;
	contextWindow: number | null;
	/** Effective settings captured for this operation, not a reference to live settings. */
	settings: Record<string, unknown>;
	target: {
		ordinaryTokens?: number;
		reserveTokens?: number;
		calibratedOrdinaryTokens?: number;
		manualNonUserTokens?: number;
		residualOrdinaryTokens?: number;
	};
	before: DiagnosticTokenQuantity;
	/** Sum of the disjoint rows' known local charges; unknown native costs remain unknown. */
	total: DiagnosticTokenQuantity;
	rows: ContextInventoryRow[];
	selection?: {
		sourceReasons: Record<string, string[]>;
		quota: Record<string, { count: number; tokens: number }>;
	};
	distribution: {
		ordinary: DiagnosticTokenQuantity;
		addedUser: DiagnosticTokenQuantity;
		shared: DiagnosticTokenQuantity;
		manualNonUserSources: number;
		/** Source quota membership overlaps and is not an additive physical bill. */
		selectedUserSources: number;
		addedUserSources: number;
	};
	providerAnchor?: DiagnosticTokenQuantity;
	warning?: string;
	notes: string[];
}
