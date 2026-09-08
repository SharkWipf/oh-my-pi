/** Local provenance only. These descriptors never become provider wire fields. */
export interface SourceRange {
	/** Half-open UTF-16 offsets in the explicitly named source or output string. */
	start: number;
	end: number;
}

/** Text uses UTF-16 offsets; an original image occupies one atomic [0, 1) unit. */
export interface SourceBlockRange extends SourceRange {
	blockIndex: number;
}

export interface SourceAtomicGroup {
	id: string;
	entryIds: string[];
}

export interface SourceMessage<TMessage> {
	entryId: string;
	/** Original submission differs from delivered message; absent means delivery coordinates. */
	projection?: "original";
	/** Position in the frozen active, post-reset ancestry, not a timestamp. */
	order: number;
	message: TMessage;
	atomicGroup?: SourceAtomicGroup;
	/** Actual ordinary coverage of a previously sparse source; message remains ORIGINAL. */
	spans?: SourceBlockRange[];
}

export interface SourceCoverageRun {
	entryId: string;
	projection?: "original";
	order: number;
	atomicGroup?: SourceAtomicGroup;
	/** Coordinates when the artifact was produced; never rewritten. */
	snapshot: SourceBlockRange;
	/** Surviving correspondence to current journal bytes; absent when unknown. */
	current?: SourceBlockRange;
	/** Coordinates into the owning archive's retained normalized text. */
	normalized?: SourceRange;
	/** Included output slice of one normalized raw unit; offsets relative to its complete normalized output. */
	normalizedUnit?: SourceRange & { length: number };
	status: "exact-current" | "historical-not-current" | "unknown";
	/** Frozen operation membership, independent of overlapping policy quota reasons. */
	contribution?: "ordinary" | "selected-user" | "manual-nonuser";
}

export type SourceLayoutPart =
	| { kind: "text"; range: SourceRange }
	| { kind: "frame"; frameIndex: number; range: SourceRange }
	| {
			kind: "original-image";
			entryId: string;
			projection?: "original";
			order: number;
			/** Immutable source block at capture. */
			blockIndex: number;
			/** Current source block; absence denotes an unresolved historical reference. */
			currentBlockIndex?: number;
		}
	| {
			kind: "source";
			entryId: string;
			projection?: "original";
			order: number;
			/** Current source blocks/ranges; absence means the complete original atom member. */
			spans?: SourceBlockRange[];
			/** Frozen ownership of this whole emitted part; mixed spans use per-run coverage instead. */
			contribution?: SourceCoverageRun["contribution"];
		}
	| {
			kind: "gap";
			beforeEntryId?: string;
			afterEntryId?: string;
			/** Positive wholly missing context-producing entries only; never partial truncation. */
			wholeMessages?: number;
			reason: "omitted-messages" | "partial-text" | "image-deleted" | "unknown-source";
		};

/** Stored once in existing compaction preserveData.sourceRepresentation. */
export interface SourceRepresentation {
	version: 1;
	/** When present, layout is the complete emitted history through this captured
	 * source entry; only later entries use the ordinary live suffix. */
	throughEntryId?: string;
	coverage: SourceCoverageRun[];
	/** Physical order; ranges name archive.text, frame indices name archive.frames. */
	layout: SourceLayoutPart[];
	/** Summary/opaque processing is not exact retained source coverage. */
	aggregate?: { entryIds: string[]; reason: "summary" | "native" | "legacy" };
}

/** One simultaneous replacement, expressed in the OLD block's UTF-16 coordinates. */
export interface SourceTextEdit extends SourceRange {
	replacementLength: number;
}

export interface SourceBlockRewrite {
	oldBlockIndex: number;
	/** null means explicit block deletion. */
	newBlockIndex: number | null;
	/** Sorted, non-overlapping edits in old coordinates. Omission means bytes unchanged. */
	textEdits?: SourceTextEdit[];
}

export interface SourceRewrite {
	entryId: string;
	/** Coordinate space changed by this rewrite; absent means delivered message. */
	projection?: "original";
	/** Omission means no usable positional map: current coverage becomes unknown. */
	blocks?: SourceBlockRewrite[];
}

/** Identity within one source projection; normal journal IDs are unchanged. */
export function compactionSourceKey(source: { entryId: string; projection?: "original" }): string {
	return source.projection === "original" ? `original:${source.entryId}` : source.entryId;
}
