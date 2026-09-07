/** Local provenance only. These descriptors never become provider wire fields. */
export interface SourceRange {
	/** Half-open UTF-16 offsets in the explicitly named source or output string. */
	start: number;
	end: number;
}

export interface SourceBlockRange extends SourceRange {
	blockIndex: number;
}

export interface SourceAtomicGroup {
	id: string;
	entryIds: string[];
}

export interface SourceMessage<TMessage> {
	entryId: string;
	/** Position in the frozen active, post-reset ancestry, not a timestamp. */
	order: number;
	message: TMessage;
	atomicGroup?: SourceAtomicGroup;
}

export interface SourceCoverageRun {
	entryId: string;
	order: number;
	atomicGroup?: SourceAtomicGroup;
	/** Coordinates when the artifact was produced; never rewritten. */
	snapshot: SourceBlockRange;
	/** Surviving correspondence to current journal bytes; absent when unknown. */
	current?: SourceBlockRange;
	/** Coordinates into the owning archive's retained normalized text. */
	normalized?: SourceRange;
	status: "exact-current" | "historical-not-current" | "unknown";
}

export type SourceLayoutPart =
	| { kind: "text"; range: SourceRange }
	| { kind: "frame"; frameIndex: number; range: SourceRange }
	| {
			kind: "original-image";
			entryId: string;
			order: number;
			/** Immutable source block at capture. */
			blockIndex: number;
			/** Current source block; absence denotes an unresolved historical reference. */
			currentBlockIndex?: number;
		}
	| {
			kind: "source";
			entryId: string;
			order: number;
			/** Current source blocks/ranges; absence means the complete original atom member. */
			spans?: SourceBlockRange[];
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
	/** Omission means no usable positional map: current coverage becomes unknown. */
	blocks?: SourceBlockRewrite[];
}
