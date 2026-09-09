import {
	compactionSourceKey,
	type SourceBlockRange,
	type SourceBlockRewrite,
	type SourceCoverageRun,
	type SourceLayoutPart,
	type SourceRange,
	type SourceRepresentation,
	type SourceRewrite,
} from "@oh-my-pi/pi-ai/compaction-source";
import type { AgentMessage } from "../types";
import { combineContentSourceOrigins, getSourceOrigin, type NativeSourcePart, setSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";

export type * from "@oh-my-pi/pi-ai/compaction-source";

export const SOURCE_REPRESENTATION_KEY = "sourceRepresentation";

export function getCompactionSourceRepresentation(
	preserveData: Record<string, unknown> | undefined,
): SourceRepresentation | undefined {
	const value = preserveData?.[SOURCE_REPRESENTATION_KEY] as SourceRepresentation | undefined;
	return value?.version === 1 && Array.isArray(value.coverage) && Array.isArray(value.layout) ? value : undefined;
}

/** Split current coordinates at known edits; artifact coordinates never move. */
function remapCoverage(run: SourceCoverageRun, rewrite: SourceRewrite): SourceCoverageRun[] {
	if (run.status !== "exact-current" || !run.current) return [run];
	if (!rewrite.blocks) return [{ ...run, current: undefined, status: "unknown" }];
	const block = rewrite.blocks.find(candidate => candidate.oldBlockIndex === run.current!.blockIndex);
	if (!block) return [run];
	if (block.newBlockIndex === null) {
		return [{ ...run, current: undefined, status: "historical-not-current" }];
	}
	const current = run.current;
	const edits = block.textEdits ?? [];
	const output: SourceCoverageRun[] = [];
	let cursor = current.start;
	let delta = 0;
	const append = (start: number, end: number, shift: number, historical: boolean) => {
		if (end <= start) return;
		const offset = start - current.start;
		const length = end - start;
		const snapshot = { ...run.snapshot, start: run.snapshot.start + offset, end: run.snapshot.start + offset + length };
		const normalized = run.normalized && run.normalized.end - run.normalized.start === current.end - current.start
			? { start: run.normalized.start + offset, end: run.normalized.start + offset + length }
			: start === current.start && end === current.end ? run.normalized : undefined;
		output.push({
			...run,
			snapshot,
			normalized,
			current: historical ? undefined : { blockIndex: block.newBlockIndex!, start: start + shift, end: end + shift },
			status: historical ? "historical-not-current" : "exact-current",
		});
	};
	for (const edit of edits) {
		if (edit.end <= current.start) {
			delta += edit.replacementLength - (edit.end - edit.start);
			continue;
		}
		if (edit.start >= current.end) break;
		append(cursor, Math.min(edit.start, current.end), delta, false);
		append(Math.max(cursor, edit.start), Math.min(edit.end, current.end), delta, true);
		cursor = Math.max(cursor, Math.min(edit.end, current.end));
		delta += edit.replacementLength - (edit.end - edit.start);
	}
	append(cursor, current.end, delta, false);
	// Empty ranges describe whole non-text blocks, not absent coverage.
	if (current.start === current.end) {
		return [{ ...run, current: { ...current, blockIndex: block.newBlockIndex } }];
	}
	return output;
}

function remapSpan(span: SourceBlockRange, block: SourceBlockRewrite): SourceBlockRange[] {
	const run: SourceCoverageRun = {
		entryId: "", order: 0, snapshot: span, current: span, status: "exact-current",
	};
	return remapCoverage(run, { entryId: "", blocks: [block] })
		.flatMap(part => part.current ? [part.current] : []);
}

function remapLayout(part: SourceLayoutPart, rewrite: SourceRewrite): SourceLayoutPart[] {
	if (part.kind !== "original-image" && part.kind !== "source") return [part];
	if (part.entryId !== rewrite.entryId || part.projection !== rewrite.projection) return [part];
	if (part.kind === "original-image") {
		if (!rewrite.blocks) return [{ ...part, currentBlockIndex: undefined }];
		const index = part.currentBlockIndex;
		if (index === undefined) return [part];
		const block = rewrite.blocks.find(candidate => candidate.oldBlockIndex === index);
		if (!block) return [part];
		return block.newBlockIndex === null ? [] : [{ ...part, currentBlockIndex: block.newBlockIndex }];
	}
	if (!part.spans || !rewrite.blocks) return [part];
	const spans = part.spans.flatMap(span => {
		const block = rewrite.blocks!.find(candidate => candidate.oldBlockIndex === span.blockIndex);
		return block ? remapSpan(span, block) : [span];
	});
	return spans.length ? [{ ...part, spans }] : [];
}

/**
 * Pure journal rewrite companion. Call for EVERY referencing compaction leaf in
 * the same transaction that publishes source bytes. Does not rerender artifacts.
 */
export function remapCompactionSourceRepresentation(
	preserveData: Record<string, unknown> | undefined,
	rewrites: readonly SourceRewrite[],
): Record<string, unknown> | undefined {
	const representation = getCompactionSourceRepresentation(preserveData);
	if (!representation || rewrites.length === 0) return preserveData;
	const byId = new Map(rewrites.map(rewrite => [compactionSourceKey(rewrite), rewrite]));
	if (!representation.coverage.some(run => byId.has(compactionSourceKey(run))) &&
		!representation.layout.some(part => "entryId" in part && byId.has(compactionSourceKey(part)))) return preserveData;
	const coverage = representation.coverage.flatMap(run => {
		const rewrite = byId.get(compactionSourceKey(run));
		return rewrite ? remapCoverage(run, rewrite) : [run];
	});
	const layout = representation.layout.flatMap(part => {
		const rewrite = "entryId" in part ? byId.get(compactionSourceKey(part)) : undefined;
		return rewrite ? remapLayout(part, rewrite) : [part];
	});
	return { ...preserveData, [SOURCE_REPRESENTATION_KEY]: { ...representation, coverage, layout } };
}

/** Source-coordinate union; never deduplicate by rendered text or semantic content. */
export function unionSourceRanges(ranges: readonly SourceRange[]): SourceRange[] {
	if (ranges.length === 0) return [];
	const sorted = [...ranges].sort((a, b) => a.start - b.start || a.end - b.end);
	const result: SourceRange[] = [];
	for (const range of sorted) {
		if (range.end <= range.start) continue;
		const previous = result[result.length - 1];
		if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
		else result.push({ ...range });
	}
	return result;
}

/** Materialize current source spans identically for reconstruction and the next
 * real compaction. Synthetic truncation markers are regenerated from real gaps. */
export function materializeCompactionSourceMessage(
	message: AgentMessage,
	spans?: readonly SourceBlockRange[],
): AgentMessage | undefined {
	if (!spans) return message;
	if (!("content" in message)) return message;
	const selected = new Map<number, SourceRange[]>();
	for (const span of spans) {
		const ranges = selected.get(span.blockIndex);
		if (ranges) ranges.push(span);
		else selected.set(span.blockIndex, [span]);
	}
	const slice = (text: string, blockIndex: number, original: object) => {
		const ranges = unionSourceRanges(selected.get(blockIndex) ?? []);
		const origin = getSourceOrigin(original);
		const parts: NativeSourcePart[] = [];
		let cursor = 0;
		let result = "";
		for (const range of ranges) {
			if (range.start > cursor) result += "[truncated]";
			const transportStart = result.length;
			result += text.slice(range.start, range.end);
			if (origin?.kind === "source") {
				for (const part of origin.parts) {
					if (part.blockIndex !== blockIndex) continue;
					parts.push({ ...part, coverage: range.start === 0 && range.end === text.length ? "full" : "partial",
						sourceSpan: { start: range.start, end: range.end }, sourceLength: text.length,
						transportSpan: { start: transportStart, end: result.length } });
				}
			}
			cursor = range.end;
		}
		if (cursor < text.length) result += "[truncated]";
		return { text: result, origin: parts.length ? { kind: "source" as const, parts } : origin?.kind === "source"
			? { kind: "synthetic" as const, reason: "truncation-marker" } : origin };
	};
	if (typeof message.content === "string") {
		if (!selected.has(0)) return undefined;
		const sliced = slice(message.content, 0, message);
		const result = { ...message, content: sliced.text } as AgentMessage;
		return sliced.origin ? setSourceOrigin(result, sliced.origin) : result;
	}
	if (!Array.isArray(message.content)) return message;
	const content: unknown[] = [];
	for (let index = 0; index < message.content.length; index++) {
		const block: unknown = message.content[index];
		if (!block || typeof block !== "object") {
			if (selected.has(index)) content.push(block);
			continue;
		}
		if ("text" in block && typeof block.text === "string") {
			const sliced = slice(block.text, index, block);
			const emitted = { ...block, text: sliced.text };
			content.push(sliced.origin ? setSourceOrigin(emitted, sliced.origin) : emitted);
		} else if ("thinking" in block && typeof block.thinking === "string") {
			if (selected.has(index)) {
				const sliced = slice(block.thinking, index, block);
				const emitted = { ...block, thinking: sliced.text };
				content.push(sliced.origin ? setSourceOrigin(emitted, sliced.origin) : emitted);
			}
		} else if (selected.has(index)) content.push(block);
	}
	return content.length ? setSourceOrigin({ ...message, content } as AgentMessage, combineContentSourceOrigins(content)) : undefined;
}
