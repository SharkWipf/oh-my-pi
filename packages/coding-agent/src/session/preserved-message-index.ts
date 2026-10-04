export interface PolicySlot {
	user: boolean;
	raw: number;
	/** NaN marks an inadmissible configured candidate. */
	candidate: number;
	eligible: boolean;
	always: boolean;
	/** A manual Always user bypasses candidate pruning. */
	manual: boolean;
	/** Complete deduplicated non-user closure, at its first source position only. */
	nonUserCount: number;
}

export type PolicyLimit =
	| { mode: "off" }
	| { mode: "all" }
	| { mode: "messages" | "tokens" | "context-percent"; value: number };
export type PolicyKind = "raw" | "eligible" | "always";
export type PolicyDirection = "first" | "recent";
export interface PolicyRange {
	/** Half-open source-position bounds; membership also requires the query predicate. */
	start: number;
	end: number;
	count: number;
	tokens: number;
	blocker?: number;
}

const BLOCK_SIZE = 4096;
const USER = 1;
const ELIGIBLE = 2;
const ALWAYS = 4;
const MANUAL = 8;
const RAW_USER = 0;
const CANDIDATE_USER = 1;
const RAW_ALWAYS = 2;
const CANDIDATE_ALWAYS = 3;
const MANUAL_NON_USER = 4;

interface NonUserAtom {
	count: number;
	tokens: number;
}

class PolicyBlock {
	readonly flags = new Uint8Array(BLOCK_SIZE);
	// Only users have price columns; the directory maps source offsets to packed columns.
	userColumns: Uint16Array | undefined;
	raw = new Float64Array(0);
	candidate = new Float64Array(0);
	userLength = 0;
	readonly freeColumns: number[] = [];
	readonly nonUsers = new Map<number, NonUserAtom>();
	readonly counts = new Float64Array(5);
	readonly tokens = new Float64Array(5);
	// Optional block-level subtotal, not a sixth dense source lane. It makes disabling
	// automatic policy O(blocks) even when every user has a manual override.
	manualUsers: NonUserAtom | undefined;

	allocateUser(): number {
		const free = this.freeColumns.pop();
		if (free !== undefined) return free;
		this.userColumns ??= new Uint16Array(BLOCK_SIZE);
		if (this.userLength === this.raw.length) {
			const capacity = Math.min(BLOCK_SIZE, Math.max(16, this.raw.length * 2));
			const raw = new Float64Array(capacity);
			const candidate = new Float64Array(capacity);
			raw.set(this.raw);
			candidate.set(this.candidate);
			this.raw = raw;
			this.candidate = candidate;
		}
		return this.userLength++;
	}

	adjust(offset: number, sign: number): void {
		const flags = this.flags[offset];
		if (flags & USER) {
			const column = this.userColumns![offset];
			const raw = this.raw[column];
			const manual = (flags & MANUAL) !== 0;
			const candidate = manual ? raw : this.candidate[column];
			const viable = manual || ((flags & ELIGIBLE) !== 0 && !Number.isNaN(candidate));
			this.counts[RAW_USER] += sign;
			this.tokens[RAW_USER] += sign * raw;
			if (viable) {
				this.counts[CANDIDATE_USER] += sign;
				this.tokens[CANDIDATE_USER] += sign * candidate;
			}
			if (flags & ALWAYS) {
				this.counts[RAW_ALWAYS] += sign;
				this.tokens[RAW_ALWAYS] += sign * raw;
				if (viable) {
					this.counts[CANDIDATE_ALWAYS] += sign;
					this.tokens[CANDIDATE_ALWAYS] += sign * candidate;
				}
			}
			if (manual) {
				const totals = (this.manualUsers ??= { count: 0, tokens: 0 });
				totals.count += sign;
				totals.tokens += sign * raw;
				if (totals.count === 0) this.manualUsers = undefined;
			}
		} else {
			const atom = this.nonUsers.get(offset);
			if (atom) {
				this.counts[MANUAL_NON_USER] += sign * atom.count;
				this.tokens[MANUAL_NON_USER] += sign * atom.tokens;
			}
		}
	}
}

/** Source-policy measures only: no source objects, replay content, or selection caches. */
export class PreservedMessageIndex {
	readonly #blocks: PolicyBlock[] = [];
	#length = 0;

	get length(): number {
		return this.#length;
	}

	append(slot: PolicySlot): number {
		const position = this.#length;
		if (position % BLOCK_SIZE === 0) this.#blocks.push(new PolicyBlock());
		this.#length++;
		this.update(position, slot);
		return position;
	}

	update(position: number, slot: PolicySlot): void {
		if (!Number.isInteger(position) || position < 0 || position >= this.#length) {
			throw new RangeError("Policy position outside the index");
		}
		const block = this.#blocks[Math.floor(position / BLOCK_SIZE)];
		const offset = position % BLOCK_SIZE;
		block.adjust(offset, -1);
		const wasUser = (block.flags[offset] & USER) !== 0;
		if (slot.user) {
			const column = wasUser ? block.userColumns![offset] : block.allocateUser();
			block.userColumns![offset] = column;
			block.raw[column] = slot.raw;
			block.candidate[column] = slot.candidate;
			block.flags[offset] =
				USER |
				(slot.eligible ? ELIGIBLE : 0) |
				(slot.always || slot.manual ? ALWAYS : 0) |
				(slot.manual ? MANUAL : 0);
			block.nonUsers.delete(offset);
		} else {
			if (wasUser) block.freeColumns.push(block.userColumns![offset]);
			block.flags[offset] = 0;
			if (slot.nonUserCount > 0) {
				block.nonUsers.set(offset, { count: slot.nonUserCount, tokens: slot.raw });
			} else {
				block.nonUsers.delete(offset);
			}
		}
		block.adjust(offset, 1);
	}

	truncate(length: number): void {
		if (!Number.isInteger(length) || length < 0 || length > this.#length) {
			throw new RangeError("Policy truncation outside the index");
		}
		if (length === this.#length) return;
		const offset = length % BLOCK_SIZE;
		if (offset !== 0) {
			const block = this.#blocks[Math.floor(length / BLOCK_SIZE)];
			const end = Math.min(BLOCK_SIZE, this.#length - Math.floor(length / BLOCK_SIZE) * BLOCK_SIZE);
			for (let i = offset; i < end; i++) {
				block.adjust(i, -1);
				if (block.flags[i] & USER) block.freeColumns.push(block.userColumns![i]);
				block.flags[i] = 0;
				block.nonUsers.delete(i);
			}
		}
		this.#blocks.length = Math.ceil(length / BLOCK_SIZE);
		this.#length = length;
	}

	#price(block: PolicyBlock, offset: number, kind: PolicyKind, hard: boolean, automatic: boolean): number {
		const flags = block.flags[offset];
		if (!(flags & USER)) return kind === "always" ? (block.nonUsers.get(offset)?.tokens ?? NaN) : NaN;
		const column = block.userColumns![offset];
		if (!automatic) return kind === "always" && flags & MANUAL ? block.raw[column] : NaN;
		if (kind === "always" && !(flags & ALWAYS)) return NaN;
		if (kind === "raw" || hard || flags & MANUAL) return block.raw[column];
		return flags & ELIGIBLE ? block.candidate[column] : NaN;
	}

	query(
		limit: PolicyLimit,
		direction: PolicyDirection,
		kind: PolicyKind,
		hardBoundary = this.#length,
		automaticEnabled = true,
	): PolicyRange {
		const first = direction === "first";
		const edge = first ? 0 : this.#length;
		const result: PolicyRange = { start: edge, end: edge, count: 0, tokens: 0 };
		if (limit.mode === "off" || (!automaticEnabled && kind !== "always")) return result;
		if (limit.mode === "context-percent")
			throw new Error("Resolve context-percent against model maximum before querying");
		const quota = limit.mode === "all" ? Infinity : limit.value;
		const byCount = limit.mode === "messages";
		const step = first ? 1 : -1;
		for (let b = first ? 0 : this.#blocks.length - 1; b >= 0 && b < this.#blocks.length; b += step) {
			const block = this.#blocks[b];
			const start = b * BLOCK_SIZE;
			const end = Math.min(start + BLOCK_SIZE, this.#length);
			// H is virtual. Only its crossing block needs per-source inspection.
			if (kind === "raw" || !automaticEnabled || hardBoundary <= start || hardBoundary >= end) {
				const hard = hardBoundary <= start;
				const lane =
					kind === "raw"
						? RAW_USER
						: kind === "eligible"
							? hard
								? RAW_USER
								: CANDIDATE_USER
							: hard
								? RAW_ALWAYS
								: CANDIDATE_ALWAYS;
				let count = automaticEnabled ? block.counts[lane] : (block.manualUsers?.count ?? 0);
				let tokens = automaticEnabled ? block.tokens[lane] : (block.manualUsers?.tokens ?? 0);
				if (kind === "always") {
					count += block.counts[MANUAL_NON_USER];
					tokens += block.tokens[MANUAL_NON_USER];
				}
				if ((byCount ? result.count + count : result.tokens + tokens) <= quota) {
					result.count += count;
					result.tokens += tokens;
					if (first) result.end = end;
					else result.start = start;
					continue;
				}
			}
			for (let position = first ? start : end - 1; position >= start && position < end; position += step) {
				const offset = position - start;
				const tokens = this.#price(block, offset, kind, position >= hardBoundary, automaticEnabled);
				if (!Number.isNaN(tokens)) {
					const count = block.flags[offset] & USER ? 1 : block.nonUsers.get(offset)!.count;
					if ((byCount ? result.count + count : result.tokens + tokens) > quota) {
						result.blocker = position;
						return result;
					}
					result.count += count;
					result.tokens += tokens;
				}
				if (first) result.end = position + 1;
				else result.start = position;
			}
		}
		return result;
	}

	/** Measures a source union without enumerating selected users or double charging overlap. */
	measureUnion(
		ranges: readonly { range: PolicyRange; kind: PolicyKind }[],
		hardBoundary = this.#length,
		automaticEnabled = true,
		userOnly = true,
	): { count: number; tokens: number } {
		const cuts = [0, this.#length, Math.max(0, Math.min(this.#length, hardBoundary))];
		for (const { range } of ranges) {
			cuts.push(Math.max(0, Math.min(this.#length, range.start)), Math.max(0, Math.min(this.#length, range.end)));
		}
		cuts.sort((a, b) => a - b);
		let count = 0;
		let tokens = 0;
		for (let i = 1; i < cuts.length; i++) {
			const start = cuts[i - 1];
			const end = cuts[i];
			if (start === end) continue;
			let raw = false;
			let eligible = false;
			let always = false;
			for (const selection of ranges) {
				if (selection.range.start > start || selection.range.end < end) continue;
				if (selection.kind === "raw") raw = automaticEnabled;
				else if (selection.kind === "eligible") eligible = automaticEnabled;
				else always = true;
			}
			if (!raw && !eligible && !always) continue;
			const kind: PolicyKind = raw ? "raw" : eligible ? "eligible" : "always";
			const hard = start >= hardBoundary;
			const lane =
				kind === "raw"
					? RAW_USER
					: kind === "eligible"
						? hard
							? RAW_USER
							: CANDIDATE_USER
						: hard
							? RAW_ALWAYS
							: CANDIDATE_ALWAYS;
			const includeNonUsers = !userOnly && always;
			for (let position = start; position < end;) {
				const b = Math.floor(position / BLOCK_SIZE);
				const block = this.#blocks[b];
				const blockStart = b * BLOCK_SIZE;
				const blockEnd = Math.min(blockStart + BLOCK_SIZE, this.#length);
				if (position === blockStart && blockEnd <= end) {
					count += automaticEnabled ? block.counts[lane] : (block.manualUsers?.count ?? 0);
					tokens += automaticEnabled ? block.tokens[lane] : (block.manualUsers?.tokens ?? 0);
					if (includeNonUsers) {
						count += block.counts[MANUAL_NON_USER];
						tokens += block.tokens[MANUAL_NON_USER];
					}
					position = blockEnd;
					continue;
				}
				const until = Math.min(blockEnd, end);
				for (; position < until; position++) {
					const offset = position - blockStart;
					if (block.flags[offset] & USER) {
						const price = this.#price(block, offset, kind, hard, automaticEnabled);
						if (!Number.isNaN(price)) {
							count++;
							tokens += price;
						}
					} else if (includeNonUsers) {
						const atom = block.nonUsers.get(offset);
						if (atom) {
							count += atom.count;
							tokens += atom.tokens;
						}
					}
				}
			}
		}
		return { count, tokens };
	}

	includes(
		position: number,
		range: PolicyRange,
		kind: PolicyKind,
		hardBoundary = this.#length,
		automaticEnabled = true,
	): boolean {
		if (
			!Number.isInteger(position) ||
			position < range.start ||
			position >= range.end ||
			position < 0 ||
			position >= this.#length
		)
			return false;
		return !Number.isNaN(
			this.#price(
				this.#blocks[Math.floor(position / BLOCK_SIZE)],
				position % BLOCK_SIZE,
				kind,
				position >= hardBoundary,
				automaticEnabled,
			),
		);
	}

	/** Enumerates anchors in source order; the source owner expands complete N closures. */
	*iterate(
		range: PolicyRange,
		kind: PolicyKind,
		hardBoundary = this.#length,
		automaticEnabled = true,
	): IterableIterator<number> {
		for (let position = Math.max(0, range.start); position < Math.min(this.#length, range.end); position++) {
			if (this.includes(position, range, kind, hardBoundary, automaticEnabled)) yield position;
		}
	}
}
