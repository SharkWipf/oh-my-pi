/**
 * Shared engine for the ephemeral fullscreen transcript selectors (esc-esc
 * rewind, `/copy`): replay session entries through {@link ChatTranscriptBuilder},
 * map each rendered turn to a selectable target, and compose gutter-prefixed
 * columns with a dotted outline around the selected target.
 */
import type { Component } from "@oh-my-pi/pi-tui";
import { visibleWidth } from "@oh-my-pi/pi-tui";
import type { SessionMessageEntry } from "../../session/session-entries";
import { type ThemeColor, theme } from "../theme/theme";
import type { ChatTranscriptBuilder } from "./chat-transcript-builder";
import { fit } from "./overlay-box";
import { isUsageRowBlock } from "./usage-row";

/** One selectable transcript item: a message entry plus its rendered block range. */
export interface OutlineTarget {
	/** Entry the selection resolves to; extended over trailing componentless tool results. */
	entryId: string;
	/** Entry that opened this turn (pre-fold) — anchor for tree lookups. */
	turnId: string;
	/** Real user prompt — drives user-turn jumps and role-specific actions. */
	isUserTurn: boolean;
	/** First transcript-container child rendered by this entry. */
	start: number;
	/** One past the last child rendered by this entry. */
	end: number;
	/** Entries this target spans: the turn plus any folded tool results. */
	entries: SessionMessageEntry[];
}

/** Composed rows of one column plus the outline's line range within them. */
export interface ComposedColumn {
	lines: string[];
	selStart: number;
	selEnd: number;
}
/** Presentation of the dotted outline: stroke color and an optional caption inset into the top rule. */
export interface OutlineStyle {
	color?: ThemeColor;
	/** Short affordance label (e.g. "3 blocks →") drawn into the top rule's right end. */
	caption?: string;
}

// User bubbles wrap their rows in OSC 133 prompt-zone marks (see
// user-message.ts). Re-emitting those inside the alternate-screen overlay
// latches the terminal's prompt semantics onto overlay rows and garbles the
// frame, so embedded rows shed them; the transcript proper keeps its zones.
const OSC133_SPAN_REGEX = /\x1b\]133;[^\x07\x1b]*(?:\x07|\x1b\\)/g;

/** Copy-on-write removal of OSC 133 spans from a rendered row array. */
export function stripPromptZones(rows: readonly string[]): readonly string[] {
	let sanitized: string[] | undefined;
	for (let index = 0; index < rows.length; index++) {
		if (!rows[index]!.includes("\x1b]133;")) continue;
		sanitized ??= rows.slice();
		sanitized[index] = rows[index]!.replace(OSC133_SPAN_REGEX, "");
	}
	return sanitized ?? rows;
}

/**
 * Prompt-zone-stripped rows for one selector column.
 *
 * The selectors recompose their whole column on every keystroke, and stripping
 * every child's rows again per frame is pure waste on a long session. Per the
 * {@link Component} render contract a child returns the same array while its
 * rows are unchanged, so the stripped copy is keyed on that array: a child that
 * repaints itself asynchronously (Kitty image conversion, todo strike frames)
 * hands back a new array and is stripped again.
 */
export class OutlineRowCache {
	#stripped = new WeakMap<Component, { rows: readonly string[]; stripped: readonly string[] }>();

	/** Drop rows when a lazy transcript releases the corresponding component. */
	forget(child: Component): void {
		this.#stripped.delete(child);
	}

	rows(children: readonly Component[], width: number): Array<readonly string[]> {
		const columns: Array<readonly string[]> = [];
		for (const child of children) {
			const rows = child.render(width);
			const cached = this.#stripped.get(child);
			if (cached && cached.rows === rows) {
				columns.push(cached.stripped);
				continue;
			}
			const stripped = stripPromptZones(rows);
			this.#stripped.set(child, { rows, stripped });
			columns.push(stripped);
		}
		return columns;
	}
}

/**
 * Append `entries` to `builder`, returning the selectable targets they
 * produce. Tool results fold into the previous target (so a turn's rewind
 * and `/copy` keep its output, including lazily created children such as
 * the grouped-read card); notices and hidden messages that render nothing
 * are skipped. Usage rows flushed at the head of an append are attributed
 * to the turn above.
 */
export function appendOutlineEntries(builder: ChatTranscriptBuilder, entries: SessionMessageEntry[]): OutlineTarget[] {
	const targets: OutlineTarget[] = [];
	for (const entry of entries) appendOutlineEntry(builder, entry, targets);
	return targets;
}

/** Append one source while retaining target folding across cooperative yields. */
export function appendOutlineEntry(builder: ChatTranscriptBuilder, entry: SessionMessageEntry, targets: OutlineTarget[]): void {
	const children = builder.container.children;
	const before = children.length;
	builder.append([entry]);
	const after = children.length;
	let start = before;
	while (start < after && isUsageRowBlock(children[start]!)) {
		const previous = targets.at(-1);
		if (previous && previous.end === start) previous.end = start + 1;
		start++;
	}
	const previous = targets.at(-1);
	if (entry.message.role === "toolResult" && previous) {
		previous.entryId = entry.id;
		previous.entries.push(entry);
		if (after > previous.end) previous.end = after;
		return;
	}
	if (start >= after) return;
	targets.push({
		entryId: entry.id,
		turnId: entry.id,
		isUserTurn: entry.message.role === "user" && userMessageHasText(entry.message),
		start,
		end: after,
		entries: [entry],
	});
}

/** Per-target "renders at least one non-blank row" flags at the given rows. */
export function outlineVisibility(
	childRows: readonly (readonly string[])[],
	targets: readonly OutlineTarget[],
): boolean[] {
	return targets.map(target => {
		for (let index = target.start; index < target.end; index++) {
			if (childRows[index]!.some(row => /\S/.test(row))) return true;
		}
		return false;
	});
}

/** Dotted horizontal rule with rounded corners, spanning the outline width. */
export function outlineRule(
	left: string,
	right: string,
	innerWidth: number,
	color: ThemeColor = "accent",
	caption?: string,
): string {
	const label = caption ? ` ${caption} ` : "";
	const fill = Math.max(0, innerWidth + 2 - visibleWidth(label));
	return (
		theme.fg(color, left + theme.boxDotted.horizontal.repeat(fill)) +
		theme.bold(theme.fg(color, label)) +
		theme.fg(color, right)
	);
}

/** Wrap pre-rendered rows in the dotted outline (rules above/below, `┆` sides). */
export function outlineRows(rows: readonly string[], innerWidth: number, style: OutlineStyle = {}): string[] {
	const color = style.color ?? "accent";
	const vertical = theme.fg(color, theme.boxDotted.vertical);
	const lines: string[] = [
		outlineRule(theme.boxRound.topLeft, theme.boxRound.topRight, innerWidth, color, style.caption),
	];
	for (const row of rows) lines.push(`${vertical} ${fit(row, innerWidth)} ${vertical}`);
	lines.push(outlineRule(theme.boxRound.bottomLeft, theme.boxRound.bottomRight, innerWidth, color));
	return lines;
}

/**
 * Compose one column: gutter-prefixed rows for `childRows[from, to)` with a
 * dotted outline around `targets[selected]`. `header` rows, when given, lead
 * the column.
 */
export function composeOutlineColumn(
	childRows: readonly (readonly string[])[],
	from: number,
	to: number,
	targets: readonly OutlineTarget[],
	selected: number,
	columnWidth: number,
	header: string[] | undefined,
	style: OutlineStyle = {},
): ComposedColumn {
	const inner = Math.max(10, columnWidth - 4);
	const lines: string[] = header ? [...header] : [];
	let selStart = -1;
	let selEnd = -1;
	const target = selected >= 0 ? targets[selected] : undefined;
	for (let index = from; index < to; index++) {
		if (target && index === target.start && target.end <= to) {
			const segment: string[] = [];
			for (let child = target.start; child < target.end; child++) segment.push(...childRows[child]!);
			// Outline only the non-blank core; edge spacers stay outside.
			let head = 0;
			let tail = segment.length;
			while (head < tail && !/\S/.test(segment[head]!)) head++;
			while (tail > head && !/\S/.test(segment[tail - 1]!)) tail--;
			for (let row = 0; row < head; row++) lines.push("");
			selStart = lines.length;
			lines.push(...outlineRows(segment.slice(head, tail), inner, style));
			selEnd = lines.length;
			for (let row = tail; row < segment.length; row++) lines.push("");
			index = target.end - 1;
			continue;
		}
		for (const row of childRows[index]!) lines.push(row ? `  ${row}` : row);
	}
	return { lines, selStart, selEnd };
}

/** Centered position rail for horizontally windowed content: `… ○ ◉ ○ …`. */
export function positionRail(
	count: number,
	active: number,
	moreLeft: boolean,
	moreRight: boolean,
	width: number,
): string {
	const dots: string[] = [];
	for (let index = 0; index < count; index++) {
		dots.push(index === active ? theme.fg("accent", theme.radio.selected) : theme.fg("dim", theme.radio.unselected));
	}
	const rail = `${moreLeft ? theme.fg("dim", "… ") : "  "}${dots.join(" ")}${moreRight ? theme.fg("dim", " …") : ""}`;
	const pad = Math.max(0, Math.floor((width - visibleWidth(rail)) / 2));
	return " ".repeat(pad) + rail;
}

/** Plain text of a user message (string or text blocks), single line. */
export function userMessageText(message: Extract<SessionMessageEntry["message"], { role: "user" }>): string {
	const text =
		typeof message.content === "string"
			? message.content
			: message.content
					.filter((block): block is { type: "text"; text: string } => block.type === "text")
					.map(block => block.text)
					.join(" ");
	return text.replace(/\s+/g, " ").trim();
}

/** Whether a user message carries prompt text (string or text blocks). */
export function userMessageHasText(message: SessionMessageEntry["message"]): boolean {
	if (message.role !== "user") return false;
	if (typeof message.content === "string") return message.content.trim().length > 0;
	return message.content.some(block => block.type === "text" && block.text.trim().length > 0);
}

/**
 * Row-demand viewport over the builder's source-owned children. The anchor is a
 * child and an intra-child row, never an estimated full-transcript row number.
 * Only traversed children are rendered; the rail reports more history rather
 * than inventing a global row count before cold history has been measured.
 */
export class OutlineViewport {
	#cache = new OutlineRowCache();
	#retained = new Set<Component>();
	#touched = new Set<Component>();
	#shown = new Set<Component>();
	#rows = new Map<number, readonly string[]>();
	#children: readonly Component[] = [];
	#target: OutlineTarget | undefined;
	#from = 0;
	#to = 0;
	#width = 80;
	#height = 1;
	#child = 0;
	#row = 0;
	#lastEnd = 0;
	#initialized = false;
	moreAbove = false;
	moreBelow = false;

	configure(children: readonly Component[], target: OutlineTarget | undefined, width: number, height: number,
		from = 0, to = children.length): void {
		this.#children = children;
		this.#target = target;
		this.#width = width;
		this.#height = Math.max(0, height);
		this.#from = from;
		this.#to = to;
		this.#child = Math.max(from, Math.min(this.#child, to));
		this.#rows.clear();
		this.#touched.clear();
	}

	/** Examine a candidate on demand; invisible targets do not become stops. */
	visible(target: OutlineTarget): boolean {
		for (let child = target.start; child < target.end; child++) {
			if (this.#raw(child).some(row => /\S/.test(row))) return true;
		}
		return false;
	}

	#raw(index: number): readonly string[] {
		const child = this.#children[index];
		if (!child) return [];
		this.#touched.add(child);
		return this.#cache.rows([child], Math.max(10, this.#width - 4))[0]!;
	}

	#unit(index: number): number {
		const target = this.#target;
		return target && index >= target.start && index < target.end ? target.start : index;
	}

	#next(index: number): number {
		return index === this.#target?.start ? this.#target.end : index + 1;
	}

	#block(index: number): readonly string[] {
		const cached = this.#rows.get(index);
		if (cached) return cached;
		let rows: string[];
		if (index === this.#target?.start) {
			const source: string[] = [];
			for (let child = index; child < this.#target.end; child++) source.push(...this.#raw(child));
			let head = 0;
			let tail = source.length;
			while (head < tail && !/\S/.test(source[head]!)) head++;
			while (tail > head && !/\S/.test(source[tail - 1]!)) tail--;
			rows = source.slice(0, head);
			if (head < tail) rows.push(...outlineRows(source.slice(head, tail), Math.max(10, this.#width - 4)));
			rows.push(...source.slice(tail));
		} else {
			rows = this.#raw(index).map(row => row ? `  ${row}` : row);
		}
		this.#rows.set(index, rows);
		return rows;
	}

	#forward(amount: number): void {
		while (this.#child < this.#to) {
			this.#child = this.#unit(this.#child);
			const length = this.#block(this.#child).length;
			const remaining = Math.max(0, length - this.#row);
			if (amount < remaining) { this.#row += amount; return; }
			amount -= remaining;
			this.#child = this.#next(this.#child);
			this.#row = 0;
			if (amount === 0 && this.#child < this.#to && this.#block(this.#child).length > 0) return;
		}
	}

	#backward(amount: number): void {
		while (amount > 0) {
			const within = Math.min(amount, this.#row);
			this.#row -= within;
			amount -= within;
			if (amount === 0 || this.#child <= this.#from) return;
			this.#child = this.#unit(this.#child - 1);
			this.#row = this.#block(this.#child).length;
		}
	}

	/** Follow the selected source without traversing unrelated preceding history. */
	follow(): void {
		const target = this.#target;
		if (!target) return;
		if (this.#initialized && target.start >= this.#child && target.end < this.#lastEnd) return;
		this.#child = target.start;
		this.#row = this.#block(target.start).length;
		this.#backward(this.#height);
		this.#initialized = true;
	}

	home(): boolean {
		const changed = this.#child !== this.#from || this.#row !== 0;
		this.#child = this.#from;
		this.#row = 0;
		this.#initialized = true;
		return changed;
	}

	end(): boolean {
		const child = this.#child;
		const row = this.#row;
		this.#child = this.#to;
		this.#row = 0;
		this.#backward(this.#height);
		this.#initialized = true;
		return child !== this.#child || row !== this.#row;
	}

	scroll(delta: number): boolean {
		const child = this.#child;
		const row = this.#row;
		if (delta > 0) this.#forward(delta);
		else this.#backward(-delta);
		// Clamp against the actual trailing rows, not a guessed transcript size.
		const available = this.#collect().length;
		if (available < this.#height) this.#backward(this.#height - available);
		this.#initialized = true;
		return child !== this.#child || row !== this.#row;
	}

	#collect(): string[] {
		const lines: string[] = [];
		this.#shown.clear();
		let child = this.#unit(this.#child);
		let row = this.#row;
		while (child < this.#to && lines.length < this.#height) {
			const block = this.#block(child);
			const count = Math.max(0, Math.min(block.length - row, this.#height - lines.length));
			if (count > 0) {
				for (let index = child; index < this.#next(child); index++) this.#shown.add(this.#children[index]!);
			}
			for (let i = 0; i < count; i++) lines.push(block[row + i]!);
			row += count;
			if (row < block.length) break;
			child = this.#next(child);
			row = 0;
		}
		this.#lastEnd = child;
		this.moreAbove = this.#child > this.#from || this.#row > 0;
		this.moreBelow = child < this.#to;
		return lines;
	}

	render(): string[] {
		if (!this.#initialized) this.home();
		return this.#collect();
	}

	/** Release candidate probes and offscreen rows, retaining only the painted units. */
	finish(): ReadonlySet<Component> {
		for (const child of this.#retained) if (!this.#shown.has(child)) this.#cache.forget(child);
		for (const child of this.#touched) if (!this.#shown.has(child)) this.#cache.forget(child);
		this.#retained = new Set(this.#shown);
		this.#rows.clear();
		return this.#retained;
	}
}

