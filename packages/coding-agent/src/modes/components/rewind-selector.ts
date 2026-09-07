/** Fullscreen source-anchored rewind over the shared transcript builder. */
import type { AgentTool } from "@oh-my-pi/pi-agent-core";
import {
	type Component,
	matchesKey,
	padding,
	routeSgrMouseInput,
	sliceByColumn,
	type TUI,
	truncateToWidth,
} from "@oh-my-pi/pi-tui";
import type { MessageRenderer } from "../../extensibility/extensions/types";
import type { SessionMessageEntry } from "../../session/session-entries";
import { theme } from "../theme/theme";
import { matchesAppToolsExpand, matchesSelectCancel, matchesSelectDown, matchesSelectUp } from "../utils/keybinding-matchers";
import { ChatTranscriptBuilder } from "./chat-transcript-builder";
import { DynamicBorder } from "./dynamic-border";
import { fit } from "./overlay-box";
import {
	appendOutlineEntry,
	type OutlineTarget,
	OutlineViewport,
	positionRail,
	userMessageHasText,
	userMessageText,
} from "./transcript-outline";

/** One alternate branch at a divergence: its root and message path root → most-recent leaf. */
export interface BranchVariantPath {
	rootId: string;
	entries: SessionMessageEntry[];
}

export interface RewindSelectorDeps {
	ui: TUI;
	getTool?: (name: string) => AgentTool | undefined;
	isBuiltInTool?: (name: string) => boolean;
	getMessageRenderer?: (customType: string) => MessageRenderer | undefined;
	cwd: string;
	hideThinkingBlock?: () => boolean;
	proseOnlyThinking?: () => boolean;
	linkTargets?: ReadonlyMap<string, string>;
	requestRender: () => void;
	siblingPaths?: (entryId: string, signal: AbortSignal) => BranchVariantPath[] | Promise<BranchVariantPath[]>;
	onSelect: (entryId: string) => void;
	onCancel: () => void;
}

interface SiblingColumn extends BranchVariantPath {
	builder?: ChatTranscriptBuilder;
	targets?: OutlineTarget[];
	viewport: OutlineViewport;
	label: string;
	loading?: boolean;
	cancelled?: boolean;
	error?: string;
}

const CHROME_ROWS = 5;
const STRIP_GAP = 2;
const SLIDE_MS = 160;
const NO_COMPONENTS: ReadonlySet<Component> = new Set();

export class RewindSelectorComponent implements Component {
	#builder: ChatTranscriptBuilder;
	#border = new DynamicBorder();
	#targets: OutlineTarget[] = [];
	#selected = 0;
	#loading = true;
	#cancelled = false;
	#indexed = 0;
	#sourceCount = 0;
	readonly ready: Promise<void>;
	#viewport = new OutlineViewport();
	#prefixViewport = new OutlineViewport();
	#stripViewport = new OutlineViewport();
	#scrollToSelection = true;
	#expanded = false;
	#width = 80;
	#height = 35;
	#stripRoot: string | undefined;
	#columns: SiblingColumn[] = [];
	#activeVariant = 0;
	#siblingSelected = 0;
	#slide: { from: number; to: number; startedAt: number } | undefined;
	#slideTimer: NodeJS.Timeout | undefined;
	#branchWork: AbortController | undefined;
	#branchesLoading = false;
	#branchError: string | undefined;

	constructor(entries: SessionMessageEntry[], private readonly deps: RewindSelectorDeps) {
		this.#builder = this.#newBuilder();
		this.#sourceCount = entries.length;
		this.ready = this.#buildIndex(entries);
	}

	get isLoading(): boolean { return this.#loading; }

	async #buildIndex(entries: SessionMessageEntry[]): Promise<void> {
		let deadline = performance.now() + 8;
		for (const entry of entries) {
			if (this.#cancelled) return;
			appendOutlineEntry(this.#builder, entry, this.#targets);
			this.#indexed++;
			if (performance.now() >= deadline) {
				this.deps.requestRender();
				await new Promise<void>(resolve => setImmediate(resolve));
				deadline = performance.now() + 8;
			}
		}
		if (this.#cancelled) return;
		this.#selected = Math.max(0, this.#targets.length - 1);
		this.#loading = false;
		this.#scrollToSelection = true;
		this.deps.requestRender();
	}

	/** Keep the last bounded viewport warm; cancel only incomplete source work. */
	suspend(): void {
		this.#stopSlide();
		this.#branchWork?.abort();
		if (this.#branchesLoading || this.#columns.some(column => column.loading)) {
			for (const column of this.#columns) { column.cancelled = true; column.builder?.dispose(); }
			this.#columns = [];
			this.#stripRoot = undefined;
			this.#branchesLoading = false;
		}
	}

	/** Reopening starts at the recent tail, without replaying unchanged sources. */
	resume(onSelect: (entryId: string) => void, onCancel: () => void): void {
		this.deps.onSelect = onSelect;
		this.deps.onCancel = onCancel;
		this.#selected = Math.max(0, this.#targets.length - 1);
		this.#activeVariant = 0;
		this.#siblingSelected = 0;
		if (this.#expanded) {
			this.#builder.setExpanded(false);
			for (const column of this.#columns) column.builder?.setExpanded(false);
		}
		this.#expanded = false;
		this.#scrollToSelection = true;
	}

	get targetCount(): number { return this.#targets.length; }

	#newBuilder(): ChatTranscriptBuilder {
		return new ChatTranscriptBuilder({
			ui: this.deps.ui,
			getTool: this.deps.getTool,
			isBuiltInTool: this.deps.isBuiltInTool,
			getMessageRenderer: this.deps.getMessageRenderer,
			cwd: this.deps.cwd,
			hideThinkingBlock: this.deps.hideThinkingBlock,
			proseOnlyThinking: this.deps.proseOnlyThinking,
			linkTargets: this.deps.linkTargets,
			requestRender: this.deps.requestRender,
			deferComponents: true,
		});
	}

	invalidate(): void {
		this.#builder.container.invalidate();
		for (const column of this.#columns) column.builder?.container.invalidate();
		this.#scrollToSelection = true;
	}

	dispose(): void {
		this.#cancelled = true;
		this.#stopSlide();
		this.#branchWork?.abort();
		for (const column of this.#columns) { column.cancelled = true; column.builder?.dispose(); }
		this.#builder.dispose();
		this.#columns = [];
	}

	#stripColumns(): SiblingColumn[] {
		const target = this.#targets[this.#selected];
		if (!target || !this.deps.siblingPaths) return [];
		if (this.#stripRoot === target.turnId) return this.#columns;
		this.#branchWork?.abort();
		for (const column of this.#columns) { column.cancelled = true; column.builder?.dispose(); }
		this.#stripRoot = target.turnId;
		this.#columns = [];
		this.#stripViewport = new OutlineViewport();
		this.#branchError = undefined;
		const work = new AbortController();
		this.#branchWork = work;
		const accept = (paths: BranchVariantPath[]) => {
			if (work.signal.aborted) return;
			this.#branchesLoading = false;
			for (const sibling of paths) {
				if (sibling.entries.length === 0) continue;
				// A branch caption need not scan an entire cold assistant run.
				const first = sibling.entries[0]!;
				const label = first.message.role === "user" && userMessageHasText(first.message) ? userMessageText(first.message) : sibling.rootId;
				this.#columns.push({ ...sibling, label, viewport: new OutlineViewport() });
			}
			this.#scrollToSelection = true;
			this.deps.requestRender();
		};
		const paths = this.deps.siblingPaths(target.turnId, work.signal);
		if (paths instanceof Promise) {
			this.#branchesLoading = true;
			void paths.then(accept, error => {
				if (work.signal.aborted) return;
				this.#branchesLoading = false;
				this.#branchError = String(error);
				this.deps.requestRender();
			});
		} else accept(paths);
		return this.#columns;
	}

	#materialize(column: SiblingColumn): { builder: ChatTranscriptBuilder; targets: OutlineTarget[] } {
		if (!column.builder) {
			column.builder = this.#newBuilder();
			column.builder.setExpanded(this.#expanded);
			column.targets = [];
			column.loading = true;
			void this.#buildColumn(column).catch(error => {
				if (column.cancelled) return;
				column.loading = false;
				column.error = String(error);
				this.deps.requestRender();
			});
		}
		return { builder: column.builder, targets: column.targets! };
	}

	async #buildColumn(column: SiblingColumn): Promise<void> {
		let deadline = performance.now() + 8;
		for (const entry of column.entries) {
			if (column.cancelled) return;
			appendOutlineEntry(column.builder!, entry, column.targets!);
			if (performance.now() >= deadline) {
				await new Promise<void>(resolve => setImmediate(resolve));
				deadline = performance.now() + 8;
			}
		}
		if (column.cancelled) return;
		column.loading = false;
		this.#scrollToSelection = true;
		this.deps.requestRender();
	}

	#outlinedTarget(): OutlineTarget | undefined {
		if (this.#activeVariant > 0) {
			const column = this.#stripColumns()[this.#activeVariant - 1];
			if (!column) return undefined;
			const { targets } = this.#materialize(column);
			return column.loading || column.error ? undefined : targets[this.#siblingSelected];
		}
		return this.#targets[this.#selected];
	}

	#slideTo(variant: number): void {
		const now = Date.now();
		this.#slide = { from: this.#slidePosition(now), to: variant, startedAt: now };
		this.#activeVariant = variant;
		this.#scrollToSelection = true;
		this.#slideTimer ??= setInterval(() => {
			if (!this.#slide || Date.now() - this.#slide.startedAt >= SLIDE_MS) this.#stopSlide();
			this.deps.requestRender();
		}, 16);
		this.deps.requestRender();
	}

	#stopSlide(): void {
		this.#slide = undefined;
		clearInterval(this.#slideTimer);
		this.#slideTimer = undefined;
	}

	#slidePosition(now: number): number {
		if (!this.#slide) return this.#activeVariant;
		const t = Math.min(1, (now - this.#slide.startedAt) / SLIDE_MS);
		return this.#slide.from + (this.#slide.to - this.#slide.from) * (1 - (1 - t) ** 3);
	}

	#activeViewport(): OutlineViewport {
		const columns = this.#stripColumns();
		if (columns.length === 0) return this.#viewport;
		return this.#activeVariant === 0 ? this.#stripViewport : columns[this.#activeVariant - 1]!.viewport;
	}

	handleInput(data: string): void {
		if (this.#loading) {
			if (matchesSelectCancel(data) || matchesKey(data, "escape")) this.deps.onCancel();
			return;
		}
		if (data.startsWith("\x1b[<")) {
			routeSgrMouseInput(data, event => {
				if (event.wheel !== null && this.#activeViewport().scroll(event.wheel * 3)) this.deps.requestRender();
				return true;
			});
			return;
		}
		if (matchesSelectCancel(data) || matchesKey(data, "escape")) { this.deps.onCancel(); return; }
		if (matchesAppToolsExpand(data)) {
			this.#expanded = !this.#expanded;
			this.#builder.setExpanded(this.#expanded);
			for (const column of this.#columns) column.builder?.setExpanded(this.#expanded);
			this.#scrollToSelection = true;
			this.deps.requestRender();
			return;
		}
		if (matchesSelectUp(data)) { this.#moveVertical(-1); return; }
		if (matchesSelectDown(data)) { this.#moveVertical(1); return; }
		if (matchesKey(data, "left")) {
			if (this.#activeVariant > 0) this.#slideTo(this.#activeVariant - 1);
			else this.#move(-1, target => target.isUserTurn);
			return;
		}
		if (matchesKey(data, "right")) {
			const columns = this.#stripColumns();
			if (this.#branchesLoading) return;
			if (this.#activeVariant < columns.length) {
				this.#siblingSelected = 0;
				this.#slideTo(this.#activeVariant + 1);
			} else if (this.#activeVariant === 0) this.#move(1, target => target.isUserTurn);
			return;
		}
		if (matchesKey(data, "enter") || matchesKey(data, "return") || data === "\n") {
			const target = this.#outlinedTarget();
			if (target) this.deps.onSelect(target.entryId);
			return;
		}
		const viewport = this.#activeViewport();
		let changed = false;
		if (matchesKey(data, "home")) changed = viewport.home();
		else if (matchesKey(data, "end")) changed = viewport.end();
		else if (matchesKey(data, "pageUp")) changed = viewport.scroll(-Math.max(1, this.#height - 1));
		else if (matchesKey(data, "pageDown")) changed = viewport.scroll(Math.max(1, this.#height - 1));
		else if (matchesKey(data, "shift+up")) changed = viewport.scroll(-5);
		else if (matchesKey(data, "shift+down")) changed = viewport.scroll(5);
		if (changed) this.deps.requestRender();
	}

	#moveVertical(delta: -1 | 1): void {
		if (this.#activeVariant === 0) { this.#move(delta, () => true); return; }
		const column = this.#stripColumns()[this.#activeVariant - 1]!;
		const { builder, targets } = this.#materialize(column);
		if (column.loading || column.error) return;
		column.viewport.configure(builder.container.children, targets[this.#siblingSelected], this.#columnWidth(), this.#height);
		let index = this.#siblingSelected + delta;
		while (index >= 0 && index < targets.length && !column.viewport.visible(targets[index]!)) index += delta;
		if (index >= 0 && index < targets.length) {
			this.#siblingSelected = index;
			this.#scrollToSelection = true;
			this.deps.requestRender();
		} else if (delta === -1) {
			this.#activeVariant = 0;
			this.#siblingSelected = 0;
			this.#stopSlide();
			this.#move(-1, () => true);
		}
	}

	#move(delta: -1 | 1, accept: (target: OutlineTarget) => boolean): void {
		this.#viewport.configure(this.#builder.container.children, this.#targets[this.#selected], this.#width - 1, this.#height);
		let index = this.#selected + delta;
		while (index >= 0 && index < this.#targets.length) {
			const target = this.#targets[index]!;
			if (accept(target) && this.#viewport.visible(target)) {
				this.#selected = index;
				this.#activeVariant = 0;
				this.#siblingSelected = 0;
				this.#stopSlide();
				this.#scrollToSelection = true;
				this.deps.requestRender();
				return;
			}
			index += delta;
		}
	}

	render(width: number): readonly string[] {
		if (this.#loading) {
			const height = Math.max(CHROME_ROWS, process.stdout.rows || 40);
			return [
				...this.#border.render(width),
				" Rewind — indexing source messages",
				...this.#border.render(width),
				fit(" " + this.#indexed + "/" + this.#sourceCount + " sources indexed; Esc cancels", width),
				...Array.from({ length: Math.max(0, height - 5) }, () => ""),
				...this.#border.render(width),
			];
		}
		if (width !== this.#width || this.#height !== Math.max(3, (process.stdout.rows || 40) - CHROME_ROWS)) this.#scrollToSelection = true;
		this.#width = width;
		this.#height = Math.max(3, (process.stdout.rows || 40) - CHROME_ROWS);
		const contentWidth = Math.max(1, width - 1);
		const children = this.#builder.container.children;
		this.#viewport.configure(children, this.#targets[this.#selected], contentWidth, this.#height);
		if (this.#targets.length > 0 && !this.#viewport.visible(this.#targets[this.#selected]!)) {
			let above = this.#selected - 1;
			while (above >= 0 && !this.#viewport.visible(this.#targets[above]!)) above--;
			let below = this.#selected + 1;
			while (above < 0 && below < this.#targets.length && !this.#viewport.visible(this.#targets[below]!)) below++;
			if (above >= 0) this.#selected = above;
			else if (below < this.#targets.length) this.#selected = below;
			this.#viewport.configure(children, this.#targets[this.#selected], contentWidth, this.#height);
		}
		const columns = this.#stripColumns();
		let lines: string[];
		if (columns.length > 0) lines = this.#renderStrip(columns, contentWidth);
		else {
			if (this.#scrollToSelection) this.#viewport.follow();
			lines = this.#viewport.render();
			this.#builder.releaseOutside(this.#viewport.finish());
		}
		this.#scrollToSelection = false;
		const viewport = this.#activeViewport();
		const output: string[] = [...this.#border.render(width)];
		output.push(` ${theme.icon.rewind} ${theme.bold("Rewind")}${theme.sep.dot}${theme.fg("dim", "pick the point to continue from")}`);
		output.push(...this.#border.render(width));
		for (let row = 0; row < this.#height; row++) {
			const rail = row === 0 && viewport.moreAbove ? "↑" : row === this.#height - 1 && viewport.moreBelow ? "↓" : " ";
			output.push(fit(lines[row] ?? "", contentWidth) + theme.fg("dim", rail));
		}
		const position = this.#targets.length > 0 ? `${this.#selected + 1}/${this.#targets.length}  ` : "";
		const lateral = this.#branchesLoading ? "loading branches" : this.#branchError ? "branch error: " + this.#branchError : columns.length > 0 ? "←/→ branches" : "←/→ user turns";
		output.push(` ${theme.fg("dim", `${position}↑/↓ step  ${lateral}  enter rewind  ctrl+o expand  esc cancel`)}`);
		output.push(...this.#border.render(width));
		return output;
	}

	#columnWidth(): number { return Math.max(24, Math.floor((this.#width - 1 - STRIP_GAP) / 2)); }

	#renderStrip(columns: SiblingColumn[], contentWidth: number): string[] {
		const anchor = this.#targets[this.#selected]!;
		const colWidth = this.#columnWidth();
		const count = columns.length + 1;
		const stride = colWidth + STRIP_GAP;
		const totalWidth = count * stride - STRIP_GAP;
		const cameraAt = (position: number) => Math.max(0, Math.min(position * stride - (contentWidth - colWidth) / 2, Math.max(0, totalWidth - contentWidth)));
		const camera = cameraAt(this.#slidePosition(Date.now()));
		const railRows = count > 2 ? 2 : 0;
		const capacity = Math.max(1, this.#height - railRows - 2);
		const rendered = new Map<number, string[]>();
		const retained = new Set<Component>();
		let height = 0;
		for (let index = 0; index < count; index++) {
			const x = index * stride - camera;
			if ((x >= contentWidth || x + colWidth <= 0) && index !== this.#activeVariant) continue;
			let viewport: OutlineViewport;
			let builder: ChatTranscriptBuilder;
			let target: OutlineTarget | undefined;
			let label: string;
			if (index === 0) {
				viewport = this.#stripViewport;
				builder = this.#builder;
				target = this.#activeVariant === 0 ? anchor : undefined;
				label = "current";
			} else {
				const column = columns[index - 1]!;
				const materialized = this.#materialize(column);
				if (column.loading || column.error) {
					const rows = [...this.#columnHeader(index, count, column.label, colWidth), fit(column.error ?? "Indexing branch sources…", colWidth)];
					rendered.set(index, rows);
					height = Math.max(height, rows.length);
					continue;
				}
				builder = materialized.builder;
				viewport = column.viewport;
				target = this.#activeVariant === index ? materialized.targets[this.#siblingSelected] : undefined;
				label = column.label;
			}
			viewport.configure(builder.container.children, target, colWidth, capacity, index === 0 ? anchor.start : 0);
			if (this.#scrollToSelection && index === this.#activeVariant) viewport.follow();
			const rows = [...this.#columnHeader(index, count, label, colWidth), ...viewport.render()];
			height = Math.max(height, rows.length);
			rendered.set(index, rows);
			const used = viewport.finish();
			if (index === 0) for (const child of used) retained.add(child);
			else builder.releaseOutside(used);
		}
		for (let index = 0; index < columns.length; index++) {
			if (!rendered.has(index + 1)) columns[index]!.builder?.releaseOutside(NO_COMPONENTS);
		}
		this.#prefixViewport.configure(this.#builder.container.children, undefined, contentWidth, Math.max(0, this.#height - railRows - height), 0, anchor.start);
		this.#prefixViewport.end();
		const lines = this.#prefixViewport.render();
		for (const child of this.#prefixViewport.finish()) retained.add(child);
		this.#builder.releaseOutside(retained);
		if (count > 2) {
			const settled = cameraAt(this.#activeVariant);
			lines.push(positionRail(count, this.#activeVariant, settled > 0.5, settled + contentWidth < totalWidth - 0.5, contentWidth), "");
		}
		for (let row = 0; row < height; row++) {
			let line = "";
			let filled = 0;
			for (const [index, rows] of rendered) {
				const x = index * stride - camera;
				const start = Math.max(0, x);
				const end = Math.min(contentWidth, x + colWidth);
				if (end <= start) continue;
				const slice = sliceByColumn(fit(rows[row] ?? "", colWidth), start - x, end - start, true);
				line += padding(Math.max(0, start - filled)) + fit(slice, end - start);
				filled = end;
			}
			lines.push(line);
		}
		return lines;
	}

	#columnHeader(index: number, count: number, label: string, columnWidth: number): string[] {
		const caption = truncateToWidth(`${theme.icon.branch} ${index + 1}/${count} ${theme.sep.dot} ${label}`, columnWidth - 2);
		return [` ${theme.fg(index === this.#activeVariant ? "accent" : "dim", caption)}`, ""];
	}
}
