import {
	type Component,
	fuzzyMatch,
	Input,
	matchesKey,
	replaceTabs,
	routeSgrMouseInput,
	type SelectItem,
	SelectList,
	type SgrMouseEvent,
	truncateToWidth,
	wrapTextWithAnsi,
} from "@oh-my-pi/pi-tui";
import { getSelectListTheme, theme } from "../theme/theme";

export type CompactionMessageState = "auto" | "never" | "always";
export type CompactionMessageRole = "user" | "assistant" | "tool" | "custom";
export interface CompactionMessageRow {
	id: string;
	role: CompactionMessageRole;
	manual: CompactionMessageState;
	/** A thunk keeps authored bodies cold during metadata-only filtering. */
	preview: string | ((width?: number) => string);
	firstIndex?: number;
	recentIndex?: number;
	hardRecent?: boolean;
	alwaysAdmitted?: boolean;
	status?: "unclassified" | "queued" | "running" | "succeeded" | "failed" | "canceled";
	statusReason?: string;
	limitation?: string;
}
export interface CompactionMessageSource {
	count(role: "user" | "all"): number;
	get(index: number, role: "user" | "all"): CompactionMessageRow;
	indexOf(id: string, role: "user" | "all"): number;
}
export interface CompactionMessageSummary {
	scope: string;
	/** Owner-computed incremental counters; never a full-history aggregation on paint. */
	lines: readonly string[];
	nonAutoCount: number;
	classifier: { available: boolean; model: string; reason?: string };
	job?: {
		state: string;
		queued: number;
		running: number;
		completed: number;
		failed: number;
		saved: number;
		pending?: number;
	};
}
type ActionResult = void | string | Promise<void | string>;
export interface CompactionMessageReset {
	scope: string;
	count: number;
	/** Captured source boundary and override revisions are revalidated by the owner. */
	apply(): ActionResult;
}
export interface CompactionMessageManagerOptions {
	source: CompactionMessageSource;
	summary(): CompactionMessageSummary;
	inspect(id: string): readonly string[];
	inspectSettings?(id: string): readonly { label: string; path: string }[];
	setState(id: string, state: CompactionMessageState): ActionResult;
	prepareResetAll(): CompactionMessageReset | Promise<CompactionMessageReset>;
	classify(id: string): ActionResult;
	backfill(workers: number): ActionResult;
	cancelJob(): ActionResult;
	resumeJob(workers: number): ActionResult;
	settings(path?: string): ActionResult;
	usage(): ActionResult;
	details(): ActionResult;
	close(): void;
	requestRender(): void;
	height?(): number;
	/** Undefined IDs mean membership/scope changed; specific IDs only patch filters. */
	subscribe?(refresh: (ids?: readonly string[]) => void): () => void;
}

const STATES: readonly CompactionMessageState[] = ["never", "auto", "always"];
const ROLES: readonly CompactionMessageRole[] = ["user", "assistant", "tool", "custom"];
const GLYPHS: Record<CompactionMessageState, string> = { never: "[ ]", auto: "[-]", always: "[*]" };
const LEGEND = [
	"Manual: [ ] Never · [-] Auto (no override) · [*] Always",
	"A: selected for compaction via filter Keep/manual Always; [-] can have A.",
	"H: recent protection · F#/R#: first/recent ranks · i Inspect: why",
	"Classifier: + saved · ! failed · x canceled · spinner: queued/running",
];
const HELP = [
	"Context: active branch, after the current clear boundary. Oldest first, newest at bottom. Filters hide sources; they never change policy or reset/backfill scope.",
	"The checkbox is only your saved manual override, not the effective keep decision. Never [ ]: n; rejects extra retention, not normal recent history or summarization. Auto [-]: - / Backspace / r; no manual override, so filters may still select it without changing the checkbox. Always [*]: y / *; requests retention subject to Rule / Manual Keep Limit. Enter / Space cycle.",
	"A means this message is actually selected by Rule / Manual Keep Limit, requested by either automatic filter Keep or manual Always. A does not change [-] into [*]. Inspect shows which rule or manual choice requested it. A is a current selection preview, not proof that compaction has run.",
	"H is current most-recent protection, even for saved Never. F#/R# are ranks in the actual First/Recent selections: F1 is the oldest selected message; R1 is the newest. They are not visible row numbers. Classifier + means saved categories, ! failed, x canceled; a spinner means queued/running, not a change to the manual override.",
	"Heuristic → Regex → saved category rules → Final Regex → Manual. Auto makes no decision at that stage; later non-Auto rules take precedence. Within a stage Keep wins over Never, then Auto.",
	"Manual Always and current H retain intact content and bypass long-message pruning, including Exclude. Other automatic selections use configured pruning; text trimming retains original images.",
	"Rule / Manual Keep Limit selects filter Keep and manual Always messages using a separate allowance with the First/Recent limit value. Complete tool exchanges stay together. Denial does not remove independent First/Recent/H or normal recent retention; overlaps are kept once.",
	"Current policy / next-compaction preview are not installed representation. Use Inspect for source facts; Usage for physical totals; Details for emitted group inventory. Estimates and unavailable attribution remain labeled.",
	"c reclassifies the selected real user, retaining valid prior tags until success. C backfills missing/current-unusable tags across the whole current scope, including hidden rows. Many requests/tokens may cost money; choose workers before launch.",
	"Classifier job offers Cancel backfill and explicit missing-only Resume. Closing this view does not cancel session work. Only queued/running rows spin; failures remain visible and retryable.",
	"Actions exposes every command. / Search; f independent role/stored-state toggles; i Inspect; l Source settings; p Policy / preview; r Reset row; R confirmed Reset all (hidden rows included, tags/settings preserved); s Settings; u Usage; d Details; j Classifier job; ? Help; Esc Back/Close. Arrows, PageUp/Down, Home/End navigate. Mouse click selects a source and opens its Actions; wheel scrolls; menu items and footer are clickable.",
];

function singleLine(text: string): string {
	return replaceTabs(Bun.stripANSI(text)).replace(/[\x00-\x1f\x7f-\x9f\u2028\u2029]/g, " ");
}

function lowerBound(values: readonly number[], value: number): number {
	let low = 0;
	let high = values.length;
	while (low < high) {
		const middle = (low + high) >>> 1;
		if (values[middle]! < value) low = middle + 1;
		else high = middle;
	}
	return low;
}

/** Fullscreen native shell. The source owns rows; this view stores only filtered indices. */
export class CompactionMessageManagerComponent implements Component {
	#roles = new Set<CompactionMessageRole>(["user"]);
	#states = new Set<CompactionMessageState>(STATES);
	#query = "";
	#indices: number[] | undefined;
	#directory: "user" | "all" = "user";
	#selectedId: string | undefined;
	#anchor = 0;
	#initialSelectionPending = true;
	#selected = -1;
	#top = 0;
	#bodyStart = 0;
	#bodyRows = 1;
	#footerRow = 0;
	#footerHits: { start: number; end: number; action: string }[] = [];
	#menu: SelectList | undefined;
	#menuTitle = "";
	#menuIntro: readonly string[] = [];
	#menuStart = 0;
	#input: Input | undefined;
	#inputKind: "search" | "workers" | undefined;
	#searchBefore = "";
	#searchAnchor: string | undefined;
	#document: { title: string; lines: readonly string[]; sourceId?: string; back?: () => void } | undefined;
	#documentTop = 0;
	#documentLength = 0;
	#notice = "";
	#scanGeneration = 0;
	#scanAt = 0;
	#scanTotal = 0;
	#scanning = false;
	#scanTimer: NodeJS.Timeout | undefined;
	#spinner: NodeJS.Timeout | undefined;
	#frame = 0;
	#unsubscribe: (() => void) | undefined;
	#disposed = false;
	#pendingStates = new Map<string, CompactionMessageState>();
	#pendingClassify = new Set<string>();
	#dialogGeneration = 0;

	constructor(private readonly options: CompactionMessageManagerOptions) {
		const count = options.source.count("user");
		if (count) this.#select(count - 1);
		this.#unsubscribe = options.subscribe?.(ids => this.refresh(ids));
	}

	invalidate(): void {
		this.#menu?.invalidate();
		this.#input?.invalidate();
	}

	/** No session job is canceled when the UI releases its own resources. */
	dispose(): void {
		if (this.#disposed) return;
		this.#disposed = true;
		this.#scanGeneration++;
		this.#dialogGeneration++;
		clearTimeout(this.#scanTimer);
		clearInterval(this.#spinner);
		this.#unsubscribe?.();
		this.#unsubscribe = undefined;
	}

	refresh(ids?: readonly string[]): void {
		if (this.#disposed) return;
		if (this.#document?.sourceId && (!ids || ids.includes(this.#document.sourceId))) {
			this.#document.lines = this.options.inspect(this.#document.sourceId);
		}
		if (this.#document?.title === "Policy / preview") this.#document.lines = this.options.summary().lines;
		if (this.#menu && this.#menuTitle === "Classifier job") this.#menuIntro = this.#jobLines();
		if (this.#document?.title === "Classifier job") this.#document.lines = this.#jobLines();
		if (!ids) this.#filter();
		else if (this.#indices) {
			for (const id of ids) {
				const index = this.options.source.indexOf(id, this.#directory);
				if (index < 0 || (this.#scanning && index >= this.#scanAt)) continue;
				const position = lowerBound(this.#indices, index);
				const present = this.#indices[position] === index;
				const matches = this.#matches(this.options.source.get(index, this.#directory));
				if (present && !matches) this.#indices.splice(position, 1);
				else if (!present && matches) this.#indices.splice(position, 0, index);
			}
			this.#restoreAnchor();
		} else this.#restoreAnchor();
		this.#requestRender();
	}

	#requestRender(): void {
		if (!this.#disposed) this.options.requestRender();
	}
	#count(): number {
		return this.#indices?.length ?? this.options.source.count(this.#directory);
	}
	#row(position: number): CompactionMessageRow | undefined {
		if (position < 0 || position >= this.#count()) return undefined;
		return this.options.source.get(this.#indices?.[position] ?? position, this.#directory);
	}
	#state(row: CompactionMessageRow): CompactionMessageState {
		return this.#pendingStates.get(row.id) ?? row.manual;
	}
	#select(position: number): void {
		this.#initialSelectionPending = false;
		this.#selected = Math.min(Math.max(0, position), this.#count() - 1);
		const row = this.#row(this.#selected);
		this.#selectedId = row?.id;
		if (row) this.#anchor = this.options.source.indexOf(row.id, "all");
	}
	#restoreAnchor(): void {
		if (this.#scanning || !this.#count()) {
			this.#selected = -1;
			return;
		}
		const source = this.options.source;
		if (this.#initialSelectionPending) {
			this.#select(this.#count() - 1);
			return;
		}
		const exact = this.#selectedId ? source.indexOf(this.#selectedId, this.#directory) : -1;
		if (exact >= 0) {
			const position = this.#indices ? lowerBound(this.#indices, exact) : exact;
			if (!this.#indices || this.#indices[position] === exact) {
				this.#select(position);
				return;
			}
		}
		// Compare authoritative chronological coordinates, never yesterday's visible index.
		let low = 0;
		let high = this.#count();
		while (low < high) {
			const middle = (low + high) >>> 1;
			const row = this.#row(middle)!;
			if (source.indexOf(row.id, "all") < this.#anchor) low = middle + 1;
			else high = middle;
		}
		this.#select(Math.min(low, this.#count() - 1));
	}
	#matches(row: CompactionMessageRow): boolean {
		return (
			this.#roles.has(row.role) &&
			this.#states.has(this.#state(row)) &&
			(!this.#query ||
				fuzzyMatch(this.#query, singleLine(typeof row.preview === "function" ? row.preview() : row.preview))
					.matches)
		);
	}
	#filter(): void {
		const generation = ++this.#scanGeneration;
		clearTimeout(this.#scanTimer);
		this.#scanning = false;
		this.#directory = this.#roles.size === 1 && this.#roles.has("user") ? "user" : "all";
		if (
			!this.#query &&
			this.#states.size === STATES.length &&
			(this.#directory === "user" || this.#roles.size === ROLES.length)
		) {
			this.#indices = undefined;
			this.#restoreAnchor();
			this.#requestRender();
			return;
		}
		this.#indices = [];
		this.#selected = -1;
		this.#scanAt = 0;
		this.#scanTotal = this.options.source.count(this.#directory);
		this.#scanning = true;
		const scan = () => {
			if (this.#disposed || generation !== this.#scanGeneration) return;
			const deadline = performance.now() + 6;
			let count = 0;
			while (this.#scanAt < this.#scanTotal && count++ < 512 && performance.now() < deadline) {
				const index = this.#scanAt++;
				if (this.#matches(this.options.source.get(index, this.#directory))) this.#indices!.push(index);
			}
			this.#scanning = this.#scanAt < this.#scanTotal;
			if (this.#scanning) this.#scanTimer = setTimeout(scan, 0);
			else this.#restoreAnchor();
			this.#requestRender();
		};
		// Defer even the first chunk: opening a filter never owns the input turn.
		this.#scanTimer = setTimeout(scan, 0);
		this.#requestRender();
	}

	#back(): void {
		this.#dialogGeneration++;
		this.#menu = undefined;
		this.#input = undefined;
		this.#inputKind = undefined;
		this.#document = undefined;
		this.#notice = "";
		this.#requestRender();
	}
	#showMenu(
		title: string,
		items: readonly SelectItem[],
		select: (value: string) => void,
		intro: readonly string[] = [],
		index = 0,
	): void {
		this.#back();
		this.#menuTitle = title;
		this.#menuIntro = intro;
		const menuItems = intro.length ? [...items, { value: "__scope", label: "Full scope / explanation…" }] : items;
		const list = new SelectList(menuItems, this.#bodyRows, getSelectListTheme(), { overflowSearch: false });
		list.setSelectedIndex(index);
		list.onSelect = item => {
			if (item.value === "__scope") {
				this.#showDocument(title, intro);
				this.#document!.back = () => this.#showMenu(title, items, select, intro, index);
			} else select(item.value);
			this.#requestRender();
		};
		list.onCancel = () => this.#back();
		list.onSelectionChange = () => this.#requestRender();
		this.#menu = list;
		this.#requestRender();
	}
	#actions(): void {
		const sourceId = this.#document?.sourceId ?? this.#selectedId;
		const items = [
			["never", "Never [ ] (n)"],
			["auto", "Auto [-] (- / Backspace)"],
			["always", "Always [*] (y / *)"],
			["cycle", "Cycle state (Space / Enter)"],
			["inspect", "Inspect (i)"],
			["source-settings", "Source settings (l)"],
			["search", "Search (/)"],
			["filters", "Filters (f)"],
			["reset", "Reset row (r)"],
			["reset-all", "Reset all (R)"],
			["classify", "Classify selected / retry (c)"],
			["backfill", "Backfill missing (C)"],
			["job", "Classifier job (j)"],
			["settings", "Settings (s)"],
			["usage", "Usage (u)"],
			["preview", "Policy / preview (p)"],
			["details", "Details (d)"],
			["help", "Help (?)"],
			["close", "Close (Esc)"],
		];
		this.#showMenu(
			"Actions",
			items.map(([value, label]) => ({ value: value!, label: label! })),
			value => this.#action(value, sourceId),
		);
	}
	#filters(selected = 0): void {
		this.#initialSelectionPending = false;
		const items: SelectItem[] = [];
		for (const role of ROLES)
			items.push({ value: `role:${role}`, label: `${this.#roles.has(role) ? "[x]" : "[ ]"} Role: ${role}` });
		for (const state of STATES)
			items.push({ value: `state:${state}`, label: `${this.#states.has(state) ? "[x]" : "[ ]"} Stored: ${state}` });
		items.push({ value: "default", label: "Restore default filters" }, { value: "done", label: "Done" });
		this.#showMenu(
			"Filters (independent toggles)",
			items,
			value => {
				if (value === "done") {
					this.#back();
					return;
				}
				if (value === "default") {
					this.#roles = new Set(["user"]);
					this.#states = new Set(STATES);
				} else if (value.startsWith("role:")) {
					const role = value.slice(5) as CompactionMessageRole;
					if (this.#roles.has(role)) this.#roles.delete(role);
					else this.#roles.add(role);
				} else {
					const state = value.slice(6) as CompactionMessageState;
					if (this.#states.has(state)) this.#states.delete(state);
					else this.#states.add(state);
				}
				this.#filter();
				this.#filters(items.findIndex(item => item.value === value));
			},
			[],
			selected,
		);
	}
	#search(): void {
		this.#initialSelectionPending = false;
		this.#back();
		this.#searchBefore = this.#query;
		this.#searchAnchor = this.#selectedId;
		const input = new Input();
		input.prompt = "/ ";
		input.focused = true;
		input.setValue(this.#query);
		input.onSubmit = () => this.#back();
		input.onEscape = () => {
			this.#query = this.#searchBefore;
			this.#selectedId = this.#searchAnchor;
			this.#filter();
			this.#back();
		};
		this.#input = input;
		this.#inputKind = "search";
		this.#requestRender();
	}
	#showDocument(title: string, lines: readonly string[]): void {
		this.#back();
		this.#document = { title, lines };
		this.#documentTop = 0;
		this.#requestRender();
	}
	async #run(action: () => ActionResult, ids?: readonly string[]): Promise<void> {
		try {
			const result = await action();
			if (this.#disposed) return;
			this.#notice = typeof result === "string" ? result : "";
			this.refresh(ids ?? []);
		} catch (error) {
			if (!this.#disposed) {
				this.#notice = `Error: ${error instanceof Error ? error.message : String(error)}`;
				this.#requestRender();
			}
		}
	}
	async #setState(state: CompactionMessageState): Promise<void> {
		const row = this.#row(this.#selected);
		if (!row) {
			this.#notice = "No selected source. Filters do not change action scope.";
			return;
		}
		if (row.limitation) {
			this.#notice = row.limitation;
			return;
		}
		if (this.#pendingStates.has(row.id)) {
			this.#notice = "Saving this source; wait before editing it again.";
			return;
		}
		if (row.manual === state) return;
		this.#pendingStates.set(row.id, state);
		this.refresh([row.id]);
		try {
			const result = await this.options.setState(row.id, state);
			if (!this.#disposed) this.#notice = typeof result === "string" ? result : "Saved";
		} catch (error) {
			if (!this.#disposed) this.#notice = `Not saved: ${error instanceof Error ? error.message : String(error)}`;
		} finally {
			this.#pendingStates.delete(row.id);
			this.refresh([row.id]);
		}
	}
	async #resetAll(): Promise<void> {
		const generation = this.#dialogGeneration;
		this.#notice = "Capturing reset scope…";
		try {
			const reset = await this.options.prepareResetAll();
			if (this.#disposed || generation !== this.#dialogGeneration) return;
			this.#showMenu(
				"Reset all overrides?",
				[
					{ value: "no", label: "No — keep overrides" },
					{ value: "yes", label: `Yes — reset ${reset.count} sources/groups` },
				],
				value => {
					this.#back();
					if (value === "yes") {
						this.#notice = "Saving captured reset…";
						void this.#run(async () => {
							const result = await reset.apply();
							this.refresh();
							return result ?? "Saved";
						});
					}
				},
				[
					reset.scope,
					`${reset.count} non-Auto sources/groups. Includes hidden rows.`,
					"Preserves tags and settings; newer edits are skipped.",
				],
			);
		} catch (error) {
			if (!this.#disposed && generation === this.#dialogGeneration)
				this.#notice = `Cannot capture reset: ${error instanceof Error ? error.message : String(error)}`;
		}
		this.#requestRender();
	}
	#classifierAvailable(): boolean {
		const classifier = this.options.summary().classifier;
		if (classifier.available) return true;
		this.#showMenu(
			"Classifier unavailable",
			[
				{ value: "configure", label: "Configure Model" },
				{ value: "back", label: "Back" },
			],
			value => {
				this.#back();
				if (value === "configure")
					void this.#run(() => this.options.settings("compaction.keepUserMessagesLlmModel"));
			},
			[classifier.reason ?? `Model unavailable: ${classifier.model}`],
		);
		return false;
	}
	#classify(): void {
		const row = this.#row(this.#selected);
		if (!row) {
			this.#notice = "No selected source to classify.";
			return;
		}
		if (row.role !== "user") {
			this.#notice = "Only real user sources can be classified; non-users have manual state.";
			return;
		}
		if (!this.#classifierAvailable()) return;
		if (this.#pendingClassify.has(row.id) || row.status === "queued" || row.status === "running") {
			this.#notice = "This source is already queued/running.";
			return;
		}
		this.#pendingClassify.add(row.id);
		void this.#run(async () => {
			try {
				return await this.options.classify(row.id);
			} finally {
				this.#pendingClassify.delete(row.id);
				this.refresh([row.id]);
			}
		}, [row.id]);
	}
	#backfill(resume = false): void {
		if (!this.#classifierAvailable()) return;
		const summary = this.options.summary();
		this.#showMenu(
			resume ? "Resume missing classification?" : "Backfill missing classification?",
			[
				{ value: "cancel", label: "Cancel — no requests" },
				{ value: "workers", label: "Choose workers and launch…" },
			],
			value => {
				this.#back();
				if (value !== "workers") return;
				const input = new Input();
				input.prompt = "Workers: ";
				input.focused = true;
				input.setValue("4");
				input.onEscape = () => this.#back();
				input.onSubmit = value => {
					const workers = Number(value);
					if (!/^\d+$/.test(value.trim()) || !Number.isSafeInteger(workers) || workers < 1) {
						this.#notice = "Workers must be a positive safe integer.";
						return;
					}
					this.#back();
					void this.#run(() => (resume ? this.options.resumeJob(workers) : this.options.backfill(workers)));
				};
				this.#input = input;
				this.#inputKind = "workers";
			},
			[
				summary.scope,
				`Model: ${summary.classifier.model}`,
				"Many model requests/tokens may incur substantial cost.",
				"Missing/current-unusable facts only; includes hidden rows. Continues after closing this view.",
			],
		);
	}
	#jobLines(): readonly string[] {
		const summary = this.options.summary();
		const job = summary.job;
		return job
			? [
					summary.scope,
					`State: ${job.state}`,
					`Queued ${job.queued} / running ${job.running}`,
					`Completed ${job.completed} / saved ${job.saved}`,
					`Failed ${job.failed} / pending ${job.pending ?? job.queued + job.running}`,
					"Cancel preserves tags and unrelated live work.",
				]
			: ["No backfill job in this session. Resume starts missing-only work."];
	}
	#job(): void {
		this.#showMenu(
			"Classifier job",
			[
				{ value: "cancel", label: "Cancel backfill only" },
				{ value: "resume", label: "Resume missing/current-unusable…" },
				{ value: "refresh", label: "Refresh status" },
				{ value: "back", label: "Back" },
			],
			value => {
				if (value === "resume") {
					this.#backfill(true);
					return;
				}
				if (value === "refresh") {
					this.#job();
					return;
				}
				this.#back();
				if (value === "cancel") {
					const job = this.options.summary().job;
					if (!job || (job.queued === 0 && job.running === 0 && job.state !== "running"))
						this.#notice = "No active backfill to cancel.";
					else void this.#run(() => this.options.cancelJob());
				}
			},
			this.#jobLines(),
		);
	}
	#sourceSettings(id: string | undefined): void {
		if (!id || this.options.source.indexOf(id, "all") < 0) {
			this.#notice = "No active source selected for source settings.";
			return;
		}
		const links = this.options.inspectSettings?.(id);
		if (!links?.length) {
			this.#notice = "No source-specific settings links available. Use s for Context settings.";
			return;
		}
		this.#showMenu(
			"Source settings",
			links.map(link => ({ value: link.path, label: singleLine(link.label) })),
			path => {
				this.#back();
				void this.#run(() => this.options.settings(path));
			},
			[`Source: ${id}`],
		);
	}
	#action(action: string, sourceId = this.#document?.sourceId ?? this.#selectedId): void {
		this.#back();
		const row = this.#row(this.#selected);
		switch (action) {
			case "never":
			case "auto":
			case "always":
				void this.#setState(action);
				break;
			case "reset":
				void this.#setState("auto");
				break;
			case "cycle":
				if (row) void this.#setState(STATES[(STATES.indexOf(this.#state(row)) + 1) % STATES.length]!);
				else this.#notice = "No selected source.";
				break;
			case "inspect":
				if (row) {
					this.#showDocument("Inspect source", this.options.inspect(row.id));
					this.#document!.sourceId = row.id;
				} else this.#notice = "No selected source to inspect.";
				break;
			case "search":
				this.#search();
				break;
			case "filters":
				this.#filters();
				break;
			case "reset-all":
				void this.#resetAll();
				break;
			case "classify":
				this.#classify();
				break;
			case "backfill":
				this.#backfill();
				break;
			case "job":
				this.#job();
				break;
			case "source-settings":
				this.#sourceSettings(sourceId);
				break;
			case "settings":
				void this.#run(() => this.options.settings());
				break;
			case "usage":
				void this.#run(() => this.options.usage());
				break;
			case "details":
				void this.#run(() => this.options.details());
				break;
			case "preview":
				this.#showDocument("Policy / preview", this.options.summary().lines);
				break;
			case "help":
				this.#showDocument("Context help", HELP);
				break;
			case "close":
				this.dispose();
				this.options.close();
				break;
		}
		this.#requestRender();
	}

	handleInput(data: string): void {
		if (this.#disposed) return;
		if (routeSgrMouseInput(data, event => this.#mouse(event))) return;
		if (this.#input) {
			const input = this.#input;
			const kind = this.#inputKind;
			input.handleInput(data);
			if (this.#input === input && kind === "search" && this.#query !== input.getValue()) {
				this.#query = input.getValue();
				if (!this.#query) this.#selectedId = this.#searchAnchor;
				this.#filter();
			}
			this.#requestRender();
			return;
		}
		if (this.#menu) {
			this.#menu.handleInput(data);
			this.#requestRender();
			return;
		}
		if (matchesKey(data, "escape")) {
			if (this.#document?.back) this.#document.back();
			else if (this.#document) this.#back();
			else this.#action("close");
			return;
		}
		const delta = matchesKey(data, "up")
			? -1
			: matchesKey(data, "down")
				? 1
				: matchesKey(data, "pageUp")
					? -this.#bodyRows
					: matchesKey(data, "pageDown")
						? this.#bodyRows
						: 0;
		if (this.#document) {
			if (delta)
				this.#documentTop = Math.max(0, Math.min(this.#documentTop + delta, this.#documentLength - this.#bodyRows));
			if (matchesKey(data, "home")) this.#documentTop = 0;
			if (matchesKey(data, "end")) this.#documentTop = Math.max(0, this.#documentLength - this.#bodyRows);
		} else {
			if (delta) this.#select(this.#selected + delta);
			if (matchesKey(data, "home")) this.#select(0);
			if (matchesKey(data, "end")) this.#select(this.#count() - 1);
		}
		if (data === "a") this.#actions();
		else if (matchesKey(data, "backspace") || data === "-") this.#action("auto");
		else if (matchesKey(data, "enter") || data === " ") this.#action("cycle");
		else {
			const actions: Record<string, string> = {
				n: "never",
				y: "always",
				"*": "always",
				"/": "search",
				f: "filters",
				i: "inspect",
				l: "source-settings",
				r: "reset",
				R: "reset-all",
				c: "classify",
				C: "backfill",
				j: "job",
				s: "settings",
				u: "usage",
				d: "details",
				p: "preview",
				"?": "help",
			};
			if (actions[data]) this.#action(actions[data]);
		}
		this.#requestRender();
	}
	#mouse(event: SgrMouseEvent): boolean {
		if (event.leftClick && event.row === this.#footerRow) {
			const hit = this.#footerHits.find(item => event.col >= item.start && event.col < item.end);
			if (hit?.action === "actions") this.#actions();
			else if (hit?.action === "back") {
				if (this.#input) this.#input.onEscape?.();
				else if (this.#document?.back) this.#document.back();
				else if (this.#menu || this.#document) this.#back();
				else this.#action("close");
			} else if (hit) this.#action(hit.action);
		} else if (this.#menu) this.#menu.routeMouse(event, event.row - this.#menuStart, event.col);
		else if (event.wheel !== null) {
			if (this.#document)
				this.#documentTop = Math.max(
					0,
					Math.min(this.#documentTop + event.wheel * 3, this.#documentLength - this.#bodyRows),
				);
			else this.#select(this.#selected + event.wheel * 3);
		} else if (
			!this.#input &&
			!this.#document &&
			event.leftClick &&
			event.row >= this.#bodyStart &&
			event.row < this.#bodyStart + this.#bodyRows
		) {
			const index = this.#top + event.row - this.#bodyStart;
			if (index < this.#count()) {
				this.#select(index);
				this.#actions();
			}
		}
		this.#requestRender();
		return true;
	}

	#renderRow(row: CompactionMessageRow, selected: boolean, width: number): string {
		const state = this.#state(row);
		const pending = row.status === "queued" || row.status === "running" || this.#pendingClassify.has(row.id);
		const frames = theme.spinnerFrames;
		const status = pending
			? frames[this.#frame % frames.length]
			: row.status === "failed"
				? "!"
				: row.status === "succeeded"
					? "+"
					: row.status === "canceled"
						? "x"
						: " ";
		let prefix = `${selected ? ">" : " "}${GLYPHS[state]}${row.hardRecent ? "H" : " "}${row.alwaysAdmitted ? "A" : " "}${status} `;
		if (width >= 48)
			prefix +=
				`${row.firstIndex === undefined ? "" : `F${row.firstIndex}`}`.padEnd(5) +
				`${row.recentIndex === undefined ? "" : `R${row.recentIndex}`}`.padEnd(5);
		if (width >= 72) prefix += `${row.role.padEnd(9)} `;
		const text = typeof row.preview === "function" ? row.preview(width) : row.preview;
		const line = truncateToWidth(prefix + singleLine(text), Math.max(0, width));
		return selected ? theme.bg("selectedBg", theme.fg("accent", line)) : line;
	}
	#syncSpinner(active: boolean): void {
		if (active && !this.#spinner && !this.#disposed) {
			this.#spinner = setInterval(() => {
				this.#frame++;
				this.#requestRender();
			}, 80);
			this.#spinner.unref();
		} else if (!active && this.#spinner) {
			clearInterval(this.#spinner);
			this.#spinner = undefined;
		}
	}
	render(width: number): readonly string[] {
		if (this.#disposed) return [];
		const height = Math.max(0, Math.floor(this.options.height?.() ?? process.stdout.rows ?? 24));
		width = Math.max(0, width);
		if (!height) return [];
		const summary = this.options.summary();
		const out: string[] = [];
		const fit = (line: string) => truncateToWidth(line, width);
		const title = this.#menu
			? this.#menuTitle
			: (this.#document?.title ?? (this.#inputKind === "workers" ? "Classifier workers" : "Context"));
		const showLegend = !this.#menu && !this.#document && this.#inputKind !== "workers";
		if (height >= 3) out.push(fit(theme.bold(singleLine(`${title} · ${summary.scope}`))));
		if (height >= 7 && (!showLegend || height >= 12))
			out.push(
				fit(
					singleLine(
						this.#notice || summary.lines[0] || "Current policy / next preview ≠ installed representation",
					),
				),
			);
		if (height >= 12 && !this.#menu && !this.#document)
			out.push(
				fit(
					singleLine(
						this.#scanning
							? `Filtering ${this.#scanAt}/${this.#scanTotal} — Esc closes`
							: `${this.#count()} visible / ${this.options.source.count("all")} sources · ${[...this.#roles].join(",") || "no roles"} · ${[...this.#states].join(",") || "no states"}${this.#query ? ` · /${this.#query}` : ""}`,
					),
				),
			);
		if (showLegend) {
			for (const line of LEGEND) {
				const wrapped = wrapTextWithAnsi(line, Math.max(1, width));
				if (out.length + wrapped.length > height - 2) break;
				for (const part of wrapped) out.push(fit(theme.fg("dim", part)));
			}
		}
		this.#bodyStart = out.length;
		this.#bodyRows = Math.max(0, height - out.length - 1);
		let active = !!summary.job && (summary.job.queued > 0 || summary.job.running > 0);
		if (this.#menu) {
			// Intro is wrapped and included in the budget, leaving at least one native menu item.
			const intro = this.#menuIntro.flatMap(line => wrapTextWithAnsi(singleLine(line), Math.max(1, width)));
			const introBudget = Math.min(intro.length, Math.max(0, this.#bodyRows - 3));
			out.push(...intro.slice(0, introBudget).map(fit));
			this.#menuStart = out.length;
			this.#menu.setMaxVisible(Math.max(1, height - out.length - 1));
			out.push(
				...this.#menu
					.render(width)
					.slice(0, Math.max(0, height - out.length - 1))
					.map(fit),
			);
		} else if (this.#document) {
			const lines = this.#document.lines.flatMap(line => wrapTextWithAnsi(singleLine(line), Math.max(1, width)));
			this.#documentLength = lines.length;
			this.#documentTop = Math.max(0, Math.min(this.#documentTop, lines.length - this.#bodyRows));
			out.push(...lines.slice(this.#documentTop, this.#documentTop + this.#bodyRows).map(fit));
		} else if (this.#inputKind === "workers") {
			if (this.#bodyRows) out.push(...this.#input!.render(width).slice(0, this.#bodyRows).map(fit));
		} else {
			if (this.#inputKind === "search" && this.#bodyRows) {
				out.push(fit(this.#input!.render(width)[0] ?? ""));
				this.#bodyStart++;
				this.#bodyRows--;
			}
			this.#top = Math.max(0, Math.min(this.#top, this.#count() - this.#bodyRows));
			if (this.#selected < this.#top) this.#top = Math.max(0, this.#selected);
			if (this.#selected >= this.#top + this.#bodyRows) this.#top = Math.max(0, this.#selected - this.#bodyRows + 1);
			if (!this.#count() && this.#bodyRows)
				out.push(
					fit(
						this.options.source.count("all")
							? this.#scanning
								? "Filtering sources…"
								: "No filter results. a Actions"
							: "No sources in this session epoch. a Actions",
					),
				);
			for (let i = this.#top; i < Math.min(this.#count(), this.#top + this.#bodyRows); i++) {
				const row = this.#row(i)!;
				active ||= row.status === "queued" || row.status === "running" || this.#pendingClassify.has(row.id);
				out.push(this.#renderRow(row, i === this.#selected, width));
			}
		}
		this.#syncSpinner(active);
		while (out.length < height - 1) out.push("");
		this.#footerRow = out.length;
		this.#footerHits = [];
		let footer = "";
		for (const [label, action] of [
			["a Actions", "actions"],
			["? Help", "help"],
			["Esc Back", "back"],
			["/ Search", "search"],
			["f Filters", "filters"],
			["i Inspect", "inspect"],
		]) {
			const start = footer.length;
			if (start) footer += " · ";
			this.#footerHits.push({ start: start ? start + 3 : 0, end: footer.length + label!.length, action: action! });
			footer += label;
		}
		out.push(fit(theme.fg("dim", footer)));
		return out;
	}
}
