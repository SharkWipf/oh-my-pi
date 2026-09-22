import type { ThinkingLevel } from "@oh-my-pi/pi-agent-core";
import type { Effort } from "@oh-my-pi/pi-ai";
import {
	type Component,
	Container,
	extractPrintableText,
	fuzzyRank,
	getKeybindings,
	getSettingItemFilterText,
	type ImageBudget,
	Input,
	matchesKey,
	routeSelectListMouse,
	routeSgrMouseInput,
	type SelectItem,
	SelectList,
	type SettingItem,
	SettingsList,
	type SgrMouseEvent,
	type Tab,
	TabBar,
	Text,
	truncateToWidth,
	visibleWidth,
} from "../index";
import type { ShapeTarget } from "@oh-my-pi/snapcompact";
import type {
	ContextLineMode,
	StatusLinePreset,
	StatusLineSegmentId,
	StatusLineSeparatorStyle,
} from "../status-line/schema";
import {
	SETTING_TABS,
	TAB_METADATA,
	type SettingTab,
	type SettingsHost,
	type SettingsDisplayEntry,
} from "./settings-defs";
import { getCurrentThemeName, getSelectListTheme, getSettingsListTheme, theme } from "../theme/theme";
import { AUTO_THINKING, type ConfiguredThinkingLevel } from "../thinking";
import { getTabBarTheme } from "../chrome/shared";
import { type ComposerPreviewStatusSource, ComposerShapePreview } from "./composer-shape-preview";
import { getComposerShapeOptions } from "./composer-shape-registry";
import { bottomBorder, divider, row, topBorder } from "../chrome/overlay-box";
import { PluginSettingsComponent, type PluginSettingsHost } from "./plugin-settings";
import { getSettingDef, getSettingsForTab, type SettingDef } from "./settings-defs";
import { SnapcompactShapePreview } from "./snapcompact-shape-preview";
import { getPreset } from "../status-line/presets";
import { FormField, SelectFormField, TextFormField } from "../components/form";
import { formTheme } from "../chrome/form-theme";
import { SettingsFormField } from "../components/settings-list";
import type { PreservationAction, PreservationLimit, PreservationRegexRule } from "./settings-defs";
import {
	ModelBrowser,
	buildBrowserItems,
	resolveRoleAssignments,
	sortModelItems,
	type ModelBrowserSource,
	type ModelBrowserRegistry,
} from "./model-browser";

/**
 * Free-text string setting field backed by the shared text form field.
 * Current values prefill, including secrets retained behind Input masking;
 * submitting an empty string clears the setting and validation errors stay inline.
 */
function createSettingsTextField(
	label: string,
	description: string,
	currentValue: string,
	secret: boolean,
	onSubmit: (value: string) => void | Promise<void>,
	onCancel: () => void,
	requestRender?: () => void,
): TextFormField {
	return new TextFormField({
		theme: formTheme,
		label,
		description: description || undefined,
		secret,
		initialValue: currentValue || undefined,
		empty: "submit",
		hint: "  Enter to save · Esc to cancel · Clear field to unset",
		onSubmit,
		onCancel,
		requestRender,
	});
}

/**
 * Single-choice setting field backed by the shared select form field.
 * Preserves the current selection, live async previews, footer previews,
 * and select/cancel dispatch of the bespoke submenu it replaces.
 */
function createSettingsSelectField(
	title: string,
	description: string,
	options: ReadonlyArray<SelectItem>,
	currentValue: string,
	onSelect: (value: string) => void,
	onCancel: () => void,
	onSelectionChange?: (value: string) => void | Promise<void>,
	getPreview?: () => string,
	footer?: Component,
	requestRender?: () => void,
): SelectFormField {
	return new SelectFormField({
		theme: formTheme,
		label: title,
		description: description || undefined,
		items: options,
		currentValue,
		maxVisible: 10,
		selectTheme: getSelectListTheme(),
		getPreview,
		onSelectionChange,
		onSubmit: onSelect,
		onCancel,
		hint: "  Enter to select · Esc to go back",
		footer,
		requestRender,
	});
}

type PreservationSettingItem = SettingItem & {
	assignment?:
		| { kind: "boolean"; get(): boolean; set(value: boolean): void }
		| { kind: "action"; get(): PreservationAction; set(value: PreservationAction): void };
};

function actionKey(data: string): PreservationAction | undefined {
	if (data === "y" || data === "*") return "keep";
	if (data === "n") return "exclude";
	if (data === "-" || matchesKey(data, "backspace")) return "auto";
	return undefined;
}

function assignPreservationItem(item: SettingItem | undefined, data: string): boolean {
	const assignment = (item as PreservationSettingItem | undefined)?.assignment;
	if (!assignment || item?.heading) return false;
	if (assignment.kind === "boolean") {
		const current = assignment.get();
		const value = data === " " ? !current : data === "y" ? true : data === "n" ? false : undefined;
		if (value === undefined) return false;
		if (value !== current) assignment.set(value);
	} else {
		const current = assignment.get();
		const value =
			data === " " ? (current === "exclude" ? "auto" : current === "auto" ? "keep" : "exclude") : actionKey(data);
		if (value === undefined) return false;
		if (value !== current) assignment.set(value);
	}
	return true;
}

/** Native nested forms, with assignment only on explicitly typed preservation rows. */
class SettingsSubmenu extends SettingsFormField {
	readonly list: SettingsList;
	constructor(
		title: string,
		items: PreservationSettingItem[],
		onChange: (id: string, value: string) => void,
		onCancel: () => void,
		private readonly getHeight: () => number,
		private readonly onRender?: () => void,
		private readonly onHint?: (hint: string) => void,
	) {
		super({
			items,
			maxVisible: 3,
			settingsTheme: getSettingsListTheme(),
			fieldTheme: formTheme,
			label: title,
			onChange,
			onCancel,
			spaceBeforeControl: false,
			spaceAfterControl: false,
			listOptions: { layout: "flat", typeToSearch: false, hint: "" },
		});
		this.list = this.settingsList;
	}
	override render(width: number): readonly string[] {
		if (this.list.hasOpenSubmenu()) return this.list.render(width);
		const assignment = (this.list.getSelectedItem() as PreservationSettingItem | undefined)?.assignment;
		this.onHint?.(
			assignment?.kind === "boolean"
				? "Space toggle · y on · n off · Enter toggle · Esc back"
				: assignment?.kind === "action"
					? "Space cycle · y/* Keep · n Never · -/Backspace Auto · Enter choices · Esc back"
					: "Enter choose · Esc back",
		);
		this.onRender?.();
		this.list.setMaxVisible(Math.max(1, this.getHeight() - 5));
		return super.render(width);
	}
	override handleInput(data: string): void {
		try {
			if (!this.list.hasOpenSubmenu() && assignPreservationItem(this.list.getSelectedItem(), data)) {
				this.setError(undefined);
				return;
			}
			super.handleInput(data);
		} catch (error) {
			this.setError(error instanceof Error ? error.message : String(error));
		}
	}
	override routeMouse(event: SgrMouseEvent, line: number, col: number): void {
		if (this.list.hasOpenSubmenu()) {
			this.list.routeSubmenuMouse(event, line, col);
			return;
		}
		const controlLine = this.controlLineAt(line);
		if (controlLine === undefined) return;
		if (event.wheel !== null) {
			this.list.handleWheel(event.wheel);
			return;
		}
		const id = this.list.hitTest(controlLine, col);
		if (event.motion) {
			this.list.setHoverItem(id ?? null);
			return;
		}
		if (event.leftClick && id !== undefined) {
			const activate = this.list.getSelectedItem()?.id === id;
			this.list.selectItem(id);
			if (activate) this.list.handleInput("\n");
		}
	}
}

/** Keep native controls above explanatory prose when the viewport is short. */
class PreservationSelectField extends SelectFormField {
	constructor(
		options: ConstructorParameters<typeof SelectFormField>[0],
		private readonly onHint: () => void,
		private readonly getHeight: () => number,
	) {
		super({
			...options,
			description: undefined,
			spaceBeforeControl: false,
			spaceAfterControl: false,
			summary: options.description ? [new Text(theme.fg("muted", options.description), 0, 0)] : undefined,
		});
	}
	override render(width: number): readonly string[] {
		this.onHint();
		this.selectList.setMaxVisible(Math.max(3, this.getHeight() - 4));
		return super.render(width);
	}
}

/** Only the explicit three-way policy chooser accepts assignment/Space confirmation. */
class PreservationActionField extends PreservationSelectField {
	override handleInput(data: string): void {
		const value = actionKey(data);
		if (value !== undefined) {
			this.selectList.setSelectedIndex(value === "auto" ? 0 : value === "keep" ? 1 : 2);
			super.handleInput("\n");
		} else super.handleInput(data === " " ? "\n" : data);
	}
}

class PreservationTextField extends TextFormField {
	constructor(
		options: ConstructorParameters<typeof TextFormField>[0],
		private readonly onHint: () => void,
	) {
		super({
			...options,
			description: undefined,
			spaceBeforeControl: false,
			spaceAfterControl: false,
			summary: options.description ? [new Text(theme.fg("muted", options.description), 0, 0)] : undefined,
		});
	}
	override render(width: number): readonly string[] {
		this.onHint();
		return super.render(width);
	}
}

class ModelSelectorSubmenu implements Component {
	#active: SelectFormField | ModelBrowser;
	constructor(
		source: ModelBrowserSource,
		registry: ModelBrowserRegistry | undefined,
		current: string,
		onSelect: (selector: string) => void,
		onCancel: () => void,
		private readonly getHeight: () => number,
	) {
		const roles = (): SelectFormField =>
			createSettingsSelectField(
				"Classifier Model",
				"Automatic uses @tiny; no active-model fallback.",
				[
					{ value: "", label: "Automatic (@tiny)" },
					...source.knownRoleIds.map(role => ({
						value: "@" + role,
						label: "@" + role,
						description: source.getRoleInfo(role).name,
					})),
					{ value: "__browse", label: "Browse models…" },
				],
				current,
				value => {
					if (value !== "__browse") {
						onSelect(value);
						return;
					}
					const browser = new ModelBrowser(source, {
						emptyText: () => "No models available — configure provider credentials.",
					});
					const available = registry?.getAvailable() ?? [];
					const assignments = resolveRoleAssignments(source, registry?.getAll() ?? [], available);
					const items = buildBrowserItems(available);
					sortModelItems(items, { roles: assignments, mruOrder: source.mruOrder });
					browser.setItems(items);
					browser.setRoles(assignments);
					browser.setMruOrder(source.mruOrder);
					browser.setPerfStats(source.modelPerf);
					browser.onActivate = item => onSelect(item.selector);
					browser.onCancel = () => {
						this.#active = roles();
					};
					this.#active = browser;
				},
				onCancel,
			);
		this.#active = roles();
	}
	render(width: number): readonly string[] {
		if (this.#active instanceof ModelBrowser) this.#active.setMaxVisible(Math.max(1, this.getHeight() - 5));
		else this.#active.selectList.setMaxVisible(Math.max(1, this.getHeight() - 5));
		return this.#active.render(width);
	}
	handleInput(data: string): void {
		this.#active.handleInput(data);
	}
	routeMouse(event: SgrMouseEvent, line: number, col: number): void {
		if (this.#active instanceof ModelBrowser) this.#active.routeMouse(event, line);
		else this.#active.routeMouse(event, line, col);
	}
	invalidate(): void {
		this.#active.invalidate();
	}
}

/**
 * Submenu for array-of-enum settings: every option is a toggle row. Enter or
 * Space flips membership; ordered lists render 1-based positions and reorder
 * the highlighted member with ←/→. Changes apply live; Esc goes back.
 */
class MultiSelectSubmenu extends Container {
	#selectList!: SelectList;
	#field!: FormField;
	#value: string[];
	#cursor = 0;
	#pressedItemId: string | undefined;
	#dropItemId: string | undefined;
	readonly #title: string;
	readonly #description: string;
	readonly #options: ReadonlyArray<SelectItem>;
	readonly #ordered: boolean;
	readonly #onApply: (value: string[]) => void;
	readonly #onClose: () => void;

	constructor(
		title: string,
		description: string,
		options: ReadonlyArray<SelectItem>,
		initial: readonly string[],
		ordered: boolean,
		onApply: (value: string[]) => void,
		onClose: () => void,
	) {
		super();
		this.#title = title;
		this.#description = description;
		this.#options = options;
		this.#ordered = ordered;
		this.#onApply = onApply;
		this.#onClose = onClose;
		// Drop stale ids (renamed/removed providers) so positions stay contiguous.
		this.#value = initial.filter(id => options.some(option => option.value === id));
		this.#rebuild();
	}

	#rebuild(): void {
		this.clear();

		const items = this.#options.map((option): SelectItem => {
			const position = this.#value.indexOf(option.value);
			const mark =
				position === -1
					? theme.fg("dim", this.#ordered ? " · " : " ○ ")
					: this.#ordered
						? theme.fg("accent", `${String(position + 1).padStart(2)}.`)
						: theme.fg("accent", " ● ");
			return { value: option.value, label: `${mark} ${option.label}`, description: option.description };
		});
		this.#selectList = new SelectList(items, Math.min(items.length, 12), getSelectListTheme());
		this.#selectList.setSelectedIndex(this.#cursor);
		this.#selectList.onSelect = item => this.#toggle(item.value);
		this.#selectList.onSelectionChange = item => {
			this.#cursor = this.#options.findIndex(option => option.value === item.value);
		};
		this.#selectList.onCancel = this.#onClose;
		const hint = this.#ordered
			? "  Click to toggle · drag selected items to reorder · ←/→ move · 1-9 place · Esc to go back"
			: "  Click/Enter/Space to toggle · Esc to go back";
		this.#field = new FormField(this.#selectList, {
			theme: formTheme,
			label: this.#title,
			description: this.#description || undefined,
			hint,
		});
		this.addChild(this.#field);
	}

	#apply(next: string[]): void {
		this.#value = next;
		this.#onApply([...next]);
		this.#rebuild();
	}

	#toggle(id: string): void {
		const next = this.#value.includes(id) ? this.#value.filter(v => v !== id) : [...this.#value, id];
		this.#apply(next);
	}

	#move(id: string, delta: -1 | 1): void {
		const from = this.#value.indexOf(id);
		if (from === -1) return;
		const to = from + delta;
		if (to < 0 || to >= this.#value.length) return;
		const next = [...this.#value];
		next[from] = next[to]!;
		next[to] = id;
		this.#apply(next);
	}

	/** Move a selected item before another selected item, retaining every other preference. */
	#moveBefore(id: string, beforeId: string): void {
		if (id === beforeId) return;
		const next = this.#value.filter(value => value !== id);
		const target = next.indexOf(beforeId);
		if (target === -1) return;
		next.splice(target, 0, id);
		this.#apply(next);
	}

	/** Splice the option into the 1-based `position` of the selection (adding it if unselected). */
	#placeAt(id: string, position: number): void {
		const next = this.#value.filter(v => v !== id);
		next.splice(Math.min(position - 1, next.length), 0, id);
		this.#apply(next);
	}

	routeMouse(event: SgrMouseEvent, line: number, _col: number): void {
		const controlLine = this.#field.controlLineAt(line);
		if (controlLine === undefined) return;
		const itemIndex = this.#selectList.hitTest(controlLine);
		if (event.wheel !== null) {
			routeSelectListMouse(this.#selectList, event, controlLine);
			return;
		}
		if (event.motion) {
			this.#selectList.setHoverIndex(itemIndex ?? null);
			const target = itemIndex === undefined ? undefined : this.#options[itemIndex]?.value;
			if (
				this.#ordered &&
				this.#pressedItemId !== undefined &&
				target !== undefined &&
				target !== this.#pressedItemId &&
				this.#value.includes(target)
			) {
				this.#dropItemId = target;
			}
			return;
		}
		if (event.leftClick && itemIndex !== undefined) {
			const item = this.#options[itemIndex];
			if (!item) return;
			this.#cursor = itemIndex;
			this.#selectList.setSelectedIndex(itemIndex);
			this.#pressedItemId = item.value;
			this.#dropItemId = item.value;
			return;
		}
		if (!event.release) return;

		const pressedItemId = this.#pressedItemId;
		const dropItemId = this.#dropItemId;
		this.#pressedItemId = undefined;
		this.#dropItemId = undefined;
		if (!pressedItemId) return;
		if (this.#ordered && dropItemId !== undefined && dropItemId !== pressedItemId) {
			this.#moveBefore(pressedItemId, dropItemId);
			return;
		}
		this.#toggle(pressedItemId);
	}

	handleInput(data: string): void {
		const current = this.#options[this.#cursor]?.value;
		if (data === " " && current !== undefined) {
			this.#toggle(current);
			return;
		}
		if (this.#ordered && current !== undefined && (data === "\x1b[D" || data === "\x1b[C")) {
			this.#move(current, data === "\x1b[D" ? -1 : 1);
			return;
		}
		if (this.#ordered && current !== undefined && data.length === 1 && data >= "1" && data <= "9") {
			this.#placeAt(current, Number(data));
			return;
		}
		this.#selectList.handleInput(data);
	}
}

class ProviderLimitsSubmenu extends Container {
	#listField: SelectFormField | undefined;
	readonly #settings: SettingsHost;
	readonly #providers: readonly string[];
	readonly #onChange: (value: Record<string, number>) => void;
	readonly #onCancel: () => void;
	readonly #requestRender: (() => void) | undefined;

	constructor(
		settings: SettingsHost,
		providers: readonly string[],
		onChange: (value: Record<string, number>) => void,
		onCancel: () => void,
		requestRender?: () => void,
	) {
		super();
		this.#settings = settings;
		this.#providers = providers;
		this.#onChange = onChange;
		this.#onCancel = onCancel;
		this.#requestRender = requestRender;
		this.#showProviderList();
	}

	#providerIds(): string[] {
		const limits = this.#settings.normalizeProviderLimits(this.#settings.get("providers.maxInFlightRequests"));
		return [...new Set([...this.#providers, ...Object.keys(limits)])].sort((a, b) => a.localeCompare(b));
	}

	#showProviderList(): void {
		this.clear();

		const limits = this.#settings.normalizeProviderLimits(this.#settings.get("providers.maxInFlightRequests"));
		const providerItems = this.#providerIds().map((provider): SelectItem => {
			const limit = limits[provider];
			return {
				value: provider,
				label: provider,
				description: limit === undefined ? "Unlimited" : `Limit: ${limit}`,
			};
		});
		const clearItem: SelectItem[] =
			Object.keys(limits).length === 0
				? []
				: [{ value: "__clear_all", label: "Clear all limits", description: "Make every provider unlimited" }];
		const items = [...providerItems, ...clearItem];
		this.#listField = new SelectFormField({
			theme: formTheme,
			label: "Max In-Flight Requests",
			description:
				"Select a provider, enter a positive number to cap concurrent LLM requests, or clear it for unlimited.",
			items,
			maxVisible: 12,
			selectTheme: getSelectListTheme(),
			hint: "  Enter to edit provider · Esc to go back",
			onSubmit: value => {
				if (value === "__clear_all") {
					this.#settings.set("providers.maxInFlightRequests", {});
					this.#onChange({});
					this.#showProviderList();
					this.#requestRender?.();
					return;
				}
				this.#showProviderEditor(value);
			},
			onCancel: this.#onCancel,
			requestRender: this.#requestRender,
		});
		this.addChild(this.#listField);
	}

	#showProviderEditor(provider: string): void {
		const limits = this.#settings.normalizeProviderLimits(this.#settings.get("providers.maxInFlightRequests"));
		this.clear();
		this.#listField = undefined;
		this.addChild(
			new TextFormField({
				theme: formTheme,
				label: `Max In-Flight Requests: ${provider}`,
				description:
					"Enter a positive number. Decimals round down. Clear the field to make this provider unlimited.",
				initialValue: limits[provider]?.toString() ?? undefined,
				empty: "submit",
				hint: "  Enter to save · Esc to cancel · Clear field to unset",
				validate: value => {
					if (value.trim() === "") return undefined;
					const limit = Number(value.trim());
					if (!Number.isFinite(limit) || limit <= 0) return "Limit must be a positive number.";
					return undefined;
				},
				onSubmit: value => {
					const next = { ...limits };
					const trimmed = value.trim();
					if (trimmed === "") {
						delete next[provider];
					} else {
						const limit = Number(trimmed);
						if (!Number.isFinite(limit) || limit <= 0) throw new Error("Limit must be a positive number.");
						next[provider] = Math.max(1, Math.floor(limit));
					}
					const normalized = this.#settings.validateProviderLimits(next);
					this.#settings.set("providers.maxInFlightRequests", normalized);
					this.#onChange(normalized);
					this.#showProviderList();
					this.#requestRender?.();
				},
				onCancel: () => {
					this.#showProviderList();
					this.#requestRender?.();
				},
				requestRender: this.#requestRender,
			}),
		);
	}

	routeMouse(event: SgrMouseEvent, line: number, col: number): void {
		this.#listField?.routeMouse(event, line, col);
	}

	handleInput(data: string): void {
		if (this.#listField) {
			this.#listField.handleInput(data);
			return;
		}
		this.children[0]?.handleInput?.(data);
	}
}

/** Stable sidebar width derived from the host's complete schema. */
function settingsSidebarWidth(entries: readonly SettingsDisplayEntry[]): number {
	let nameWidth = 0;
	for (const tab of SETTING_TABS) {
		for (const def of getSettingsForTab(entries, tab)) {
			if (def.group) nameWidth = Math.max(nameWidth, visibleWidth(def.group));
		}
	}
	return Math.min(22, nameWidth) + 4;
}

function getSettingsTabs(): Tab[] {
	return [
		...SETTING_TABS.map(id => {
			const meta = TAB_METADATA[id];
			const icon = theme.symbol(meta.icon);
			return { id, label: `${icon} ${meta.label}`, short: icon };
		}),
		{ id: "plugins", label: `${theme.icon.package} Plugins`, short: theme.icon.package },
	];
}

/**
 * Dynamic context for settings that need runtime data.
 * Some settings (like thinking level) are managed by the session, not Settings.
 */
export interface SettingsSelectorOptions {
	initialTab?: SettingTab;
	initialSettingPath?: string;
}

export interface SettingsRuntimeContext {
	modelSource?: ModelBrowserSource;
	modelRegistry?: ModelBrowserRegistry;
	maxContextTokens?: number;
	getPreservationLimitUsage?: (path: string) => string | undefined;
	settings: SettingsHost;
	plugins: PluginSettingsHost;
	/** Available thinking levels (from session) */
	availableThinkingLevels: Effort[];
	/** Current thinking level (from session) */
	thinkingLevel: ThinkingLevel | undefined;
	/** Available themes */
	availableThemes: string[];
	/** Provider/source ids shown in /model. */
	providers: string[];
	/** Active model (api + id); resolves what the snapcompact `auto` shape maps to. */
	model?: ShapeTarget;
	/** Shared TUI image budget (graphics ids + transmit-once) for image previews. */
	imageBudget?: ImageBudget;
	/** Schedules a re-render after async preview work completes. */
	requestRender?: () => void;
	/** Live status renderer for composer-shape previews (the session's status line). */
	composerPreviewStatus?: ComposerPreviewStatusSource;
}

/** Status line settings subset for preview */
export interface StatusLinePreviewSettings {
	preset?: StatusLinePreset;
	contextLine?: ContextLineMode;
	leftSegments?: StatusLineSegmentId[];
	rightSegments?: StatusLineSegmentId[];
	separator?: StatusLineSeparatorStyle;
	sessionAccent?: boolean;
	transparent?: boolean;
	compactThinkingLevel?: boolean;
}

export interface SettingsCallbacks {
	/** Called when any setting value changes */
	onChange: (path: string, newValue: unknown) => void;
	/** Called for theme preview while browsing */
	onThemePreview?: (theme: string) => void | Promise<void>;
	/** Called for status line preview while configuring */
	onStatusLinePreview?: (settings: StatusLinePreviewSettings) => void;
	/** Get current rendered status line for inline preview */
	getStatusLinePreview?: () => string;
	/** Called when plugins change */
	onPluginsChanged?: () => void | Promise<void>;
	/** Called when settings panel is closed */
	onCancel: () => void;
}

/**
 * Main tabbed settings selector component.
 * Uses declarative settings definitions from settings-defs.ts.
 */
export class SettingsSelectorComponent implements Component {
	#tabBar: TabBar;
	#currentList: SettingsList | null = null;
	#searchList: SettingsList | null = null;
	#pluginComponent: PluginSettingsComponent | null = null;
	#currentTabId: SettingTab | "plugins" = "appearance";
	#preSearchTabId: SettingTab | "plugins" = "appearance";
	#searchQuery = "";
	/** Single-line editor backing the search banner (cursor, word ops, paste). */
	#searchInput = new Input();
	#searchMatchCount = 0;
	/** First matching item id per tab id, for Tab-key jumps while searching. */
	#searchFirstMatch = new Map<string, string>();
	#textInputActive = false;
	#hasSectionJump = false;
	// Frame geometry from the last render, for mouse hit-testing (the
	// fullscreen overlay paints from screen row 0, so mouse rows map 1:1).
	#tabRowStart = 0;
	#tabRowCount = 0;
	#contentRowStart = 0;
	#contentRowCount = 0;
	#sidebarWidth: number;
	readonly #context: SettingsRuntimeContext;
	readonly #callbacks: SettingsCallbacks;
	#submenuHint = "";

	constructor(context: SettingsRuntimeContext, callbacks: SettingsCallbacks, options: SettingsSelectorOptions = {}) {
		this.#context = context;
		this.#callbacks = callbacks;
		this.#sidebarWidth = settingsSidebarWidth(context.settings.entries);
		// No label prefix (the frame title already says Settings) and no
		// "(tab to cycle)" hint (folded into the footer hint line).
		this.#tabBar = new TabBar("", getSettingsTabs(), getTabBarTheme());
		this.#tabBar.showHint = false;
		this.#tabBar.onTabChange = () => {
			const tabId = this.#tabBar.getActiveTab().id as SettingTab | "plugins";
			if (this.#searchList) {
				// While searching, tabs act as jump targets into the result list.
				const firstId = this.#searchFirstMatch.get(tabId);
				if (firstId) this.#searchList.selectItem(firstId);
				return;
			}
			this.#switchToTab(tabId);
		};

		const path = options.initialSettingPath;
		const categoryPath = context.settings.preservation?.categories.some(category => category.path === path)
			? context.settings.preservation.categories[0]?.path
			: path;
		const initialTab =
			options.initialTab ??
			(categoryPath ? getSettingDef(context.settings.entries, categoryPath)?.tab : undefined) ??
			"appearance";
		this.#tabBar.setActiveById(initialTab);
		this.#switchToTab(initialTab);
		if (categoryPath) this.#currentList?.selectItem(categoryPath);
	}

	invalidate(): void {
		this.#tabBar.invalidate();
		this.#currentList?.invalidate();
		this.#searchList?.invalidate();
		this.#pluginComponent?.invalidate();
	}

	/** Swap the active content (per-tab list, search list, or plugins). */
	#setContent(build: () => void): void {
		this.#currentList = null;
		this.#searchList = null;
		this.#pluginComponent = null;
		build();
	}

	#switchToTab(tabId: SettingTab | "plugins"): void {
		this.#currentTabId = tabId;
		this.#setContent(() => {
			if (tabId === "plugins") {
				this.#showPluginsTab();
			} else {
				this.#showSettingsTab(tabId);
			}
		});
	}

	#footerHintText(): string {
		const list = this.#searchList ?? this.#currentList;
		if (list?.hasOpenSubmenu()) return this.#submenuHint || "Enter select · Esc back";
		if (this.#searchList) {
			return "Enter to change · Tab to jump tabs · Esc to exit search";
		}
		if (this.#currentTabId === "plugins") {
			return "Tab to switch tabs · Esc to close";
		}
		if (this.#currentList?.sectionFocused) {
			return "↑/↓ to jump sections · Tab/Enter to settings · ←/→ to switch tabs · Esc to close";
		}
		const nav = this.#hasSectionJump ? "Tab to jump sections · ←/→ to switch tabs" : "Tab to switch tabs";
		const assignment = (this.#currentList?.getSelectedItem() as PreservationSettingItem | undefined)?.assignment;
		return assignment
			? `Space toggle · y on · n off · ${nav} · / search · Esc close`
			: `Enter/Space to change · ${nav} · Type or / to search · Esc to close`;
	}

	/** Single-line search banner: accent icon, editable query with live cursor, right-aligned match count. */
	#renderSearchBanner(width: number): string {
		const icon = theme.symbol("icon.search");
		const countText = this.#searchMatchCount === 1 ? "1 match" : `${this.#searchMatchCount} matches`;
		const rightWidth = visibleWidth(countText) + 1; // trailing margin
		const prefix = ` ${theme.fg("accent", icon)} `;
		// The input pads itself to exactly this width and keeps the cursor in view.
		const inputWidth = Math.max(4, width - visibleWidth(prefix) - rightWidth - 1);
		const inputLine = this.#searchInput.render(inputWidth)[0] ?? "";
		const count = theme.fg(this.#searchMatchCount > 0 ? "dim" : "warning", countText);
		return truncateToWidth(`${prefix}${theme.bold(inputLine)} ${count} `, width);
	}

	/**
	 * Fullscreen frame: title border, tab row, divider, optional search banner,
	 * the active content sized to fill the terminal, the appearance preview,
	 * then a footer hint pinned above the bottom border.
	 */
	render(width: number): readonly string[] {
		this.#submenuHint = "";
		const height = Math.max(14, process.stdout.rows || 40);
		const innerWidth = Math.max(1, width - 4);

		const tabLines = this.#tabBar.render(innerWidth);
		const searching = this.#searchList !== null;
		const showPreview = !searching && this.#currentTabId === "appearance";
		const previewLines = showPreview ? ["", theme.fg("muted", "Preview:"), this.#getStatusPreviewString()] : [];

		// Fixed chrome: top border, tabs, divider, [search row], divider, hint, bottom border.
		const fixedRows = 1 + tabLines.length + 1 + (searching ? 1 : 0) + 1 + 1 + 1;
		const contentRows = Math.max(7, height - fixedRows - previewLines.length);

		const list = this.#searchList ?? this.#currentList;
		let contentLines: readonly string[];
		if (list) {
			// SettingsList pads itself to viewport + blank + 3 description rows.
			list.setMaxVisible(contentRows - 4);
			contentLines = list.render(innerWidth);
		} else if (this.#pluginComponent) {
			contentLines = this.#pluginComponent.render(innerWidth);
		} else {
			contentLines = [];
		}

		const out: string[] = [];
		out.push(topBorder(width, "Settings"));
		this.#tabRowStart = out.length;
		this.#tabRowCount = tabLines.length;
		for (const line of tabLines) {
			out.push(row(line, width));
		}
		out.push(divider(width));
		if (searching) {
			out.push(row(this.#renderSearchBanner(innerWidth), width));
		}
		this.#contentRowStart = out.length;
		this.#contentRowCount = contentRows;
		for (let i = 0; i < contentRows; i++) {
			out.push(row(contentLines[i] ?? "", width));
		}
		for (const line of previewLines) {
			out.push(row(line, width));
		}
		out.push(divider(width));
		out.push(row(theme.fg("dim", this.#footerHintText()), width));
		out.push(bottomBorder(width));
		return out;
	}

	/**
	 * Route an SGR mouse report against the frame geometry of the last render.
	 * Wheel scrolls the focused list, motion drives the hover highlights (tabs
	 * and rows), and a left click activates: tabs switch (or jump, while
	 * searching), a row click selects, and a click on the already-selected row
	 * activates it (toggle / open submenu).
	 */
	#handleMouse(data: string): boolean {
		return routeSgrMouseInput(data, event => this.#routeMouseEvent(event));
	}

	#routeMouseEvent(event: SgrMouseEvent): boolean {
		const list = this.#searchList ?? this.#currentList;
		// row() insets content by the border column plus a space.
		const contentColInset = 2;
		const innerCol = event.col - contentColInset;
		const contentLine = event.row - this.#contentRowStart;

		// An open submenu owns the pointer: wheel, hover, and clicks route into
		// it (text-input submenus ignore routed events).
		if (list?.hasOpenSubmenu()) {
			list.routeSubmenuMouse(event, contentLine, innerCol);
			return true;
		}

		const tabLine = event.row - this.#tabRowStart;
		const overTabs = tabLine >= 0 && tabLine < this.#tabRowCount;
		const overContent = contentLine >= 0 && contentLine < this.#contentRowCount;

		if (event.wheel !== null) {
			if (overContent) {
				list?.handleWheelAt(event.wheel, contentLine, innerCol);
			}
			return true;
		}

		if (event.motion) {
			const hovered = overTabs ? this.#tabBar.tabAt(tabLine, innerCol) : undefined;
			this.#tabBar.setHoverTab(hovered && !hovered.muted ? hovered.id : null);
			// hoverTest: never light up pane rows while the pointer is on the
			// sidebar — only rows the pointer is actually on.
			list?.setHoverItem(overContent ? (list.hoverTest(contentLine, innerCol) ?? null) : null);
			return true;
		}
		if (!event.leftClick) return true;

		if (overTabs) {
			const tab = this.#tabBar.tabAt(tabLine, innerCol);
			if (tab) this.#tabBar.selectTab(tab.id);
			return true;
		}
		if (overContent && list) {
			const itemId = list.hoverTest(contentLine, innerCol);
			const id = itemId ?? list.hitTest(contentLine, innerCol);
			if (id !== undefined) {
				const wasSelected = list.getSelectedItem()?.id === id;
				list.selectItem(id);
				// Only repeated setting-row clicks activate. Sidebar section clicks navigate.
				if (wasSelected && itemId !== undefined) list.handleInput("\n");
			}
		}
		return true;
	}

	// ═══════════════════════════════════════════════════════════════════════
	// Global search (type-to-search across every tab)
	// ═══════════════════════════════════════════════════════════════════════

	/** Swap the tab content for the global search result list. */
	#startSearch(initialQuery: string): void {
		this.#preSearchTabId = this.#currentTabId;
		this.#searchInput = new Input();
		this.#searchInput.prompt = "";
		this.#searchInput.setValue(initialQuery);
		const list = new SettingsList(
			[],
			10,
			getSettingsListTheme(),
			(id, newValue) => this.#onSearchSettingChange(id, newValue),
			() => this.#callbacks.onCancel(),
			{
				layout: "flat",
				typeToSearch: false,
				emptyText: "No matching settings",
				hint: "",
			},
		);
		// Keep the footer tab highlight on the tab owning the selected result.
		list.onSelectionChange = item => this.#syncTabBarToSelection(item);
		this.#setContent(() => {
			this.#searchList = list;
		});
		this.#setSearchQuery(initialQuery);
	}

	/**
	 * Recompute matches across every settings tab. Results render as one flat
	 * list with a heading row per tab; the footer tab bar reorders to show
	 * matching tabs (with counts) first and the rest muted at the end.
	 */
	#setSearchQuery(query: string): void {
		if (!this.#searchList) return;

		this.#searchQuery = query;

		const counts = new Map<SettingTab, number>();
		const items: SettingItem[] = [];
		const tabResults: { tab: SettingTab; matched: SettingItem[]; bestScore: number; order: number }[] = [];
		this.#searchFirstMatch.clear();
		let total = 0;
		for (const tab of SETTING_TABS) {
			const candidates: SettingItem[] = [];
			for (const def of getSettingsForTab(this.#context.settings.entries, tab)) {
				const item = this.#defToItem(def);
				if (item) candidates.push(item);
			}
			const ranked = fuzzyRank(candidates, query, getSettingItemFilterText);
			const matched = ranked.map(result => result.item);
			counts.set(tab, matched.length);
			if (matched.length === 0) continue;
			total += matched.length;
			tabResults.push({
				tab,
				matched,
				bestScore: ranked[0]?.score ?? 0,
				order: SETTING_TABS.indexOf(tab),
			});
		}

		tabResults.sort((a, b) => a.bestScore - b.bestScore || a.order - b.order);
		for (const result of tabResults) {
			const meta = TAB_METADATA[result.tab];
			items.push({
				id: `__tab:${result.tab}`,
				label: `${theme.symbol(meta.icon)} ${meta.label}`,
				currentValue: "",
				heading: true,
			});
			this.#searchFirstMatch.set(result.tab, result.matched[0]?.id ?? "");
			items.push(...result.matched);
		}

		this.#searchList.setItems(items);
		this.#searchMatchCount = total;
		this.#tabBar.setTabs(
			this.#buildSearchTabs(
				counts,
				tabResults.map(result => result.tab),
			),
		);
		this.#syncTabBarToSelection(this.#searchList.getSelectedItem());
	}

	/**
	 * Leave search mode. With `jumpToSelection`, land on the tab containing
	 * the selected result and keep it selected there — search doubles as
	 * navigation. Otherwise restore the pre-search tab.
	 */
	#endSearch(jumpToSelection: boolean): void {
		if (!this.#searchList) return;
		const selected = jumpToSelection ? this.#searchList.getSelectedItem() : undefined;
		const selectedDef = selected ? getSettingDef(this.#context.settings.entries, selected.id) : undefined;
		const targetTab: SettingTab | "plugins" = selectedDef?.tab ?? this.#preSearchTabId;

		this.#searchQuery = "";
		this.#searchFirstMatch.clear();
		this.#searchMatchCount = 0;
		this.#tabBar.setTabs(getSettingsTabs(), targetTab);
		this.#switchToTab(targetTab);
		if (selectedDef) {
			this.#currentList?.selectItem(selectedDef.path);
		}
	}

	/** Matching tabs first (counts attached), ordered by best result score; the rest stay muted at the end. */
	#buildSearchTabs(counts: Map<SettingTab, number>, matchedTabOrder: readonly SettingTab[]): Tab[] {
		const matched: Tab[] = [];
		const empty: Tab[] = [];
		const matchedIds = new Set<SettingTab>(matchedTabOrder);
		for (const id of matchedTabOrder) {
			const meta = TAB_METADATA[id];
			const icon = theme.symbol(meta.icon);
			const count = counts.get(id) ?? 0;
			if (count > 0) {
				matched.push({ id, label: `${icon} ${meta.label} (${count})`, short: `${icon} ${count}` });
			}
		}
		for (const id of SETTING_TABS) {
			if (matchedIds.has(id)) continue;
			const meta = TAB_METADATA[id];
			const icon = theme.symbol(meta.icon);
			empty.push({ id, label: `${icon} ${meta.label}`, short: icon, muted: true });
		}
		// Plugins hosts its own UI; it is not part of the schema-backed search.
		empty.push({
			id: "plugins",
			label: `${theme.icon.package} Plugins`,
			short: theme.icon.package,
			muted: true,
		});
		return [...matched, ...empty];
	}

	#syncTabBarToSelection(item: SettingItem | undefined): void {
		if (!this.#searchList || !item) return;
		const def = getSettingDef(this.#context.settings.entries, item.id);
		if (def) this.#tabBar.setActiveById(def.tab);
	}

	/** Value-change dispatch for the search result list (any tab's setting). */
	#onSearchSettingChange(path: string, newValue: string): void {
		const def = getSettingDef(this.#context.settings.entries, path);
		if (!def) return;
		if (def.type === "boolean") {
			const boolValue = newValue === "true";
			this.#context.settings.set(path, boolValue);
			this.#callbacks.onChange(path, boolValue);
		} else if (def.type === "enum") {
			this.#context.settings.set(path, newValue);
			this.#callbacks.onChange(path, newValue);
		}
		// Submenu/text types already persisted inside their own done callbacks.
		if (def.tab === "appearance") {
			this.#triggerStatusLinePreview();
		}
		// Values feed the searchable text and condition gates may have flipped:
		// recompute results in place (selection is preserved by item id).
		this.#setSearchQuery(this.#searchQuery);
	}

	/**
	 * Convert a setting definition to a SettingItem for the UI.
	 */
	#defToItem(def: SettingDef): PreservationSettingItem | null {
		// Check condition: applies to every variant — booleans, enums, submenus, text inputs.
		if (def.condition && !def.condition()) {
			return null;
		}

		const currentValue = this.#getCurrentValue(def);
		const item = {
			id: def.path,
			label: def.label,
			description:
				def.path === "compaction.keepUserMessagesLlm" && !this.#context.settings.get("compaction.keepUserMessages")
					? `Live tagging inactive while Remember User Messages is off. Saved toggle is preserved. ${def.description}`
					: def.description,
			warning: def.warning,
			changed: this.#isChanged(def, currentValue),
		};

		switch (def.type) {
			case "preservationLimit":
				return {
					...item,
					currentValue: this.#limitLabel(currentValue),
					submenu: (_cv, done) => this.#createLimit(def, done),
				};
			case "preservationCap":
				return { ...item, currentValue: this.#capSummary(), submenu: (_cv, done) => this.#createCap(done) };
			case "categoryDispositions":
				return {
					...item,
					currentValue: this.#categorySummary(),
					submenu: (_cv, done) => this.#createCategories(done),
				};
			case "regexRules":
				return {
					...item,
					currentValue: `${Object.keys(this.#regexRules()).length} rules`,
					submenu: (_cv, done) => this.#createRegexRules(done),
				};
			case "modelSelector":
				return {
					...item,
					currentValue: String(currentValue || "Automatic (@tiny)"),
					submenu: (_cv, done) => this.#createModelSelector(def, done),
				};
			case "positiveTokens":
				return {
					...item,
					currentValue: String(currentValue),
					submenu: (_cv, done) =>
						this.#validatedInput(
							def.label,
							"Positive safe integer; no arbitrary maximum.",
							String(currentValue),
							value => {
								if (!Number.isSafeInteger(Number(value)) || Number(value) <= 0)
									throw new Error("Enter a positive safe integer");
								this.#save(def.path, Number(value));
								done(value);
							},
							() => done(),
						),
				};

			case "boolean":
				return {
					...item,
					currentValue: currentValue ? "true" : "false",
					values: ["true", "false"],
					assignment: def.menuconfig
						? {
								kind: "boolean",
								get: () => this.#context.settings.get(def.path) === true,
								set: value => {
									this.#save(def.path, value);
									this.#refreshCurrentTabItems(getSettingsForTab(this.#context.settings.entries, def.tab));
								},
							}
						: undefined,
				};

			case "enum":
				return { ...item, currentValue: String(currentValue ?? ""), values: [...def.values] };

			case "submenu":
				return {
					...item,
					currentValue: this.#getSubmenuCurrentValue(def.path, currentValue),
					submenu: (cv, done) => this.#createSubmenu(def, cv, done),
				};

			case "text":
				return {
					...item,
					currentValue: this.#formatTextInputValue(def, currentValue),
					submenu: (cv, done) => this.#createTextInput(def, cv, done),
				};

			case "providerLimits":
				return {
					...item,
					currentValue: this.#formatProviderLimitsValue(currentValue),
					submenu: (_cv, done) => this.#createProviderLimitsInput(done),
				};

			case "multiselect":
				return {
					...item,
					currentValue: this.#formatMultiSelectValue(def, currentValue),
					submenu: (_cv, done) => this.#createMultiSelect(def, done),
				};
		}
	}

	#save(path: string, value: unknown): void {
		this.#context.settings.set(path, value);
		this.#callbacks.onChange(path, value);
	}

	#settingsMenu(
		title: string,
		items: PreservationSettingItem[],
		onChange: (id: string, value: string) => void,
		onCancel: () => void,
		getHeight: () => number,
		onRender?: () => void,
	): SettingsSubmenu {
		return new SettingsSubmenu(title, items, onChange, onCancel, getHeight, onRender, hint => {
			this.#submenuHint = hint;
		});
	}

	#preservationSelectField(
		label: string,
		description: string,
		items: ReadonlyArray<SelectItem>,
		currentValue: string,
		onSubmit: (value: string) => void,
		onCancel: () => void,
	): PreservationSelectField {
		return new PreservationSelectField(
			{
				theme: formTheme,
				label,
				description,
				items,
				currentValue,
				selectTheme: getSelectListTheme(),
				onSubmit,
				onCancel,
				requestRender: this.#context.requestRender,
			},
			() => {
				this.#submenuHint = "Enter select · Esc back";
			},
			() => this.#contentRowCount,
		);
	}

	#validatedInput(
		label: string,
		description: string,
		current: string,
		submit: (value: string) => void,
		cancel: () => void,
	): TextFormField {
		return new PreservationTextField(
			{
				theme: formTheme,
				label,
				description,
				initialValue: current,
				empty: "submit",
				hint: "Enter to save · Esc to cancel",
				onSubmit: submit,
				onCancel: cancel,
				requestRender: this.#context.requestRender,
			},
			() => {
				this.#submenuHint = "Enter save · Esc cancel";
			},
		);
	}

	#regexRules(): Record<string, PreservationRegexRule> {
		return this.#context.settings.get("compaction.keepUserMessagesRegexRules") as Record<
			string,
			PreservationRegexRule
		>;
	}

	#limitLabel(value: unknown): string {
		const limit = this.#context.settings.preservation!.parseLimit(value);
		if (!limit) return `Invalid: ${String(value)}`;
		if (!("value" in limit)) return limit.mode === "off" ? "Off" : "All";
		return limit.mode === "context-percent" ? `${limit.value}%` : `${limit.value} ${limit.mode}`;
	}

	#limitSummary(value: unknown): string {
		const limit = this.#context.settings.preservation!.parseLimit(value);
		if (!limit) return "Invalid limit — choose a valid mode/value.";
		if (limit.mode !== "context-percent")
			return `Effective limit: ${this.#limitLabel(value)}. Keep whole messages; stop before the first one that exceeds the budget.`;
		const maximum = this.#context.maxContextTokens;
		return maximum !== undefined && Number.isFinite(maximum) && maximum > 0
			? `${limit.value}% of the model maximum (${maximum} tokens) = ${Math.floor((maximum * limit.value) / 100)} tokens. Not a percentage of current usage or free space.`
			: `${limit.value}% configured; the token budget needs the active model's maximum context size.`;
	}

	#createLimit(def: SettingDef, done: (value?: string) => void): SettingsSubmenu {
		let displayedValue: unknown;
		let displayedMaximum: number | undefined;
		const usage: SettingItem = {
			id: "usage",
			label: "Current selection",
			currentValue: "Unavailable",
			description: "Current selection counts appear once message retention has been calculated for this session.",
		};
		const save = (limit: PreservationLimit) => {
			this.#save(def.path, this.#context.settings.preservation!.serializeLimit(limit));
			menu.list.setItems(items());
		};
		const items = (): SettingItem[] => {
			const raw = this.#context.settings.get(def.path);
			displayedValue = raw;
			displayedMaximum = this.#context.maxContextTokens;
			const limit = this.#context.settings.preservation!.parseLimit(raw);
			const rows: SettingItem[] = [
				{
					id: "mode",
					label: "Mode",
					currentValue: limit?.mode ?? "Invalid",
					description: def.description,
					submenu: (_cv, close) =>
						this.#preservationSelectField(
							"Limit Mode",
							"Off selects none here; All removes this limit. Messages counts whole messages; Tokens budgets their content; % uses the model maximum, not current usage. Linked Rule / Manual Keep treats Off and All as uncapped.",
							[
								{ value: "off", label: "Off", description: "No messages from this selection" },
								{ value: "all", label: "All", description: "Every eligible message" },
								{ value: "messages", label: "Messages", description: "A positive whole-message count" },
								{ value: "tokens", label: "Tokens", description: "A token allowance, including zero" },
								{
									value: "context-percent",
									label: "% maximum context",
									description: "0–100% of the model context size",
								},
							],
							limit?.mode ?? "",
							mode => {
								if (mode === "off" || mode === "all") save({ mode });
								else if (mode === "messages" || mode === "tokens" || mode === "context-percent") {
									save({
										mode,
										value:
											limit && "value" in limit && limit.mode === mode
												? limit.value
												: mode === "messages"
													? 1
													: 0,
									});
								}
								close(mode);
							},
							() => close(),
						),
				},
			];
			if (limit && "value" in limit) {
				const mode = limit.mode;
				const description =
					mode === "messages"
						? "Maximum number of whole messages to keep; enter a positive whole number."
						: mode === "tokens"
							? "Token allowance for kept messages; enter a whole number of zero or more. Zero is a zero-token allowance, not Off or All."
							: "Percentage of the model maximum context size: 0–100, decimals allowed. 0% is a zero-token allowance.";
				rows.push({
					id: "value",
					label: "Value",
					currentValue: String(limit.value),
					description,
					submenu: (_cv, close) => {
						const presets =
							mode === "context-percent"
								? [0, 1, 3, 5, 10, 15, 25, 30, 50, 75, 100]
								: mode === "messages"
									? [1, 3, 5, 10, 15, 25, 50, 100, 250, 500, 1000]
									: [0, 1, 3, 5, 10, 15, 25, 1000, 2000, 4000, 8000, 16000, 32000];
						return this.#settingsMenu(
							"Limit Value",
							[
								{
									id: "custom",
									label: "Custom number…",
									currentValue: String(limit.value),
									description,
									submenu: (_v, finish) =>
										this.#validatedInput(
											"Limit Value",
											description,
											String(limit.value),
											value => {
												const parsed = this.#context.settings.preservation!.parseLimit(
													`${mode}:${value.trim()}`,
												);
												if (!parsed) throw new Error(description);
												save(parsed);
												finish(value);
												close(value);
											},
											() => finish(),
										),
								},
								...presets.map(value => ({
									id: String(value),
									label: String(value),
									currentValue: "",
									values: [String(value)],
								})),
							],
							(id, value) => {
								if (id !== "custom") {
									save({ mode, value: Number(value) });
									close(value);
								}
							},
							() => close(),
							() => this.#contentRowCount,
						);
					},
				});
			}
			rows.push({
				id: "effective",
				label: "Effective",
				currentValue: this.#limitLabel(raw),
				description: this.#limitSummary(raw),
			});
			rows.push(usage);
			return rows;
		};
		const menu: SettingsSubmenu = this.#settingsMenu(
			def.label,
			items(),
			() => {},
			() => done(this.#limitLabel(this.#context.settings.get(def.path))),
			() => this.#contentRowCount,
			() => {
				if (
					displayedValue !== this.#context.settings.get(def.path) ||
					displayedMaximum !== this.#context.maxContextTokens
				)
					menu.list.setItems(items());
				const summary = this.#context.getPreservationLimitUsage?.(def.path);
				usage.currentValue = summary ?? "Unavailable";
				usage.description =
					summary ??
					"Current selection counts appear once message retention has been calculated for this session.";
			},
		);
		return menu;
	}

	#capSummary(): string {
		const cap = this.#context.settings.get("compaction.keepUserMessagesFilterKeepCap");
		if (cap === "uncapped") return "No cap";
		const raw = this.#context.settings.get(
			cap === "keep-first" ? "compaction.keepFirstLimit" : "compaction.keepLastLimit",
		);
		const limit = this.#context.settings.preservation!.parseLimit(raw);
		return `${cap === "keep-first" ? "First" : "Recent"}: ${limit?.mode === "off" || limit?.mode === "all" ? "no cap" : this.#limitLabel(raw)}`;
	}

	#createCap(done: (value?: string) => void): SettingsSubmenu {
		let displayedSummary: string;
		const usage: SettingItem = { id: "usage", label: "Current selection", currentValue: "Unavailable" };
		const items = (): SettingItem[] => {
			displayedSummary = this.#capSummary();
			const cap = this.#context.settings.get("compaction.keepUserMessagesFilterKeepCap");
			const rows: SettingItem[] = [
				{
					id: "direction",
					label: "Selection order",
					currentValue: cap === "keep-last" ? "Newest first" : cap === "keep-first" ? "Oldest first" : "No cap",
					description:
						"Extra retention for filter Keep or manual Always messages. Newest/Oldest first uses the Recent/First limit value separately; overlap is kept once. No cap keeps all marked messages.",
					submenu: (_cv, close) =>
						this.#preservationSelectField(
							"Rule / Manual Keep Order",
							"Uses the First/Recent limit value as a separate Keep/Always allowance, not the window's remaining budget or an overall cap. Messages may also qualify through First/Recent or recent protection; duplicates are kept once.",
							[
								{ value: "keep-last", label: "Newest first", description: "Use Keep Recent Limit" },
								{ value: "keep-first", label: "Oldest first", description: "Use Keep First Limit" },
								{ value: "uncapped", label: "No cap", description: "Keep all Keep/Always messages" },
							],
							String(cap),
							value => {
								this.#save("compaction.keepUserMessagesFilterKeepCap", value);
								close(value);
								menu.list.setItems(items());
							},
							() => close(),
						),
				},
			];
			if (cap !== "uncapped") {
				const path = cap === "keep-first" ? "compaction.keepFirstLimit" : "compaction.keepLastLimit";
				const def = getSettingDef(this.#context.settings.entries, path)!;
				rows.push({
					id: "edge",
					label: cap === "keep-first" ? "Edit Keep First Limit…" : "Edit Keep Recent Limit…",
					currentValue: this.#limitLabel(this.#context.settings.get(path)),
					description:
						"Changes the ordinary Keep First/Recent Limit too. Off or All means no cap here; 0 tokens or 0% is a zero-token allowance. Assistant/tool exchanges stay together; each message counts toward the allowance.",
					submenu: (_cv, close) =>
						this.#createLimit(def, value => {
							close(value);
							menu.list.setItems(items());
						}),
				});
			}
			rows.push({
				id: "effective",
				label: "Effective",
				currentValue: this.#capSummary(),
				description:
					"This limits the Keep/Always selection, not the total retained context. A message beyond this cap can still qualify through First/Recent or recent protection. Overlapping selections keep one copy.",
			});
			rows.push(usage);
			return rows;
		};
		const menu: SettingsSubmenu = this.#settingsMenu(
			"Rule / Manual Keep Limit",
			items(),
			() => {},
			() => done(this.#capSummary()),
			() => this.#contentRowCount,
			() => {
				if (displayedSummary !== this.#capSummary()) menu.list.setItems(items());
				const summary = this.#context.getPreservationLimitUsage?.("compaction.keepUserMessagesFilterKeepCap");
				usage.currentValue = summary ?? "Unavailable";
				usage.description =
					summary ??
					"Current selection counts appear once message retention has been calculated for this session.";
			},
		);
		return menu;
	}

	#categorySummary(): string {
		let keep = 0,
			never = 0;
		const categories = this.#context.settings.preservation!.categories;
		for (const { path } of categories) {
			const value = this.#context.settings.get(path);
			if (value === "keep") keep++;
			if (value === "exclude") never++;
		}
		return keep + " Keep · " + never + " Never · " + (categories.length - keep - never) + " Auto";
	}

	#createCategories(done: (value?: string) => void): SettingsSubmenu {
		const items = (): PreservationSettingItem[] =>
			this.#context.settings.preservation!.categories.map(({ path, label, description }) => {
				const current = () => this.#context.settings.get(path) as PreservationAction;
				const save = (value: PreservationAction) => {
					if (current() !== value) this.#save(path, value);
					menu.list.setItems(items());
				};
				return {
					id: path,
					label,
					currentValue: this.#actionLabel(current()),
					description: description + " Auto defers. Keep > Never > Auto within categories; no category priority.",
					assignment: { kind: "action", get: current, set: save },
					submenu: (_cv, close) =>
						this.#actionSelector(
							current(),
							value => {
								save(value);
								close(this.#actionLabel(value));
							},
							() => close(),
						),
				};
			});
		const menu = this.#settingsMenu(
			"Category Rules — stored tags only",
			items(),
			() => {},
			() => done(this.#categorySummary()),
			() => this.#contentRowCount,
		);
		return menu;
	}

	#actionLabel(value: PreservationAction): string {
		return value === "exclude" ? "Never" : value === "keep" ? "Keep" : "Auto";
	}

	#actionSelector(
		currentValue: string,
		save: (value: PreservationAction) => void,
		onCancel: () => void,
	): PreservationActionField {
		return new PreservationActionField(
			{
				theme: formTheme,
				label: "Action",
				description:
					"Auto leaves the decision to other rules and First/Recent limits. Keep requests extra retention within Rule / Manual Keep Limit. Never rejects extra retention, not normal recent history. Manual choices and recent protection take precedence.",
				items: [
					{ value: "auto", label: "Auto", description: "No decision from this rule" },
					{ value: "keep", label: "Keep", description: "Request extra retention" },
					{ value: "exclude", label: "Never", description: "Reject extra retention" },
				],
				currentValue,
				selectTheme: getSelectListTheme(),
				onSubmit: value => save(value as PreservationAction),
				onCancel,
				hint: "Enter/Space select · y/* Keep · n Never · -/Backspace Auto · Esc back",
				requestRender: this.#context.requestRender,
			},
			() => {
				this.#submenuHint = "Enter/Space select · y/* Keep · n Never · -/Backspace Auto · Esc back";
			},
			() => this.#contentRowCount,
		);
	}

	#createRegexRules(done: (value?: string) => void): SettingsSubmenu {
		const refresh = () => menu.list.setItems(items());
		const items = (): SettingItem[] => [
			{
				id: "add",
				label: "Add regex…",
				currentValue: "",
				description: "RE2 condition. Auto starts disabled; ordinary and Final stages each use Keep > Never > Auto.",
				submenu: (_cv, close) =>
					this.#validatedInput(
						"New Regex Condition",
						"RE2 syntax; must compile before saving.",
						"",
						condition => {
							this.#context.settings.preservation!.validateRegexCondition(condition);
							const rules = this.#regexRules();
							if (Object.hasOwn(rules, condition)) throw new Error("A rule with this condition already exists");
							this.#save("compaction.keepUserMessagesRegexRules", {
								...rules,
								[condition]: { state: "auto", caseInsensitive: true, final: false },
							});
							close();
							refresh();
							menu.list.selectItem(`rule:${condition}`);
						},
						() => close(),
					),
			},
			...Object.entries(this.#regexRules()).map(([condition, rule]) => ({
				id: `rule:${condition}`,
				label: condition,
				currentValue: `${this.#actionLabel(rule.state)} ${rule.caseInsensitive ? "i" : ""}${rule.final ? " Final" : ""}`,
				submenu: (_cv: string, close: (value?: string) => void) =>
					this.#createRegexRuleEditor(condition, selectedCondition => {
						close();
						refresh();
						menu.list.selectItem(`rule:${selectedCondition}`);
					}),
			})),
		];
		const menu: SettingsSubmenu = this.#settingsMenu(
			"Custom Regex Rules",
			items(),
			() => {},
			() => done(`${Object.keys(this.#regexRules()).length} rules`),
			() => this.#contentRowCount,
		);
		return menu;
	}

	#createRegexRuleEditor(initialCondition: string, done: (condition: string) => void): SettingsSubmenu {
		let condition = initialCondition;
		const current = () => this.#regexRules()[condition]!;
		const save = (rule: PreservationRegexRule) => {
			const previous = current();
			if (
				rule.state === previous.state &&
				rule.caseInsensitive === previous.caseInsensitive &&
				(rule.final ?? false) === (previous.final ?? false)
			)
				return;
			this.#context.settings.preservation!.validateRegexCondition(condition, rule.caseInsensitive);
			this.#save("compaction.keepUserMessagesRegexRules", {
				...this.#regexRules(),
				[condition]: rule,
			});
			menu.list.setItems(items());
		};
		const items = (): PreservationSettingItem[] => {
			const rule = current();
			return [
				{
					id: "condition",
					label: "Condition",
					currentValue: condition,
					submenu: (_cv, close) =>
						this.#validatedInput(
							"Regex Condition",
							"RE2 syntax. Invalid expressions are never saved.",
							condition,
							value => {
								this.#context.settings.preservation!.validateRegexCondition(value, current().caseInsensitive);
								const rules = this.#regexRules();
								if (value !== condition && Object.hasOwn(rules, value))
									throw new Error("A rule with this condition already exists");
								const renamed = Object.fromEntries(
									Object.entries(rules).map(([key, entry]) => [key === condition ? value : key, entry]),
								);
								this.#save("compaction.keepUserMessagesRegexRules", renamed);
								condition = value;
								close(value);
								menu.list.setItems(items());
							},
							() => close(),
						),
				},
				{
					id: "state",
					label: "Action",
					assignment: {
						kind: "action",
						get: () => current().state,
						set: value => save({ ...current(), state: value }),
					},
					currentValue: this.#actionLabel(rule.state),
					description:
						"Auto disables this rule. Keep wins same-stage conflicts; Never leaves ordinary vanilla treatment unchanged.",
					submenu: (_cv, close) =>
						this.#actionSelector(
							current().state,
							value => {
								save({ ...current(), state: value });
								close(this.#actionLabel(value));
							},
							() => close(),
						),
				},
				{
					id: "case",
					label: "Case insensitive",
					assignment: {
						kind: "boolean",
						get: () => current().caseInsensitive,
						set: value => save({ ...current(), caseInsensitive: value }),
					},
					currentValue: String(rule.caseInsensitive),
					values: ["true", "false"],
				},
				{
					id: "final",
					label: "Final",
					assignment: {
						kind: "boolean",
						get: () => current().final ?? false,
						set: value => save({ ...current(), final: value }),
					},
					currentValue: String(rule.final ?? false),
					values: ["false", "true"],
					description: "Final runs after stored classifier policy, before manual state. Auto remains neutral.",
				},
				{
					id: "delete",
					label: "Delete rule…",
					currentValue: "",
					submenu: (_cv, close) =>
						this.#preservationSelectField(
							"Delete this regex rule?",
							condition,
							[
								{ value: "no", label: "No — keep rule" },
								{ value: "yes", label: "Yes — delete rule" },
							],
							"no",
							value => {
								if (value === "yes") {
									const rules = { ...this.#regexRules() };
									delete rules[condition];
									this.#save("compaction.keepUserMessagesRegexRules", rules);
									close();
									done(condition);
								} else close();
							},
							() => close(),
						),
				},
			];
		};
		const menu: SettingsSubmenu = this.#settingsMenu(
			"Regex Rule",
			items(),
			(id, value) => {
				if (id === "case") save({ ...current(), caseInsensitive: value === "true" });
				if (id === "final") save({ ...current(), final: value === "true" });
			},
			() => done(condition),
			() => this.#contentRowCount,
		);
		return menu;
	}

	#createModelSelector(def: SettingDef, done: (value?: string) => void): ModelSelectorSubmenu {
		if (!this.#context.modelSource) throw new Error("Classifier model selection requires a model browser source");
		return new ModelSelectorSubmenu(
			this.#context.modelSource,
			this.#context.modelRegistry,
			String(this.#context.settings.get(def.path) ?? ""),
			value => {
				this.#save(def.path, value || undefined);
				done(value || "Automatic (@tiny)");
			},
			() => done(),
			() => this.#contentRowCount,
		);
	}

	/**
	 * Get the current value for a setting.
	 */
	#getCurrentValue(def: SettingDef): unknown {
		return this.#context.settings.get(def.path);
	}

	#isChanged(def: SettingDef, currentValue: unknown): boolean {
		const defaultValue: unknown = def.defaultValue;
		if (Array.isArray(currentValue) && Array.isArray(defaultValue)) {
			return (
				currentValue.length !== defaultValue.length ||
				currentValue.some((entry, index) => entry !== defaultValue[index])
			);
		}
		return !Object.is(currentValue, defaultValue);
	}

	#getSubmenuCurrentValue(path: string, value: unknown): string {
		const rawValue = String(value ?? "");
		if (path === "compaction.thresholdPercent" && (rawValue === "-1" || rawValue === "")) {
			return "default";
		}
		if (path === "compaction.thresholdTokens" && (rawValue === "-1" || rawValue === "")) {
			return "default";
		}
		return rawValue;
	}

	/**
	 * Create a submenu for a submenu-type setting.
	 */
	#createSubmenu(
		def: SettingDef & { type: "submenu" },
		currentValue: string,
		done: (value?: string) => void,
	): Component {
		let options = def.options;

		// Special case: inject runtime options for thinking level
		if (def.path === "defaultThinkingLevel") {
			// Prepend `auto`; the rest are the model's runtime-supported efforts.
			const levels: ConfiguredThinkingLevel[] = [AUTO_THINKING, ...this.#context.availableThinkingLevels];
			options = levels.map(level => {
				const baseOpt = options.find(o => o.value === level);
				return baseOpt || { value: level, label: level };
			});
		} else if (def.path === "theme.dark" || def.path === "theme.light") {
			options = this.#context.availableThemes.map(t => ({ value: t, label: t }));
		} else if (def.path === "composer.shape") {
			options = getComposerShapeOptions();
		}
		// Preview handlers
		let onPreview: ((value: string) => void | Promise<void>) | undefined;
		let onPreviewCancel: (() => void) | undefined;
		let footer: Component | undefined;

		const activeThemeBeforePreview = getCurrentThemeName() ?? currentValue;
		if (def.path === "theme.dark" || def.path === "theme.light") {
			onPreview = value => {
				return this.#callbacks.onThemePreview?.(value);
			};
			onPreviewCancel = () => {
				this.#callbacks.onThemePreview?.(activeThemeBeforePreview);
			};
		} else if (def.path === "statusLine.preset") {
			onPreview = value => {
				const presetDef = getPreset(
					value as "default" | "minimal" | "compact" | "full" | "nerd" | "ascii" | "custom",
				);
				this.#callbacks.onStatusLinePreview?.({
					preset: value as StatusLinePreset,
					leftSegments: presetDef.leftSegments,
					rightSegments: presetDef.rightSegments,
					separator: presetDef.separator,
				});
			};
			onPreviewCancel = () => {
				const currentPreset = this.#context.settings.get("statusLine.preset") as StatusLinePreset;
				const presetDef = getPreset(currentPreset);
				this.#callbacks.onStatusLinePreview?.({
					preset: currentPreset,
					leftSegments: presetDef.leftSegments,
					rightSegments: presetDef.rightSegments,
					separator: presetDef.separator,
				});
			};
		} else if (def.path === "statusLine.separator") {
			onPreview = value => {
				this.#callbacks.onStatusLinePreview?.({ separator: value as StatusLineSeparatorStyle });
			};
			onPreviewCancel = () => {
				const separator = this.#context.settings.get("statusLine.separator") as StatusLineSeparatorStyle;
				this.#callbacks.onStatusLinePreview?.({ separator });
			};
		} else if (def.path === "statusLine.contextLine") {
			onPreview = value => {
				this.#callbacks.onStatusLinePreview?.({ contextLine: value as ContextLineMode });
			};
			onPreviewCancel = () => {
				this.#callbacks.onStatusLinePreview?.({
					contextLine: this.#context.settings.get("statusLine.contextLine") as ContextLineMode,
				});
			};
		} else if (def.path === "snapcompact.shape") {
			const shapePreview = new SnapcompactShapePreview(currentValue, {
				model: this.#context.model,
				imageBudget: this.#context.imageBudget,
				requestRender: this.#context.requestRender,
			});
			onPreview = value => shapePreview.setValue(value);
			footer = shapePreview;
		} else if (def.path === "composer.shape") {
			const shapePreview = new ComposerShapePreview(String(currentValue ?? "band"), {
				requestRender: this.#context.requestRender,
				status: this.#context.composerPreviewStatus,
			});
			onPreview = value => shapePreview.setValue(value);
			footer = shapePreview;
		}
		// Provide status line preview for theme selection
		const isThemeSetting = def.path === "theme.dark" || def.path === "theme.light";
		const getPreview = isThemeSetting ? this.#callbacks.getStatusLinePreview : undefined;

		return createSettingsSelectField(
			def.label,
			def.description,
			options,
			currentValue,
			value => {
				this.#setSettingValue(def.path, value);
				this.#callbacks.onChange(def.path, value);
				done(value);
			},
			() => {
				onPreviewCancel?.();
				done();
			},
			onPreview,
			getPreview,
			footer,
			this.#context.requestRender,
		);
	}

	/**
	 * Create a text input submenu for a plain string setting.
	 */
	#createTextInput(
		def: SettingDef & { type: "text" },
		_currentValue: string,
		done: (value?: string) => void,
	): Component {
		this.#textInputActive = true;
		const wrappedDone = (value?: string) => {
			this.#textInputActive = false;
			done(value);
		};
		return createSettingsTextField(
			def.label,
			def.description,
			this.#formatTextInputEditValue(def.path, this.#context.settings.get(def.path)),
			def.secret,
			value => {
				// Empty string clears the setting; undefined-typed string settings
				// store "" which the browser.ts expandPath ignores (no-op fallback).
				this.#setSettingValue(def.path, value);
				this.#callbacks.onChange(def.path, this.#context.settings.get(def.path));
				wrappedDone(this.#formatTextInputValue(def, this.#context.settings.get(def.path)));
			},
			() => wrappedDone(),
			this.#context.requestRender,
		);
	}

	#createProviderLimitsInput(done: (value?: string) => void): Container {
		return new ProviderLimitsSubmenu(
			this.#context.settings,
			this.#context.providers,
			value => {
				this.#callbacks.onChange("providers.maxInFlightRequests", value);
				done(this.#formatProviderLimitsValue(value));
			},
			() => done(),
			this.#context.requestRender,
		);
	}

	#formatProviderLimitsValue(value: unknown): string {
		const limits = this.#context.settings.normalizeProviderLimits(value);
		const entries = Object.entries(limits).sort(([a], [b]) => a.localeCompare(b));
		if (entries.length === 0) return "Unlimited";
		return entries.map(([provider, limit]) => `${provider}: ${limit}`).join(", ");
	}

	#getMultiSelectOptions(def: SettingDef & { type: "multiselect" }) {
		if (def.path !== "providers.webSearchOrder") return def.options;
		const excluded: unknown = this.#context.settings.get("providers.webSearchExclude");
		if (!Array.isArray(excluded)) return def.options;
		return def.options.filter(option => !excluded.includes(option.value));
	}

	#createMultiSelect(def: SettingDef & { type: "multiselect" }, done: (value?: string) => void): Container {
		const options = this.#getMultiSelectOptions(def);
		const current: unknown = this.#context.settings.get(def.path);
		const initial = Array.isArray(current)
			? current.filter((entry): entry is string => typeof entry === "string")
			: [];
		return new MultiSelectSubmenu(
			def.label,
			def.description,
			options,
			initial,
			def.ordered,
			value => {
				this.#context.settings.set(def.path, value);
				this.#callbacks.onChange(def.path, value);
			},
			() => done(this.#formatMultiSelectValue(def, this.#context.settings.get(def.path))),
		);
	}

	#formatMultiSelectValue(def: SettingDef & { type: "multiselect" }, value: unknown): string {
		const options = this.#getMultiSelectOptions(def);
		const labels = Array.isArray(value)
			? value.flatMap(entry => {
					if (typeof entry !== "string") return [];
					const option = options.find(candidate => candidate.value === entry);
					return option ? [option.label] : [];
				})
			: [];
		if (labels.length === 0) return def.ordered ? "default" : "none";
		return def.ordered ? labels.join(" → ") : labels.join(", ");
	}

	#formatTextInputValue(def: SettingDef & { type: "text" }, value: unknown): string {
		if (def.secret) return value ? "••••••••" : "";
		return this.#formatTextInputEditValue(def.path, value);
	}

	#formatTextInputEditValue(_path: string, value: unknown): string {
		if (value === undefined || value === null) return "";
		if (typeof value === "object") return JSON.stringify(value);
		return String(value);
	}

	/**
	 * Set a setting value, handling type conversion.
	 */
	#setSettingValue(path: string, value: string): void {
		const currentValue = this.#context.settings.get(path);
		const schemaType = getSettingDef(this.#context.settings.entries, path)?.schemaType;
		if (path === "compaction.thresholdPercent" && value === "default") {
			this.#context.settings.set(path, -1);
		} else if (path === "compaction.thresholdTokens" && value === "default") {
			this.#context.settings.set(path, -1);
		} else if (schemaType === "record") {
			let parsed: unknown;
			try {
				parsed = JSON.parse(value || "{}");
			} catch {
				throw new Error(`Invalid record JSON for ${path}`);
			}
			if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
				throw new Error(`Invalid record JSON for ${path}`);
			}
			if (path === "providers.maxInFlightRequests") {
				parsed = this.#context.settings.validateProviderLimits(parsed);
			}
			this.#context.settings.set(path, parsed);
		} else if (typeof currentValue === "number") {
			this.#context.settings.set(path, Number(value));
		} else if (typeof currentValue === "boolean") {
			this.#context.settings.set(path, value === "true");
		} else {
			this.#context.settings.set(path, value);
		}
	}

	/**
	 * Show a settings tab using definitions.
	 */
	#showSettingsTab(tabId: SettingTab): void {
		const defs = getSettingsForTab(this.#context.settings.entries, tabId);

		const items = this.#buildItemsForDefs(defs);
		// Mirror SettingsList's section detection (leading ungrouped items form
		// an implicit section) so the footer hint only advertises PgUp/PgDn
		// when the jump actually changes sections.
		const sectionCount = items.filter(item => item.heading).length + (items.length > 0 && !items[0].heading ? 1 : 0);
		this.#hasSectionJump = sectionCount >= 2;

		this.#currentList = new SettingsList(
			items,
			10,
			getSettingsListTheme(),
			(id, newValue) => {
				const def = defs.find(d => d.path === id);
				if (!def) return;

				const path = def.path;

				if (def.type === "boolean") {
					const boolValue = newValue === "true";
					this.#context.settings.set(path, boolValue);
					this.#callbacks.onChange(path, boolValue);

					if (tabId === "appearance") {
						this.#triggerStatusLinePreview();
					}
				} else if (def.type === "enum") {
					this.#context.settings.set(path, newValue);
					this.#callbacks.onChange(path, newValue);
				}
				// Submenu/text types already persisted the value inside their own
				// done callbacks before SettingsList re-dispatches here. Re-run the
				// definition-to-item mapping so condition-gated settings (e.g. the
				// Hindsight cluster guarded by memory.backend) appear/disappear
				// immediately instead of waiting for the next tab switch.
				this.#refreshCurrentTabItems(defs);
			},
			() => this.#callbacks.onCancel(),
			// The selector owns type-to-search and the footer hint; pin the
			// split sidebar width so the divider never jumps between tabs.
			{ typeToSearch: false, hint: "", sidebarWidth: this.#sidebarWidth },
		);
	}

	/**
	 * Map a definition list to UI items, dropping any whose condition is false.
	 * Inserts a heading row whenever the (group-sorted) definition list crosses
	 * into a new group; groups whose items are all condition-hidden emit none.
	 */
	#buildItemsForDefs(defs: SettingDef[]): SettingItem[] {
		const items: SettingItem[] = [];
		let lastGroup: string | undefined;
		for (const def of defs) {
			const item = this.#defToItem(def);
			if (!item) continue;
			if (def.group && def.group !== lastGroup) {
				items.push({ id: `__heading:${def.group}`, label: def.group, currentValue: "", heading: true });
				lastGroup = def.group;
			}
			items.push(item);
		}
		return items;
	}

	/** Re-evaluate condition gates against the current settings and refresh the active list. */
	#refreshCurrentTabItems(defs: SettingDef[]): void {
		if (this.#currentTabId === "plugins" || !this.#currentList) return;
		this.#currentList.setItems(this.#buildItemsForDefs(defs));
	}

	/**
	 * Get the status line preview string.
	 */
	#getStatusPreviewString(): string {
		if (this.#callbacks.getStatusLinePreview) {
			return this.#callbacks.getStatusLinePreview();
		}
		return theme.fg("dim", "(preview not available)");
	}

	/**
	 * Trigger status line preview with current settings.
	 */
	#triggerStatusLinePreview(): void {
		const statusLineSettings: StatusLinePreviewSettings = {
			preset: this.#context.settings.get("statusLine.preset") as StatusLinePreset,
			leftSegments: this.#context.settings.get("statusLine.leftSegments") as StatusLineSegmentId[],
			rightSegments: this.#context.settings.get("statusLine.rightSegments") as StatusLineSegmentId[],
			separator: this.#context.settings.get("statusLine.separator") as StatusLineSeparatorStyle,
			sessionAccent: this.#context.settings.get("statusLine.sessionAccent") as boolean,
			transparent: this.#context.settings.get("statusLine.transparent") as boolean,
		};
		this.#callbacks.onStatusLinePreview?.(statusLineSettings);
	}

	#showPluginsTab(): void {
		this.#pluginComponent = new PluginSettingsComponent(this.#context.plugins, {
			onClose: () => this.#callbacks.onCancel(),
			onPluginChanged: () => this.#callbacks.onPluginsChanged?.(),
			requestRender: this.#context.requestRender,
		});
	}

	handleInput(data: string): void {
		// SGR mouse reports (the fullscreen overlay enables tracking).
		if (data.startsWith("\x1b[<")) {
			this.#handleMouse(data);
			return;
		}

		// Text-input submenus take every byte: arrow keys must reach the
		// cursor and Tab must not switch tabs.
		if (this.#textInputActive) {
			(this.#searchList ?? this.#currentList)?.handleInput(data);
			return;
		}

		const activeList = this.#searchList ?? this.#currentList;

		// An open submenu owns input entirely — Tab/arrows/typing belong to it.
		if (activeList?.hasOpenSubmenu()) {
			activeList.handleInput(data);
			return;
		}

		if (this.#searchList) {
			this.#handleSearchModeInput(data, this.#searchList);
			return;
		}

		if (!activeList?.sectionFocused && assignPreservationItem(activeList?.getSelectedItem(), data)) return;
		if (this.#currentTabId !== "plugins" && data === "/") {
			this.#startSearch("");
			return;
		}

		// Tab toggles keyboard focus between section headings and setting rows
		// (fast section hopping); tabs without sections keep Tab switching tabs.
		if (matchesKey(data, "tab") || matchesKey(data, "shift+tab")) {
			if (this.#currentList?.hasSectionFocusTargets()) {
				this.#currentList.toggleSectionFocus();
				return;
			}
			this.#tabBar.handleInput(data);
			return;
		}
		if (matchesKey(data, "left") || matchesKey(data, "right")) {
			this.#tabBar.handleInput(data);
			return;
		}

		// Printable characters start a search across every settings tab. The
		// plugins tab keeps its own local filtering instead.
		if (this.#currentTabId !== "plugins") {
			const printable = extractPrintableText(data);
			if (printable !== undefined && printable.trim().length > 0) {
				this.#startSearch(printable);
				return;
			}
		}

		if (this.#currentList) {
			this.#currentList.handleInput(data);
		} else if (this.#pluginComponent) {
			this.#pluginComponent.handleInput(data);
		}
	}

	#handleSearchModeInput(data: string, list: SettingsList): void {
		const kb = getKeybindings();
		if (kb.matches(data, "tui.select.cancel")) {
			// Exit search, landing on the tab of the selected result.
			this.#endSearch(true);
			return;
		}
		if (matchesKey(data, "tab") || matchesKey(data, "shift+tab")) {
			// Jump between tabs that have matches (muted tabs are skipped).
			this.#tabBar.handleInput(data);
			return;
		}
		// Selection, paging, and activation stay with the result list.
		if (
			kb.matches(data, "tui.select.up") ||
			kb.matches(data, "tui.select.down") ||
			kb.matches(data, "tui.select.pageUp") ||
			kb.matches(data, "tui.select.pageDown") ||
			kb.matches(data, "tui.select.confirm") ||
			data === "\n"
		) {
			list.handleInput(data);
			return;
		}
		// Everything else edits the query like a regular single-line editor:
		// cursor movement, word ops, kill ring, undo, paste.
		this.#searchInput.handleInput(data);
		const value = this.#searchInput.getValue();
		if (value !== this.#searchQuery) this.#setSearchQuery(value);
	}
}
