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
	replaceTabs,
	routeSelectListMouse,
	routeSgrMouseInput,
	type SelectItem,
	SelectList,
	type SettingItem,
	SettingsList,
	type SgrMouseEvent,
	Spacer,
	type Tab,
	TabBar,
	Text,
	truncateToWidth,
	visibleWidth,
} from "@oh-my-pi/pi-tui";
import type { ShapeTarget } from "@oh-my-pi/snapcompact";
import type { ModelRegistry } from "../../config/model-registry";
import { formatModelRoleAlias, getKnownRoleIds, getRoleInfo } from "../../config/model-roles";
import {
	parsePreservationLimit,
	serializePreservationLimit,
	PRESERVED_USER_MESSAGE_CATEGORIES,
	PRESERVED_USER_MESSAGE_CATEGORY_LABELS,
	PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS,
	type PreservationAction,
	type PreservationLimit,
	type PreservedUserMessageRegexRule,
	validatePreservedUserMessageRegexCondition,
} from "../../session/preserved-message-settings";
import { buildBrowserItems, ModelBrowser, resolveRoleAssignments, sortModelItems } from "./model-browser";
import {
	getDefault,
	getType,
	normalizeProviderMaxInFlightRequests,
	type SettingPath,
	settings,
	validateProviderMaxInFlightRequests,
} from "../../config/settings";
import type {
	ContextLineMode,
	SettingTab,
	StatusLinePreset,
	StatusLineSegmentId,
	StatusLineSeparatorStyle,
} from "../../config/settings-schema";
import { getUi, SETTING_TABS, TAB_METADATA } from "../../config/settings-schema";
import { getCurrentThemeName, getSelectListTheme, getSettingsListTheme, theme } from "../../modes/theme/theme";
import { AUTO_THINKING, type ConfiguredThinkingLevel } from "../../thinking";
import { getTabBarTheme } from "../shared";
import { type ComposerPreviewStatusSource, ComposerShapePreview } from "./composer-shape-preview";
import { getComposerShapeOptions } from "./composer-shape-registry";
import { bottomBorder, divider, row, topBorder } from "./overlay-box";
import { handleInputOrEscape, PluginSettingsComponent } from "./plugin-settings";
import { getSettingDef, getSettingsForTab, type SettingDef } from "./settings-defs";
import { SnapcompactShapePreview } from "./snapcompact-shape-preview";
import { getPreset } from "./status-line/presets";

/**
 * A submenu component for selecting from a list of options.
 */
/**
 * Submenu component for free-text string settings.
 * Mirrors the ConfigInputSubmenu pattern from plugin-settings.ts.
 */
class TextInputSubmenu extends Container {
	#input: Input;
	#error: Text;

	constructor(
		label: string,
		description: string,
		currentValue: string,
		secret: boolean,
		private readonly onSubmit: (value: string) => void,
		private readonly onCancel: () => void,
		private readonly getHeight?: () => number,
		allowUnset = true,
	) {
		super();

		this.addChild(new Text(theme.bold(theme.fg("accent", label)), 0, 0));
		if (description) {
			this.addChild(new Spacer(1));
			this.addChild(new Text(theme.fg("muted", description), 0, 0));
		}
		this.addChild(new Spacer(1));

		this.#input = new Input();
		this.#input.mask = secret;
		if (currentValue) {
			this.#input.setValue(currentValue);
		}
		this.#error = new Text("", 0, 0);
		this.#input.onSubmit = value => {
			try {
				this.onSubmit(value); // empty string clears the setting
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				this.#error.setText(theme.fg("error", truncateToWidth(replaceTabs(message).replace(/[\r\n]+/g, " "), 100)));
			}
		};
		this.addChild(this.#input);
		this.addChild(new Spacer(1));
		this.addChild(this.#error);
		this.addChild(
			new Text(
				theme.fg(
					"dim",
					allowUnset
						? "  Enter to save · Esc to cancel · Clear field to unset"
						: "  Enter to save · Esc to cancel",
				),
				0,
				0,
			),
		);
	}

	handleInput(data: string): void {
		handleInputOrEscape(data, this.#input, this.onCancel);
	}

	override render(width: number): readonly string[] {
		if (!this.getHeight) return super.render(width);
		const title = truncateToWidth(this.children[0]?.render(width).join(" ") ?? "", width);
		return [
			title,
			...this.#input.render(width),
			...this.#error.render(width),
			...this.children
				.filter(child => child instanceof Text && child !== this.children[0] && child !== this.#error)
				.flatMap(child => child.render(width)),
		].slice(0, Math.max(1, this.getHeight()));
	}
}

/** Native nested settings retain upstream selection, keyboard and mouse behavior. */
class SettingsSubmenu extends Container {
	readonly list: SettingsList;
	constructor(
		private readonly title: string,
		items: SettingItem[],
		onChange: (id: string, value: string) => void,
		onCancel: () => void,
		private readonly getHeight: () => number,
		private readonly onRender?: () => void,
	) {
		super();
		this.list = new SettingsList(items, 3, getSettingsListTheme(), onChange, onCancel, {
			layout: "flat",
			typeToSearch: false,
			hint: "",
		});
		this.addChild(this.list);
	}
	override render(width: number): readonly string[] {
		if (this.list.hasOpenSubmenu()) return this.list.render(width);
		this.onRender?.();
		this.list.setMaxVisible(Math.max(1, this.getHeight() - 5));
		return [truncateToWidth(theme.bold(theme.fg("accent", this.title)), width), ...this.list.render(width)];
	}
	handleInput(data: string): void {
		this.list.handleInput(data);
	}
	routeMouse(event: SgrMouseEvent, line: number, col: number): void {
		if (this.list.hasOpenSubmenu()) {
			this.list.routeSubmenuMouse(event, line, col);
			return;
		}
		line--;
		if (event.wheel !== null) {
			this.list.handleWheel(event.wheel);
			return;
		}
		const id = this.list.hitTest(line, col);
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

class ModelSelectorSubmenu extends Container {
	#active: Component;
	constructor(
		registry: ModelRegistry | undefined,
		current: string,
		onSelect: (selector: string) => void,
		onCancel: () => void,
		private readonly getHeight: () => number,
	) {
		super();
		const roles = () =>
			new SelectSubmenu(
				"Classifier Model",
				"Automatic uses @tiny; no active-model fallback.",
				[
					{ value: "", label: "Automatic (@tiny)" },
					...getKnownRoleIds(settings).map(role => ({
						value: formatModelRoleAlias(role),
						label: formatModelRoleAlias(role),
						description: getRoleInfo(role, settings).name,
					})),
					{ value: "__browse", label: "Browse models…" },
				],
				current,
				value => {
					if (value !== "__browse") {
						onSelect(value);
						return;
					}
					const browser = new ModelBrowser(settings, {
						emptyText: () => "No models available — configure provider credentials.",
					});
					const available = registry?.getAvailable() ?? [];
					const assignments = resolveRoleAssignments(settings, registry?.getAll() ?? [], available);
					const storage = settings.getStorage();
					const mruOrder = storage?.getModelUsageOrder() ?? [];
					const items = buildBrowserItems(available);
					sortModelItems(items, { roles: assignments, mruOrder });
					browser.setItems(items);
					browser.setRoles(assignments);
					browser.setMruOrder(mruOrder);
					browser.setPerfStats(storage?.getModelPerf() ?? new Map());
					browser.onActivate = item => onSelect(item.selector);
					browser.onCancel = () => {
						this.#active = roles();
					};
					this.#active = browser;
				},
				onCancel,
				undefined,
				undefined,
				undefined,
				getHeight,
			);
		this.#active = roles();
	}
	override render(width: number): readonly string[] {
		if (this.#active instanceof ModelBrowser) this.#active.setMaxVisible(Math.max(1, this.getHeight() - 5));
		return this.#active.render(width);
	}
	handleInput(data: string): void {
		this.#active.handleInput?.(data);
	}
	routeMouse(event: SgrMouseEvent, line: number, col: number): void {
		if (this.#active instanceof ModelBrowser) this.#active.routeMouse(event, line);
		else if (this.#active instanceof SelectSubmenu) this.#active.routeMouse(event, line, col);
	}
}

class SelectSubmenu extends Container {
	#selectList: SelectList;
	#previewText: Text | null = null;
	#previewUpdateRequestId: number = 0;
	#selectListLineOffset = 0;

	constructor(
		title: string,
		description: string,
		options: ReadonlyArray<SelectItem>,
		currentValue: string,
		onSelect: (value: string) => void,
		onCancel: () => void,
		onSelectionChange?: (value: string) => void | Promise<void>,
		private readonly getPreview?: () => string,
		footer?: Component,
		private readonly getHeight?: () => number,
	) {
		super();

		// Title
		this.addChild(new Text(theme.bold(theme.fg("accent", title)), 0, 0));

		// Description
		if (description) {
			this.addChild(new Spacer(1));
			this.addChild(new Text(theme.fg("muted", description), 0, 0));
		}

		// Preview (if provided)
		if (getPreview) {
			this.addChild(new Spacer(1));
			this.addChild(new Text(theme.fg("muted", "Preview:"), 0, 0));
			this.#previewText = new Text(getPreview(), 0, 0);
			this.addChild(this.#previewText);
		}

		// Spacer
		this.addChild(new Spacer(1));

		// Select list
		this.#selectList = new SelectList(options, Math.min(options.length, 10), getSelectListTheme());

		// Pre-select current value
		const currentIndex = options.findIndex(o => o.value === currentValue);
		if (currentIndex !== -1) {
			this.#selectList.setSelectedIndex(currentIndex);
		}

		this.#selectList.onSelect = item => {
			onSelect(item.value);
		};

		this.#selectList.onCancel = onCancel;

		if (onSelectionChange) {
			this.#selectList.onSelectionChange = item => {
				const requestId = ++this.#previewUpdateRequestId;
				const result = onSelectionChange(item.value);
				if (result && typeof (result as Promise<void>).then === "function") {
					void (result as Promise<void>).finally(() => {
						if (requestId === this.#previewUpdateRequestId) {
							this.#updatePreview();
						}
					});
					return;
				}
				if (requestId === this.#previewUpdateRequestId) {
					this.#updatePreview();
				}
			};
		}

		this.addChild(this.#selectList);

		// Hint
		this.addChild(new Spacer(1));
		this.addChild(new Text(theme.fg("dim", "  Enter to select · Esc to go back"), 0, 0));

		// Footer (e.g. the snapcompact shape preview) below the interactive rows,
		// so the list never shifts while browsing.
		if (footer) {
			this.addChild(new Spacer(1));
			this.addChild(footer);
		}
	}

	#updatePreview(): void {
		if (this.#previewText && this.getPreview) {
			this.#previewText.setText(this.getPreview());
		}
	}

	/**
	 * Concatenate children like Container.render, recording where the select
	 * list lands so routed mouse events can be hit-tested against it.
	 */
	override render(width: number): readonly string[] {
		const lines: string[] = [];
		if (this.getHeight) {
			const height = Math.max(1, this.getHeight());
			for (const child of this.children) {
				if (child === this.#selectList) break;
				lines.push(...child.render(Math.max(1, width)));
			}
			if (lines.length > height - 2) lines.length = Math.max(1, height - 2);
			this.#selectList.setMaxVisible(Math.max(1, height - lines.length));
			this.#selectListLineOffset = lines.length;
			lines.push(...this.#selectList.render(width));
			return lines;
		}
		for (const child of this.children) {
			const childLines = child.render(Math.max(1, width));
			if (child === this.#selectList) {
				this.#selectListLineOffset = lines.length;
			}
			lines.push(...childLines);
		}
		return lines;
	}

	/** Mouse routed from the host: wheel steps, hover lights, click confirms. */
	routeMouse(event: SgrMouseEvent, line: number, _col: number): void {
		routeSelectListMouse(this.#selectList, event, line - this.#selectListLineOffset);
	}

	handleInput(data: string): void {
		this.#selectList.handleInput(data);
	}
}

/**
 * Submenu for array-of-enum settings: every option is a toggle row. Enter or
 * Space flips membership; ordered lists render 1-based positions and reorder
 * the highlighted member with ←/→. Changes apply live; Esc goes back.
 */
class MultiSelectSubmenu extends Container {
	#selectList!: SelectList;
	#value: string[];
	#cursor = 0;
	#selectListLineOffset = 0;
	#pressedItemId: string | undefined;
	#dropItemId: string | undefined;
	constructor(
		private readonly title: string,
		private readonly description: string,
		private readonly options: ReadonlyArray<SelectItem>,
		initial: readonly string[],
		private readonly ordered: boolean,
		private readonly onApply: (value: string[]) => void,
		private readonly onClose: () => void,
	) {
		super();
		// Drop stale ids (renamed/removed providers) so positions stay contiguous.
		this.#value = initial.filter(id => options.some(option => option.value === id));
		this.#rebuild();
	}

	#rebuild(): void {
		this.clear();
		this.addChild(new Text(theme.bold(theme.fg("accent", this.title)), 0, 0));
		if (this.description) {
			this.addChild(new Spacer(1));
			this.addChild(new Text(theme.fg("muted", this.description), 0, 0));
		}
		this.addChild(new Spacer(1));

		const items = this.options.map((option): SelectItem => {
			const position = this.#value.indexOf(option.value);
			const mark =
				position === -1
					? theme.fg("dim", this.ordered ? " · " : " ○ ")
					: this.ordered
						? theme.fg("accent", `${String(position + 1).padStart(2)}.`)
						: theme.fg("accent", " ● ");
			return { value: option.value, label: `${mark} ${option.label}`, description: option.description };
		});
		this.#selectList = new SelectList(items, Math.min(items.length, 12), getSelectListTheme());
		this.#selectList.setSelectedIndex(this.#cursor);
		this.#selectList.onSelect = item => this.#toggle(item.value);
		this.#selectList.onSelectionChange = item => {
			this.#cursor = this.options.findIndex(option => option.value === item.value);
		};
		this.#selectList.onCancel = this.onClose;
		this.addChild(this.#selectList);

		this.addChild(new Spacer(1));
		const hint = this.ordered
			? "  Click to toggle · drag selected items to reorder · ←/→ move · 1-9 place · Esc to go back"
			: "  Click/Enter/Space to toggle · Esc to go back";
		this.addChild(new Text(theme.fg("dim", hint), 0, 0));
	}

	#apply(next: string[]): void {
		this.#value = next;
		this.onApply([...next]);
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

	/** Concatenate children, recording the select list's line offset for mouse routing. */
	override render(width: number): readonly string[] {
		const lines: string[] = [];
		for (const child of this.children) {
			const childLines = child.render(Math.max(1, width));
			if (child === this.#selectList) {
				this.#selectListLineOffset = lines.length;
			}
			lines.push(...childLines);
		}
		return lines;
	}

	routeMouse(event: SgrMouseEvent, line: number, _col: number): void {
		const itemIndex = this.#selectList.hitTest(line - this.#selectListLineOffset);
		if (event.wheel !== null) {
			routeSelectListMouse(this.#selectList, event, line - this.#selectListLineOffset);
			return;
		}
		if (event.motion) {
			this.#selectList.setHoverIndex(itemIndex ?? null);
			const target = itemIndex === undefined ? undefined : this.options[itemIndex]?.value;
			if (
				this.ordered &&
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
			const item = this.options[itemIndex];
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
		if (this.ordered && dropItemId !== undefined && dropItemId !== pressedItemId) {
			this.#moveBefore(pressedItemId, dropItemId);
			return;
		}
		this.#toggle(pressedItemId);
	}

	handleInput(data: string): void {
		const current = this.options[this.#cursor]?.value;
		if (data === " " && current !== undefined) {
			this.#toggle(current);
			return;
		}
		if (this.ordered && current !== undefined && (data === "\x1b[D" || data === "\x1b[C")) {
			this.#move(current, data === "\x1b[D" ? -1 : 1);
			return;
		}
		if (this.ordered && current !== undefined && data.length === 1 && data >= "1" && data <= "9") {
			this.#placeAt(current, Number(data));
			return;
		}
		this.#selectList.handleInput(data);
	}
}

class ProviderLimitsSubmenu extends Container {
	#selectList: SelectList | undefined;

	constructor(
		private readonly providers: readonly string[],
		private readonly onChange: (value: Record<string, number>) => void,
		private readonly onCancel: () => void,
		private readonly requestRender?: () => void,
	) {
		super();
		this.#showProviderList();
	}

	#providerIds(): string[] {
		const limits = normalizeProviderMaxInFlightRequests(settings.get("providers.maxInFlightRequests"));
		return [...new Set([...this.providers, ...Object.keys(limits)])].sort((a, b) => a.localeCompare(b));
	}

	#showProviderList(): void {
		this.clear();
		this.addChild(new Text(theme.bold(theme.fg("accent", "Max In-Flight Requests")), 0, 0));
		this.addChild(new Spacer(1));
		this.addChild(
			new Text(
				theme.fg(
					"muted",
					"Select a provider, enter a positive number to cap concurrent LLM requests, or clear it for unlimited.",
				),
				0,
				0,
			),
		);
		this.addChild(new Spacer(1));

		const limits = normalizeProviderMaxInFlightRequests(settings.get("providers.maxInFlightRequests"));
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
		this.#selectList = new SelectList(items, Math.min(Math.max(items.length, 1), 12), getSelectListTheme());
		this.#selectList.onSelect = item => {
			if (item.value === "__clear_all") {
				settings.set("providers.maxInFlightRequests", {});
				this.onChange({});
				this.#showProviderList();
				this.requestRender?.();
				return;
			}
			this.#showProviderEditor(item.value);
		};
		this.#selectList.onCancel = this.onCancel;
		this.addChild(this.#selectList);
		this.addChild(new Spacer(1));
		this.addChild(new Text(theme.fg("dim", "  Enter to edit provider · Esc to go back"), 0, 0));
	}

	#showProviderEditor(provider: string): void {
		const limits = normalizeProviderMaxInFlightRequests(settings.get("providers.maxInFlightRequests"));
		this.clear();
		this.#selectList = undefined;
		this.addChild(
			new TextInputSubmenu(
				`Max In-Flight Requests: ${provider}`,
				"Enter a positive number. Decimals round down. Clear the field to make this provider unlimited.",
				limits[provider]?.toString() ?? "",
				false,
				value => {
					const next = { ...limits };
					const trimmed = value.trim();
					if (trimmed === "") {
						delete next[provider];
					} else {
						const limit = Number(trimmed);
						if (!Number.isFinite(limit) || limit <= 0) throw new Error("Limit must be a positive number.");
						next[provider] = Math.max(1, Math.floor(limit));
					}
					const normalized = validateProviderMaxInFlightRequests(next);
					settings.set("providers.maxInFlightRequests", normalized);
					this.onChange(normalized);
					this.#showProviderList();
					this.requestRender?.();
				},
				() => {
					this.#showProviderList();
					this.requestRender?.();
				},
			),
		);
	}

	handleInput(data: string): void {
		if (this.#selectList) {
			this.#selectList.handleInput(data);
			return;
		}
		this.children[0]?.handleInput?.(data);
	}
}

let cachedSidebarWidth: number | undefined;
/**
 * Split-sidebar width derived from every group name in the schema (not just
 * the visible tab), so the divider column never moves when switching tabs or
 * when condition-gated groups appear.
 */
function settingsSidebarWidth(): number {
	if (cachedSidebarWidth === undefined) {
		let nameWidth = 0;
		for (const tab of SETTING_TABS) {
			for (const def of getSettingsForTab(tab)) {
				if (def.group) nameWidth = Math.max(nameWidth, visibleWidth(def.group));
			}
		}
		cachedSidebarWidth = Math.min(22, nameWidth) + 4;
	}
	return cachedSidebarWidth;
}

function getSettingsTabs(): Tab[] {
	return [
		...SETTING_TABS.map(id => {
			const meta = TAB_METADATA[id];
			const icon = theme.symbol(meta.icon as Parameters<typeof theme.symbol>[0]);
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
	initialSettingPath?: SettingPath;
}

export interface SettingsRuntimeContext {
	/** Registry powering the native classifier role/model browser. */
	modelRegistry?: ModelRegistry;
	/** Owner-formatted live selected quota/overflow; undefined means unavailable, never zero. */
	getPreservationLimitUsage?: (path: SettingPath) => string | undefined;
	/** Effective selected model maximum context; never subtract reserve. */
	maxContextTokens?: number;
	/** Available thinking levels (from session) */
	availableThinkingLevels: Effort[];
	/** Current thinking level (from session) */
	thinkingLevel: ThinkingLevel | undefined;
	/** Available themes */
	availableThemes: string[];
	/** Provider/source ids shown in /model. */
	providers: string[];
	/** Working directory for plugins tab */
	cwd: string;
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
	onChange: (path: SettingPath, newValue: unknown) => void;
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

	constructor(
		private readonly context: SettingsRuntimeContext,
		private readonly callbacks: SettingsCallbacks,
		options: SettingsSelectorOptions = {},
	) {
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

		const initialDef = options.initialSettingPath ? getSettingDef(options.initialSettingPath) : undefined;
		const initialTab = initialDef?.tab ?? options.initialTab ?? "appearance";
		this.#tabBar.setTabs(getSettingsTabs(), initialTab);
		this.#switchToTab(initialTab);
		if (options.initialSettingPath) this.#currentList?.selectItem(options.initialSettingPath);
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
		if (this.#searchList) {
			return "Enter to change · Tab to jump tabs · Esc to exit search";
		}
		if (this.#currentTabId === "plugins") {
			return "Tab to switch tabs · Esc to close";
		}
		if (this.#currentList?.sectionFocused) {
			return "↑/↓ to jump sections · Tab/Enter to settings · ←/→ to switch tabs · Esc to close";
		}
		if ((this.#searchList ?? this.#currentList)?.hasOpenSubmenu()) return "Esc back · Enter change · ↑/↓ move";
		const nav = this.#hasSectionJump ? "Tab to jump sections · ←/→ to switch tabs" : "Tab to switch tabs";
		return `Esc close · Enter change · ${nav} · Type to search`;
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
		const height = Math.max(1, process.stdout.rows || 40);
		const innerWidth = Math.max(1, width - 4);

		const tabLines = this.#tabBar.render(innerWidth);
		const searching = this.#searchList !== null;
		const showPreview = !searching && this.#currentTabId === "appearance" && height >= 22;
		const previewLines = showPreview ? ["", theme.fg("muted", "Preview:"), this.#getStatusPreviewString()] : [];

		// Fixed chrome: top border, tabs, divider, [search row], divider, hint, bottom border.
		const fixedRows = 1 + tabLines.length + 1 + (searching ? 1 : 0) + 1 + 1 + 1;
		const contentRows = Math.max(0, height - fixedRows - previewLines.length);
		this.#contentRowCount = contentRows;

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
			(id, newValue) => this.#onSearchSettingChange(id as SettingPath, newValue),
			() => this.callbacks.onCancel(),
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
		if (query.length === 0) {
			this.#endSearch(false);
			return;
		}
		this.#searchQuery = query;

		const counts = new Map<SettingTab, number>();
		const items: SettingItem[] = [];
		const tabResults: { tab: SettingTab; matched: SettingItem[]; bestScore: number; order: number }[] = [];
		this.#searchFirstMatch.clear();
		let total = 0;
		for (const tab of SETTING_TABS) {
			const candidates: SettingItem[] = [];
			for (const def of getSettingsForTab(tab)) {
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
				label: `${theme.symbol(meta.icon as Parameters<typeof theme.symbol>[0])} ${meta.label}`,
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
		const selectedDef = selected ? getSettingDef(selected.id as SettingPath) : undefined;
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
			const icon = theme.symbol(meta.icon as Parameters<typeof theme.symbol>[0]);
			const count = counts.get(id) ?? 0;
			if (count > 0) {
				matched.push({ id, label: `${icon} ${meta.label} (${count})`, short: `${icon} ${count}` });
			}
		}
		for (const id of SETTING_TABS) {
			if (matchedIds.has(id)) continue;
			const meta = TAB_METADATA[id];
			const icon = theme.symbol(meta.icon as Parameters<typeof theme.symbol>[0]);
			empty.push({ id, label: `${icon} ${meta.label}`, short: icon, muted: true });
		}
		// Plugins hosts its own UI; it is not part of the schema-backed search.
		empty.push({ id: "plugins", label: `${theme.icon.package} Plugins`, short: theme.icon.package, muted: true });
		return [...matched, ...empty];
	}

	#syncTabBarToSelection(item: SettingItem | undefined): void {
		if (!this.#searchList || !item) return;
		const def = getSettingDef(item.id as SettingPath);
		if (def) this.#tabBar.setActiveById(def.tab);
	}

	/** Value-change dispatch for the search result list (any tab's setting). */
	#onSearchSettingChange(path: SettingPath, newValue: string): void {
		const def = getSettingDef(path);
		if (!def) return;
		if (def.type === "boolean") {
			const boolValue = newValue === "true";
			settings.set(path, boolValue as never);
			this.callbacks.onChange(path, boolValue);
		} else if (def.type === "enum") {
			settings.set(path, newValue as never);
			this.callbacks.onChange(path, newValue);
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
	#defToItem(def: SettingDef): SettingItem | null {
		// Check condition: applies to every variant — booleans, enums, submenus, text inputs.
		if (def.condition && !def.condition()) {
			return null;
		}

		const currentValue = this.#getCurrentValue(def);
		const item = {
			id: def.path,
			label: def.label,
			description:
				def.path === "compaction.keepUserMessagesLlm" && !settings.get("compaction.keepUserMessages")
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
					currentValue: `${Object.keys(settings.get("compaction.keepUserMessagesRegexRules")).length} rules`,
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
				return { ...item, currentValue: currentValue ? "true" : "false", values: ["true", "false"] };

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

	/**
	 * Get the current value for a setting.
	 */
	#save(path: SettingPath, value: unknown): void {
		settings.set(path, value as never);
		this.callbacks.onChange(path, value);
	}

	#validatedInput(
		label: string,
		description: string,
		current: string,
		submit: (value: string) => void,
		cancel: () => void,
	): TextInputSubmenu {
		return new TextInputSubmenu(
			label,
			description,
			current,
			false,
			submit,
			cancel,
			() => this.#contentRowCount,
			false,
		);
	}

	#limitLabel(value: unknown): string {
		const limit = parsePreservationLimit(value);
		if (!limit) return `Invalid: ${String(value)}`;
		if (!("value" in limit)) return limit.mode === "off" ? "Off" : "All";
		return limit.mode === "context-percent" ? `${limit.value}%` : `${limit.value} ${limit.mode}`;
	}

	#limitSummary(value: unknown): string {
		const limit = parsePreservationLimit(value);
		if (!limit) return "Invalid limit — choose a valid mode/value.";
		if (limit.mode !== "context-percent")
			return `Effective limit: ${this.#limitLabel(value)}. Keep whole messages; stop before the first one that exceeds the budget.`;
		const maximum = this.context.maxContextTokens;
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
			this.#save(def.path, serializePreservationLimit(limit));
			menu.list.setItems(items());
		};
		const items = (): SettingItem[] => {
			const raw = settings.get(def.path);
			displayedValue = raw;
			displayedMaximum = this.context.maxContextTokens;
			const limit = parsePreservationLimit(raw);
			const rows: SettingItem[] = [
				{
					id: "mode",
					label: "Mode",
					currentValue: limit?.mode ?? "Invalid",
					description: def.description,
					submenu: (_cv, close) =>
						new SelectSubmenu(
							"Limit Mode",
							"Off selects none here; All removes this limit. Messages counts whole messages; Tokens budgets their content; % uses the model maximum, not current usage. Linked Rule / Manual Keep treats Off and All as uncapped.",
							[
								{ value: "off", label: "Off", description: "No messages from this selection" },
								{ value: "all", label: "All", description: "Every eligible message" },
								{ value: "messages", label: "Messages", description: "A positive whole-message count" },
								{ value: "tokens", label: "Tokens", description: "A token allowance, including zero" },
								{ value: "context-percent", label: "% maximum context", description: "0–100% of the model context size" },
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
							undefined,
							undefined,
							undefined,
							() => this.#contentRowCount,
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
						return new SettingsSubmenu(
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
												const parsed = parsePreservationLimit(`${mode}:${value.trim()}`);
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
		const menu: SettingsSubmenu = new SettingsSubmenu(
			def.label,
			items(),
			() => {},
			() => done(this.#limitLabel(settings.get(def.path))),
			() => this.#contentRowCount,
			() => {
				if (displayedValue !== settings.get(def.path) || displayedMaximum !== this.context.maxContextTokens)
					menu.list.setItems(items());
				const summary = this.context.getPreservationLimitUsage?.(def.path);
				usage.currentValue = summary ?? "Unavailable";
				usage.description =
					summary ?? "Current selection counts appear once message retention has been calculated for this session.";
			},
		);
		return menu;
	}

	#capSummary(): string {
		const cap = settings.get("compaction.keepUserMessagesFilterKeepCap");
		if (cap === "uncapped") return "No cap";
		const raw = settings.get(cap === "keep-first" ? "compaction.keepFirstLimit" : "compaction.keepLastLimit");
		const limit = parsePreservationLimit(raw);
		return `${cap === "keep-first" ? "First" : "Recent"}: ${limit?.mode === "off" || limit?.mode === "all" ? "no cap" : this.#limitLabel(raw)}`;
	}

	#createCap(done: (value?: string) => void): SettingsSubmenu {
		let displayedSummary: string;
		const usage: SettingItem = { id: "usage", label: "Current selection", currentValue: "Unavailable" };
		const items = (): SettingItem[] => {
			displayedSummary = this.#capSummary();
			const cap = settings.get("compaction.keepUserMessagesFilterKeepCap");
			const rows: SettingItem[] = [
				{
					id: "direction",
					label: "Selection order",
					currentValue: cap === "keep-last" ? "Newest first" : cap === "keep-first" ? "Oldest first" : "No cap",
					description: "Extra retention for filter Keep or manual Always messages. Newest/Oldest first uses the Recent/First limit value separately; overlap is kept once. No cap keeps all marked messages.",
					submenu: (_cv, close) =>
						new SelectSubmenu(
							"Rule / Manual Keep Order",
							"Uses the First/Recent limit value as a separate Keep/Always allowance, not the window's remaining budget or an overall cap. Messages may also qualify through First/Recent or recent protection; duplicates are kept once.",
							[
								{ value: "keep-last", label: "Newest first", description: "Use Keep Recent Limit" },
								{ value: "keep-first", label: "Oldest first", description: "Use Keep First Limit" },
								{ value: "uncapped", label: "No cap", description: "Keep all Keep/Always messages" },
							],
							cap,
							value => {
								this.#save("compaction.keepUserMessagesFilterKeepCap", value);
								close(value);
								menu.list.setItems(items());
							},
							() => close(),
							undefined,
							undefined,
							undefined,
							() => this.#contentRowCount,
						),
				},
			];
			if (cap !== "uncapped") {
				const path = cap === "keep-first" ? "compaction.keepFirstLimit" : "compaction.keepLastLimit";
				const def = getSettingDef(path)!;
				rows.push({
					id: "edge",
					label: cap === "keep-first" ? "Edit Keep First Limit…" : "Edit Keep Recent Limit…",
					currentValue: this.#limitLabel(settings.get(path)),
					description: "Changes the ordinary Keep First/Recent Limit too. Off or All means no cap here; 0 tokens or 0% is a zero-token allowance. Assistant/tool exchanges stay together; each message counts toward the allowance.",
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
					description: "This limits the Keep/Always selection, not the total retained context. A message beyond this cap can still qualify through First/Recent or recent protection. Overlapping selections keep one copy.",
			});
			rows.push(usage);
			return rows;
		};
		const menu: SettingsSubmenu = new SettingsSubmenu(
			"Rule / Manual Keep Limit",
			items(),
			() => {},
			() => done(this.#capSummary()),
			() => this.#contentRowCount,
			() => {
				if (displayedSummary !== this.#capSummary()) menu.list.setItems(items());
				const summary = this.context.getPreservationLimitUsage?.("compaction.keepUserMessagesFilterKeepCap");
				usage.currentValue = summary ?? "Unavailable";
				usage.description =
					summary ?? "Current selection counts appear once message retention has been calculated for this session.";
			},
		);
		return menu;
	}

	#categorySummary(): string {
		let keep = 0,
			never = 0;
		for (const category of PRESERVED_USER_MESSAGE_CATEGORIES) {
			const value = settings.get(PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS[category]);
			if (value === "keep") keep++;
			if (value === "exclude") never++;
		}
		return `${keep} Keep · ${never} Never · ${PRESERVED_USER_MESSAGE_CATEGORIES.length - keep - never} Auto`;
	}

	#createCategories(done: (value?: string) => void): SettingsSubmenu {
		return new SettingsSubmenu(
			"Category Rules — stored tags only",
			PRESERVED_USER_MESSAGE_CATEGORIES.map(category => {
				const path = PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS[category];
				return {
					id: path,
					label: PRESERVED_USER_MESSAGE_CATEGORY_LABELS[category],
					currentValue: this.#actionLabel(settings.get(path)),
					description: `${getUi(path)?.description ?? ""} Auto defers. Keep > Never > Auto within categories; no category priority.`,
					submenu: (_cv: string, close: (value?: string) => void) =>
						this.#actionSelector(
							settings.get(path),
							value => {
								this.#save(path, value);
								close(this.#actionLabel(value));
							},
							() => close(),
						),
				};
			}),
			() => {},
			() => done(this.#categorySummary()),
			() => this.#contentRowCount,
		);
	}

	#actionLabel(value: PreservationAction): string {
		return value === "exclude" ? "Never" : value === "keep" ? "Keep" : "Auto";
	}

	#actionSelector(current: string, save: (value: PreservationAction) => void, cancel: () => void): SelectSubmenu {
		return new SelectSubmenu(
			"Action",
			"Auto leaves the decision to other rules and First/Recent limits. Keep requests extra retention within Rule / Manual Keep Limit. Never rejects extra retention, not normal recent history. Manual choices and recent protection take precedence.",
			[
				{ value: "auto", label: "Auto", description: "No decision from this rule" },
				{ value: "keep", label: "Keep", description: "Request extra retention" },
				{ value: "exclude", label: "Never", description: "Reject extra retention" },
			],
			current,
			value => save(value as PreservationAction),
			cancel,
			undefined,
			undefined,
			undefined,
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
							validatePreservedUserMessageRegexCondition(condition);
							const rules = settings.get("compaction.keepUserMessagesRegexRules");
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
			...Object.entries(settings.get("compaction.keepUserMessagesRegexRules")).map(([condition, rule]) => ({
				id: `rule:${condition}`,
				label: condition,
				currentValue: `${this.#actionLabel(rule.state)} ${rule.caseInsensitive ? "i" : ""}${rule.final ? " Final" : ""}`,
				submenu: (_cv: string, close: (value?: string) => void) =>
					this.#createRegexRuleEditor(condition, () => {
						close();
						refresh();
					}),
			})),
		];
		const menu: SettingsSubmenu = new SettingsSubmenu(
			"Custom Regex Rules",
			items(),
			() => {},
			() => done(`${Object.keys(settings.get("compaction.keepUserMessagesRegexRules")).length} rules`),
			() => this.#contentRowCount,
		);
		return menu;
	}

	#createRegexRuleEditor(initialCondition: string, done: () => void): SettingsSubmenu {
		let condition = initialCondition;
		const current = () => settings.get("compaction.keepUserMessagesRegexRules")[condition];
		const save = (rule: PreservedUserMessageRegexRule) => {
			validatePreservedUserMessageRegexCondition(condition, rule.caseInsensitive);
			this.#save("compaction.keepUserMessagesRegexRules", {
				...settings.get("compaction.keepUserMessagesRegexRules"),
				[condition]: rule,
			});
			menu.list.setItems(items());
		};
		const items = (): SettingItem[] => {
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
								validatePreservedUserMessageRegexCondition(value, current().caseInsensitive);
								const rules = settings.get("compaction.keepUserMessagesRegexRules");
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
					currentValue: String(rule.caseInsensitive),
					values: ["true", "false"],
				},
				{
					id: "final",
					label: "Final",
					currentValue: String(rule.final ?? false),
					values: ["false", "true"],
					description: "Final runs after stored classifier policy, before manual state. Auto remains neutral.",
				},
				{
					id: "delete",
					label: "Delete rule…",
					currentValue: "",
					submenu: (_cv, close) =>
						new SelectSubmenu(
							"Delete this regex rule?",
							condition,
							[
								{ value: "no", label: "No — keep rule" },
								{ value: "yes", label: "Yes — delete rule" },
							],
							"no",
							value => {
								if (value === "yes") {
									const rules = { ...settings.get("compaction.keepUserMessagesRegexRules") };
									delete rules[condition];
									this.#save("compaction.keepUserMessagesRegexRules", rules);
									close();
									done();
								} else close();
							},
							() => close(),
							undefined,
							undefined,
							undefined,
							() => this.#contentRowCount,
						),
				},
			];
		};
		const menu: SettingsSubmenu = new SettingsSubmenu(
			"Regex Rule",
			items(),
			(id, value) => {
				if (id === "case") save({ ...current(), caseInsensitive: value === "true" });
				if (id === "final") save({ ...current(), final: value === "true" });
			},
			done,
			() => this.#contentRowCount,
		);
		return menu;
	}

	#createModelSelector(def: SettingDef, done: (value?: string) => void): ModelSelectorSubmenu {
		return new ModelSelectorSubmenu(
			this.context.modelRegistry,
			String(settings.get(def.path) ?? ""),
			value => {
				this.#save(def.path, value || undefined);
				done(value || "Automatic (@tiny)");
			},
			() => done(),
			() => this.#contentRowCount,
		);
	}

	#getCurrentValue(def: SettingDef): unknown {
		return settings.get(def.path);
	}

	#isChanged(def: SettingDef, currentValue: unknown): boolean {
		const defaultValue: unknown = getDefault(def.path);
		if (Array.isArray(currentValue) && Array.isArray(defaultValue)) {
			return (
				currentValue.length !== defaultValue.length ||
				currentValue.some((entry, index) => entry !== defaultValue[index])
			);
		}
		return !Object.is(currentValue, defaultValue);
	}

	#getSubmenuCurrentValue(path: SettingPath, value: unknown): string {
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
	): Container {
		let options = def.options;

		// Special case: inject runtime options for thinking level
		if (def.path === "defaultThinkingLevel") {
			// Prepend `auto`; the rest are the model's runtime-supported efforts.
			const levels: ConfiguredThinkingLevel[] = [AUTO_THINKING, ...this.context.availableThinkingLevels];
			options = levels.map(level => {
				const baseOpt = options.find(o => o.value === level);
				return baseOpt || { value: level, label: level };
			});
		} else if (def.path === "theme.dark" || def.path === "theme.light") {
			options = this.context.availableThemes.map(t => ({ value: t, label: t }));
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
				return this.callbacks.onThemePreview?.(value);
			};
			onPreviewCancel = () => {
				this.callbacks.onThemePreview?.(activeThemeBeforePreview);
			};
		} else if (def.path === "statusLine.preset") {
			onPreview = value => {
				const presetDef = getPreset(
					value as "default" | "minimal" | "compact" | "full" | "nerd" | "ascii" | "custom",
				);
				this.callbacks.onStatusLinePreview?.({
					preset: value as StatusLinePreset,
					leftSegments: presetDef.leftSegments,
					rightSegments: presetDef.rightSegments,
					separator: presetDef.separator,
				});
			};
			onPreviewCancel = () => {
				const currentPreset = settings.get("statusLine.preset");
				const presetDef = getPreset(currentPreset);
				this.callbacks.onStatusLinePreview?.({
					preset: currentPreset,
					leftSegments: presetDef.leftSegments,
					rightSegments: presetDef.rightSegments,
					separator: presetDef.separator,
				});
			};
		} else if (def.path === "statusLine.separator") {
			onPreview = value => {
				this.callbacks.onStatusLinePreview?.({ separator: value as StatusLineSeparatorStyle });
			};
			onPreviewCancel = () => {
				const separator = settings.get("statusLine.separator");
				this.callbacks.onStatusLinePreview?.({ separator });
			};
		} else if (def.path === "statusLine.contextLine") {
			onPreview = value => {
				this.callbacks.onStatusLinePreview?.({ contextLine: value as ContextLineMode });
			};
			onPreviewCancel = () => {
				this.callbacks.onStatusLinePreview?.({ contextLine: settings.get("statusLine.contextLine") });
			};
		} else if (def.path === "snapcompact.shape") {
			const shapePreview = new SnapcompactShapePreview(currentValue, {
				model: this.context.model,
				imageBudget: this.context.imageBudget,
				requestRender: this.context.requestRender,
			});
			onPreview = value => shapePreview.setValue(value);
			footer = shapePreview;
		} else if (def.path === "composer.shape") {
			const shapePreview = new ComposerShapePreview(String(currentValue ?? "band"), {
				requestRender: this.context.requestRender,
				status: this.context.composerPreviewStatus,
			});
			onPreview = value => shapePreview.setValue(value);
			footer = shapePreview;
		}
		// Provide status line preview for theme selection
		const isThemeSetting = def.path === "theme.dark" || def.path === "theme.light";
		const getPreview = isThemeSetting ? this.callbacks.getStatusLinePreview : undefined;

		return new SelectSubmenu(
			def.label,
			def.description,
			options,
			currentValue,
			value => {
				this.#setSettingValue(def.path, value);
				this.callbacks.onChange(def.path, value);
				done(value);
			},
			() => {
				onPreviewCancel?.();
				done();
			},
			onPreview,
			getPreview,
			footer,
		);
	}

	/**
	 * Create a text input submenu for a plain string setting.
	 */
	#createTextInput(
		def: SettingDef & { type: "text" },
		_currentValue: string,
		done: (value?: string) => void,
	): Container {
		this.#textInputActive = true;
		const wrappedDone = (value?: string) => {
			this.#textInputActive = false;
			done(value);
		};
		return new TextInputSubmenu(
			def.label,
			def.description,
			this.#formatTextInputEditValue(def.path, settings.get(def.path)),
			def.secret,
			value => {
				// Empty string clears the setting; undefined-typed string settings
				// store "" which the browser.ts expandPath ignores (no-op fallback).
				this.#setSettingValue(def.path, value);
				this.callbacks.onChange(def.path, settings.get(def.path));
				wrappedDone(this.#formatTextInputValue(def, settings.get(def.path)));
			},
			() => wrappedDone(),
		);
	}

	#createProviderLimitsInput(done: (value?: string) => void): Container {
		return new ProviderLimitsSubmenu(
			this.context.providers,
			value => {
				this.callbacks.onChange("providers.maxInFlightRequests", value);
				done(this.#formatProviderLimitsValue(value));
			},
			() => done(),
			this.context.requestRender,
		);
	}

	#formatProviderLimitsValue(value: unknown): string {
		const limits = normalizeProviderMaxInFlightRequests(value);
		const entries = Object.entries(limits).sort(([a], [b]) => a.localeCompare(b));
		if (entries.length === 0) return "Unlimited";
		return entries.map(([provider, limit]) => `${provider}: ${limit}`).join(", ");
	}

	#getMultiSelectOptions(def: SettingDef & { type: "multiselect" }) {
		if (def.path !== "providers.webSearchOrder") return def.options;
		const excluded: unknown = settings.get("providers.webSearchExclude");
		if (!Array.isArray(excluded)) return def.options;
		return def.options.filter(option => !excluded.includes(option.value));
	}

	#createMultiSelect(def: SettingDef & { type: "multiselect" }, done: (value?: string) => void): Container {
		const options = this.#getMultiSelectOptions(def);
		const current: unknown = settings.get(def.path);
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
				settings.set(def.path, value as never);
				this.callbacks.onChange(def.path, value);
			},
			() => done(this.#formatMultiSelectValue(def, settings.get(def.path))),
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

	#formatTextInputEditValue(_path: SettingPath, value: unknown): string {
		if (value === undefined || value === null) return "";
		if (typeof value === "object") return JSON.stringify(value);
		return String(value);
	}

	/**
	 * Set a setting value, handling type conversion.
	 */
	#setSettingValue(path: SettingPath, value: string): void {
		const currentValue = settings.get(path);
		const schemaType = getType(path);
		if (path === "compaction.thresholdPercent" && value === "default") {
			settings.set(path, -1 as never);
		} else if (path === "compaction.thresholdTokens" && value === "default") {
			settings.set(path, -1 as never);
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
				parsed = validateProviderMaxInFlightRequests(parsed);
			}
			settings.set(path, parsed as never);
		} else if (typeof currentValue === "number") {
			settings.set(path, Number(value) as never);
		} else if (typeof currentValue === "boolean") {
			settings.set(path, (value === "true") as never);
		} else {
			settings.set(path, value as never);
		}
	}

	/**
	 * Show a settings tab using definitions.
	 */
	#showSettingsTab(tabId: SettingTab): void {
		const defs = getSettingsForTab(tabId);

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
					settings.set(path, boolValue as never);
					this.callbacks.onChange(path, boolValue);

					if (tabId === "appearance") {
						this.#triggerStatusLinePreview();
					}
				} else if (def.type === "enum") {
					settings.set(path, newValue as never);
					this.callbacks.onChange(path, newValue);
				}
				// Submenu/text types already persisted the value inside their own
				// done callbacks before SettingsList re-dispatches here. Re-run the
				// definition-to-item mapping so condition-gated settings (e.g. the
				// Hindsight cluster guarded by memory.backend) appear/disappear
				// immediately instead of waiting for the next tab switch.
				this.#refreshCurrentTabItems(defs);
			},
			() => this.callbacks.onCancel(),
			// The selector owns type-to-search and the footer hint; pin the
			// split sidebar width so the divider never jumps between tabs.
			{ typeToSearch: false, hint: "", sidebarWidth: settingsSidebarWidth() },
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
		if (this.callbacks.getStatusLinePreview) {
			return this.callbacks.getStatusLinePreview();
		}
		return theme.fg("dim", "(preview not available)");
	}

	/**
	 * Trigger status line preview with current settings.
	 */
	#triggerStatusLinePreview(): void {
		const statusLineSettings: StatusLinePreviewSettings = {
			preset: settings.get("statusLine.preset"),
			leftSegments: settings.get("statusLine.leftSegments"),
			rightSegments: settings.get("statusLine.rightSegments"),
			separator: settings.get("statusLine.separator"),
			sessionAccent: settings.get("statusLine.sessionAccent"),
			transparent: settings.get("statusLine.transparent"),
		};
		this.callbacks.onStatusLinePreview?.(statusLineSettings);
	}

	#showPluginsTab(): void {
		this.#pluginComponent = new PluginSettingsComponent(this.context.cwd, {
			onClose: () => this.callbacks.onCancel(),
			onPluginChanged: () => this.callbacks.onPluginsChanged?.(),
			requestRender: this.context.requestRender,
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
