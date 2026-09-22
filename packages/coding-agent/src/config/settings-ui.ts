import { TERMINAL } from "@oh-my-pi/pi-tui";
import {
	SETTING_TABS,
	type PreservationControl,
	type SettingsDisplayEntry,
	type SettingsHost,
} from "@oh-my-pi/pi-tui/overlays/settings-defs";
import {
	parsePreservationLimit,
	serializePreservationLimit,
	PRESERVED_USER_MESSAGE_CATEGORIES,
	PRESERVED_USER_MESSAGE_CATEGORY_LABELS,
	PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS,
	validatePreservedUserMessageRegexCondition,
} from "../session/preserved-message-settings";
import {
	normalizeProviderMaxInFlightRequests,
	Settings,
	settings,
	validateProviderMaxInFlightRequests,
} from "./settings";
import {
	getDefault,
	getEnumValues,
	getPathsForTab,
	getType,
	getUi,
	isCredential,
	type SettingPath,
} from "./settings-schema";

const CONDITIONS: Record<string, () => boolean> = {
	macOS: () => process.platform === "darwin",
	hasImageProtocol: () => !!TERMINAL.imageProtocol,
	advisorEnabled: () => {
		try {
			return Settings.instance.get("advisor.enabled") === true;
		} catch {
			return false;
		}
	},
	vimModeEnabled: () => {
		try {
			return Settings.instance.get("tui.vimMode") === true;
		} catch {
			return false;
		}
	},
	hindsightActive: () => {
		try {
			return Settings.instance.get("memory.backend") === "hindsight";
		} catch {
			return false;
		}
	},
	mnemopiActive: () => {
		try {
			return Settings.instance.get("memory.backend") === "mnemopi";
		} catch {
			return false;
		}
	},
	autolearnActive: () => {
		try {
			return Settings.instance.get("autolearn.enabled") === true;
		} catch {
			return false;
		}
	},
	autoThinkingActive: () => {
		try {
			return Settings.instance.get("defaultThinkingLevel") === "auto";
		} catch {
			return false;
		}
	},
	usageAwareFallbackEnabled: () => {
		try {
			return Settings.instance.get("retry.usageAwareFallback") === true;
		} catch {
			return false;
		}
	},
	planModeEnabled: () => {
		try {
			return Settings.instance.get("plan.enabled");
		} catch {
			return false;
		}
	},
	planAutosaveEnabled: () => {
		try {
			return Settings.instance.get("plan.enabled") && Settings.instance.get("plan.autosave");
		} catch {
			return false;
		}
	},
};

const PRESERVATION_CONTROLS: Readonly<Record<string, PreservationControl>> = {
	"compaction.keepFirstLimit": "preservationLimit",
	"compaction.keepLastLimit": "preservationLimit",
	"compaction.keepRecentUserMessagesLimit": "preservationLimit",
	"compaction.keepUserMessagesFilterKeepCap": "preservationCap",
	"compaction.keepUserMessagesLlmModel": "modelSelector",
	"compaction.keepUserMessagesRegexRules": "regexRules",
	"compaction.maxTokensPerUserMessage": "positiveTokens",
};
const PRESERVATION_BOOLEANS = new Set<string>([
	"compaction.keepUserMessages",
	"compaction.keepUserMessagesHeuristic",
	"compaction.keepUserMessagesRegex",
	"compaction.keepUserMessagesClassifierFilter",
	"compaction.keepUserMessagesLlm",
]);

/** Adapt the application schema and settings store to the terminal overlay. */
export function createSettingsHost(): SettingsHost {
	const entries: SettingsDisplayEntry[] = [];
	const categories = PRESERVED_USER_MESSAGE_CATEGORIES.map(category => {
		const path = PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS[category];
		return {
			path,
			label: PRESERVED_USER_MESSAGE_CATEGORY_LABELS[category],
			description: getUi(path)?.description ?? "",
		};
	});
	for (const tab of SETTING_TABS) {
		for (const path of getPathsForTab(tab)) {
			const originalUi = getUi(path);
			const category = categories.find(category => category.path === path);
			if (category && category !== categories[0]) continue;
			const ui = category && originalUi ? { ...originalUi, label: "Category Rules" } : originalUi;
			const control = category ? "categoryDispositions" : PRESERVATION_CONTROLS[path];
			const condition =
				control === "regexRules"
					? () => settings.get("compaction.keepUserMessagesRegex")
					: control === "positiveTokens"
						? () =>
								settings.get("compaction.keepUserMessages") &&
								settings.get("compaction.pruneLongUserMessages") !== "no"
						: ui?.condition
							? CONDITIONS[ui.condition]
							: undefined;
			entries.push({
				control,
				menuconfig: PRESERVATION_BOOLEANS.has(path),
				path,
				type: getType(path),
				defaultValue: getDefault(path),
				ui,
				enumValues: getEnumValues(path),
				credential: isCredential(path),
				condition,
			});
		}
	}
	return {
		entries,
		preservation: {
			categories,
			parseLimit: parsePreservationLimit,
			serializeLimit: serializePreservationLimit,
			validateRegexCondition: validatePreservedUserMessageRegexCondition,
		},
		get: path => settings.get(path as SettingPath),
		set: (path, value) => settings.set(path as SettingPath, value as never),
		normalizeProviderLimits: normalizeProviderMaxInFlightRequests,
		validateProviderLimits: validateProviderMaxInFlightRequests,
	};
}
