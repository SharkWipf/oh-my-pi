/**
 * Built-in model roles and role metadata helpers.
 */

import { modelKind, type Model } from "@oh-my-pi/pi-catalog/types";
import {
	KIND_ROLE_IDS,
	MODEL_ROLE_IDS,
	type ModelBrowserRegistry,
	type ModelBrowserRoleInfo,
	type ModelRole,
	type ModelRoleLookup,
} from "@oh-my-pi/pi-tui/overlays/model-browser";
import { isValidThemeColor } from "@oh-my-pi/pi-tui/theme";
import type { Settings } from "./settings";

/** Explicit role shutdown; unlike clearing a role, never permits automatic selection. */
export const MODEL_ROLE_DISABLED = "none";

export function isDisabledModelRoleValue(value: string | undefined): boolean {
	return value?.trim().toLowerCase() === MODEL_ROLE_DISABLED;
}

/** Resolve one configured role selector without inventing fallback candidates. */
function configuredRoleSelector(role: string, settings: ModelRoleLookup): string | undefined {
	const visited = new Set<string>();
	while (!visited.has(role)) {
		visited.add(role);
		const value = settings.getModelRole(role)?.trim();
		if (!value || value.includes(",")) return value;
		const alias =
			value === "*"
				? "default"
				: value.startsWith("@")
					? value.slice(1)
					: value.startsWith("pi/")
						? value.slice(3)
						: undefined;
		if (!alias) return value;
		role = alias.replace(/:(?:off|minimal|low|medium|high|xhigh|max|auto)$/, "");
	}
	return undefined;
}

/** Explicit shutdown follows aliases, but never invents another model. */
export function isModelRoleDisabled(role: string, settings: ModelRoleLookup): boolean {
	return isDisabledModelRoleValue(configuredRoleSelector(role, settings));
}

/** A single configured local model is not approval for an implicit online fallback. */
export function isLocalModelRoleSelection(role: string, settings: ModelRoleLookup): boolean {
	const selector = configuredRoleSelector(role, settings);
	return selector !== undefined && !selector.includes(",") && selector.startsWith("local/");
}

/** Canonical prefix for a configured model role selector. */
export const MODEL_ROLE_ALIAS_PREFIX = "@";

/** Legacy prefix accepted for backwards-compatible role selectors. */
export const LEGACY_MODEL_ROLE_ALIAS_PREFIX = "pi/";

/** Shorthand selector for the default model role. */
export const DEFAULT_MODEL_ROLE_ALIAS = "*";

/** Format a model role as its canonical selector. */
export function formatModelRoleAlias(role: string): string {
	return `${MODEL_ROLE_ALIAS_PREFIX}${role}`;
}

export type { ModelRole } from "@oh-my-pi/pi-tui/overlays/model-browser";
export { CHAT_MODEL_ROLE_IDS, KIND_ROLE_IDS, MODEL_ROLE_IDS } from "@oh-my-pi/pi-tui/overlays/model-browser";

export type ModelRoleInfo = ModelBrowserRoleInfo;

function acceptsChat(model: Model): boolean {
	return modelKind(model) === "chat";
}

function acceptsTinyOrChat(model: Model): boolean {
	const kind = modelKind(model);
	return kind === "tiny" || kind === "chat";
}

function acceptsWeb(model: Model): boolean {
	const kind = modelKind(model);
	return kind === "search" || (kind === "chat" && model.webSearch !== undefined);
}

function acceptsJudge(model: Model): boolean {
	const kind = modelKind(model);
	return kind === "judge" || kind === "tiny" || kind === "chat";
}

export const MODEL_ROLES: Record<ModelRole, ModelRoleInfo> = {
	default: {
		tag: "DEFAULT",
		name: "Default",
		color: "success",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Primary conversation and user-requested work; receives conversation context.",
	},
	smol: {
		tag: "SMOL",
		name: "Fast",
		color: "warning",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Fast user-selected agent work; optional helpers use it only when their feature is enabled.",
	},
	slow: {
		tag: "SLOW",
		name: "Thinking",
		color: "accent",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Thorough user-selected agent work; receives the requested agent context.",
	},
	vision: {
		tag: "VISION",
		name: "Vision",
		color: "error",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Image questions and enabled attachment descriptions; sends image and question.",
	},
	plan: {
		tag: "PLAN",
		name: "Architect",
		color: "muted",
		section: "chat",
		accepts: acceptsChat,
		purpose: "User-selected planning; receives task and planning context.",
	},
	commit: {
		tag: "COMMIT",
		name: "Commit",
		color: "dim",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Requested commit messages and enabled helpers; sends change summaries or diffs.",
	},
	tiny: {
		tag: "TINY",
		name: "Tiny",
		color: "dim",
		section: "chat",
		accepts: acceptsTinyOrChat,
		purpose: "Enabled titles and lightweight classifiers; sends message or task excerpts.",
	},
	memory: {
		tag: "MEMORY",
		name: "Memory",
		color: "dim",
		section: "chat",
		accepts: acceptsTinyOrChat,
		purpose: "Enabled memory processing; sends stored memories and selected conversation excerpts.",
	},
	task: {
		tag: "TASK",
		name: "Subtask",
		color: "muted",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Requested subagents; sends the assigned task and selected context.",
	},
	advisor: {
		tag: "ADVISOR",
		name: "Advisor",
		color: "accent",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Enabled second-opinion review; sends conversation and tool activity.",
	},
	requirements: {
		tag: "REQ",
		name: "Requirements Extractor",
		color: "accent",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Enabled requirements extraction; sends original messages and contextual evidence.",
	},
	requirementsEvidence: {
		tag: "EVIDENCE",
		name: "Requirements Evidence",
		color: "accent",
		section: "chat",
		accepts: acceptsChat,
		purpose: "Enabled requirements evidence review; sends candidates and their original supporting sources.",
	},
	requirementsSanity: {
		tag: "SANITY",
		name: "Requirements Sanity",
		color: "accent",
		section: "chat",
		accepts: acceptsChat,
		purpose:
			"Enabled isolated requirements sanity review; sends candidates without session history or recalled memories.",
	},
	image: {
		tag: "IMAGE",
		name: "Image generation",
		color: "accent",
		section: "kind",
		purpose: "Requested image generation; sends prompt and reference images.",
		accepts: model => modelKind(model) === "image",
	},
	web: {
		tag: "WEB",
		name: "Web search",
		color: "success",
		section: "kind",
		accepts: acceptsWeb,
		purpose: "Requested web search; sends query to the selected search provider.",
	},
	speech: {
		tag: "SPEECH",
		name: "Speech",
		color: "warning",
		section: "kind",
		purpose: "Requested or enabled vocalization; sends spoken text to the selected speech model.",
		accepts: model => modelKind(model) === "tts",
	},
	dictation: {
		tag: "DICTATION",
		name: "Dictation",
		color: "warning",
		section: "kind",
		purpose: "User-started dictation; sends microphone audio to the selected transcription model.",
		accepts: model => modelKind(model) === "stt",
	},
	judge: {
		tag: "JUDGE",
		name: "Judge",
		color: "muted",
		section: "kind",
		accepts: acceptsJudge,
		purpose:
			"Find, Eval judge/batches, auto reasoning, smart stops and judged TTSR; sends supplied context to its provider.",
	},
};

export type RoleInfo = ModelRoleInfo;

/** Whether a role belongs to the non-chat model-kind section. */
export function isKindRole(role: string): boolean {
	return KIND_ROLE_IDS.some(id => id === role);
}

function isModelRole(role: string): role is ModelRole {
	return MODEL_ROLE_IDS.some(id => id === role);
}

/** Available models eligible for a role, including keyless runner models. */
export function roleCandidatePool(role: string, settings: Settings, registry: ModelBrowserRegistry): Model[] {
	return registry.getAvailable("all").filter(getRoleInfo(role, settings).accepts);
}

/**
 * Return the canonical set of known roles for selector/carousel UI.
 *
 * Built-ins always come first. Configured cycle order, model assignments, and
 * tag metadata can introduce additional custom roles without requiring duplicate
 * entries across settings.
 */
export function getKnownRoleIds(settings: Settings): string[] {
	const roles: string[] = MODEL_ROLE_IDS.filter(role => !MODEL_ROLES[role].hidden);
	const seen = new Set<string>(roles);
	const addRole = (role: string) => {
		if (seen.has(role)) return;
		seen.add(role);
		roles.push(role);
	};

	for (const role of settings.get("cycleOrder")) addRole(role);
	for (const role in settings.getModelRoles()) addRole(role);
	for (const role in settings.get("modelTags")) addRole(role);

	return roles;
}

/**
 * Get role info for a role name (built-in or custom).
 * Configured metadata overrides built-in defaults when present.
 */
export function getRoleInfo(role: string, settings: Settings): RoleInfo {
	const builtIn = isModelRole(role) ? MODEL_ROLES[role] : undefined;
	const configuredTags = settings.get("modelTags");
	const configured = Object.hasOwn(configuredTags, role) ? configuredTags[role] : undefined;

	if (configured) {
		return {
			tag: builtIn?.tag,
			name: configured.name || builtIn?.name || role,
			color: configured.color && isValidThemeColor(configured.color) ? configured.color : builtIn?.color,
			hidden: configured.hidden ?? builtIn?.hidden,
			accepts: builtIn?.accepts ?? acceptsChat,
			purpose: builtIn?.purpose,
			section: builtIn?.section ?? "chat",
		};
	}

	if (builtIn) return builtIn;

	return {
		name: role,
		color: "muted",
		accepts: acceptsChat,
		section: "chat",
		purpose: "User-configured role; receives the context supplied by its caller.",
	};
}
