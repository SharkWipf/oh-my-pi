import type { Model } from "@oh-my-pi/pi-ai";
import { type ConfiguredThinkingLevel, prewalkWouldBeNoop } from "@oh-my-pi/pi-tui/thinking";
import { logger } from "@oh-my-pi/pi-utils";
import type { ModelRegistry } from "../config/model-registry";
import {
	type AgentModelPatternResolutionOptions,
	normalizeModelPatternList,
	resolveAgentPrewalkPattern,
	resolveModelOverride,
} from "../config/model-resolver";
import type { Settings } from "../config/settings";
import type { Prewalk } from "../session/agent-session";
import type { AgentDefinition } from "./types";

/** Resolve an agent's prewalk default, including the bundled task opt-in. */
export function resolveAgentPrewalkDefault(agent: AgentDefinition, taskPrewalk: boolean): boolean | string | undefined {
	return agent.prewalk ?? (taskPrewalk && agent.source === "bundled" && agent.name === "task" ? true : undefined);
}

/** Inspect raw selections before role expansion erases choice provenance. */
export function hasExplicitSubagentModelChoice(
	agent: AgentDefinition,
	options: AgentModelPatternResolutionOptions,
): boolean {
	if (
		normalizeModelPatternList(options.requestModel).length ||
		normalizeModelPatternList(options.settingsOverride).length
	) {
		return true;
	}
	const patterns = normalizeModelPatternList(options.agentModel);
	if (patterns.length === 0) return false;
	// Only the bundled generic task's unconfigured default is implicit. User
	// definitions and tagged agents choosing the same role remain explicit.
	return !(
		agent.source === "bundled" &&
		agent.name === "task" &&
		patterns.length === 1 &&
		(patterns[0] === "@task" || patterns[0] === "pi/task") &&
		!options.settings?.getModelRole("task")
	);
}

/** Resolve a launch's planner and execution target without changing execution routing. */
export async function resolveSubagentPrewalk(options: {
	agent: AgentDefinition;
	settings: Settings;
	modelRegistry: ModelRegistry;
	model: Model | undefined;
	thinkingLevel: ConfiguredThinkingLevel | undefined;
	hasExplicitModelChoice: boolean;
}): Promise<{
	model: Model | undefined;
	thinkingLevel: ConfiguredThinkingLevel | undefined;
	prewalk?: Prewalk;
	automatic: boolean;
}> {
	const { agent, settings, modelRegistry, model, thinkingLevel, hasExplicitModelChoice } = options;
	const settingsOverride = settings.get("task.agentPrewalk")[agent.name];
	const agentPrewalk = resolveAgentPrewalkDefault(agent, settings.get("task.prewalk"));
	const pattern = resolveAgentPrewalkPattern({ settingsOverride, agentPrewalk });
	const override = settingsOverride?.trim().toLowerCase();
	const explicitlyOff = override === "off" || override === "false" || (!override && agentPrewalk === false);
	const automatic = settings.get("task.prewalkWithoutModelOverride") && !hasExplicitModelChoice && !explicitlyOff;
	let target: Prewalk | undefined;
	if (pattern || automatic) await modelRegistry.awaitBackgroundRefresh();
	if (pattern) {
		const resolved = resolveModelOverride([pattern], modelRegistry, settings);
		if (!resolved.model || !modelRegistry.hasConfiguredAuth(resolved.model)) {
			logger.warn("Subagent prewalk target unavailable; skipping prewalk", {
				agent: agent.name,
				pattern,
				warning: resolved.warning,
			});
		} else {
			target = { target: resolved.model, thinkingLevel: resolved.thinkingLevel };
		}
	}
	if (automatic && model) {
		const planner = resolveModelOverride(["@slow"], modelRegistry, settings);
		if (!planner.model || !modelRegistry.hasConfiguredAuth(planner.model)) {
			logger.warn("Subagent prewalk planner unavailable; skipping automatic prewalk", {
				agent: agent.name,
				pattern: "@slow",
				warning: planner.warning,
			});
		} else {
			const execution = target ?? { target: model, thinkingLevel };
			// Equal model AND effort needs neither a handoff nor its plan nudges.
			if (prewalkWouldBeNoop(planner.model, planner.thinkingLevel, execution.target, execution.thinkingLevel)) {
				return { model: execution.target, thinkingLevel: execution.thinkingLevel, automatic: true };
			}
			return { model: planner.model, thinkingLevel: planner.thinkingLevel, prewalk: execution, automatic: true };
		}
	}
	if (target && prewalkWouldBeNoop(model, thinkingLevel, target.target, target.thinkingLevel)) {
		logger.debug("Subagent prewalk target matches starting model and thinking level; skipping prewalk", {
			agent: agent.name,
			pattern,
		});
		target = undefined;
	}
	return { model, thinkingLevel, prewalk: target, automatic: false };
}
