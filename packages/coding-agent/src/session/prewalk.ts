import {
	type Agent,
	type AgentMessage,
	type AgentToolResult,
	type AgentTurnEndContext,
	isSyntheticToolResultMessage,
} from "@oh-my-pi/pi-agent-core";
import { invalidateMessageCache } from "@oh-my-pi/pi-agent-core/compaction";
import type { AssistantMessage, Model, ToolResultMessage } from "@oh-my-pi/pi-ai";
import { isRecord, logger, prompt } from "@oh-my-pi/pi-utils";
import type { Settings } from "../config/settings";
import type { LocalProtocolOptions } from "../internal-urls";
import { resolveApprovedPlan } from "../plan-mode/approved-plan";
import { autosaveApprovedPlan } from "../plan-mode/plan-autosave";
import { listPlanFiles, readPlanFile } from "../plan-mode/plan-files";
import type { PlanModeState } from "../plan-mode/state";
import planYoloHandoffPrompt from "../prompts/system/plan-yolo-handoff.md" with { type: "text" };
import prewalkChecklistPrompt from "../prompts/system/prewalk-checklist.md" with { type: "text" };
import prewalkContinuePrompt from "../prompts/system/prewalk-continue.md" with { type: "text" };
import prewalkPlanPrompt from "../prompts/system/prewalk-plan.md" with { type: "text" };
import { type ConfiguredThinkingLevel, prewalkWouldBeNoop } from "@oh-my-pi/pi-tui/thinking";
import { isMCPToolName } from "../tools/builtin-names";
import {
	replaceTabs,
	shortenEmbeddedPaths,
	shortenPath,
	TRUNCATE_LENGTHS,
	truncateToWidth,
} from "@oh-my-pi/pi-tui/render/render-utils";
import type { PlanProposalHandler } from "../tools/resolve";
import { ToolError } from "@oh-my-pi/pi-tui/tools/tool-errors";
import type { PlanYolo, Prewalk } from "./agent-session-types";
import { PREWALK_PLAN_MESSAGE_TYPE } from "./messages";
import type { SessionManager } from "./session-manager";
import { sameMessageContent, sessionMessagePersistenceKey } from "./turn-persistence";
const PREWALK_CONTINUE_MESSAGE_TYPE = "prewalk-continue";
const PREWALK_CHECKLIST_MESSAGE_TYPE = "prewalk-checklist";

/** Hidden plan steering is consumed within the live run and must not reappear after a context rebuild. */
export function isPrewalkPlanNudge(message: AgentMessage): boolean {
	return message.role === "custom" && message.customType === PREWALK_PLAN_MESSAGE_TYPE;
}
const PREWALK_ACTION_TOOLS: Record<string, true> = {
	edit: true,
	write: true,
};
const PLAN_YOLO_HANDOFF_MESSAGE_TYPE = "plan-yolo-handoff";

function evalStatusEvents(result: ToolResultMessage): unknown[] | undefined {
	if (result.toolName !== "eval" || !isRecord(result.details)) return undefined;
	// Live progress from an auto-backgrounded cell is not a completed action.
	if (isRecord(result.details.async) && result.details.async.state === "running") return undefined;
	return Array.isArray(result.details.statusEvents) ? result.details.statusEvents : undefined;
}

/**
 * Whether a completed tool result starts implementation. Successful nested
 * edit/write calls count even if their Eval cell later fails. Device dispatch
 * counts only at the write/exec tier; read-only navigation and help do not.
 */
function isPrewalkImplementationAction(result: ToolResultMessage): boolean {
	if (result.toolName === "eval") {
		return (
			evalStatusEvents(result)?.some(
				event =>
					isRecord(event) &&
					typeof event.op === "string" &&
					PREWALK_ACTION_TOOLS[event.op] &&
					event.committed === true &&
					event.hasError !== true &&
					event.error === undefined &&
					isPrewalkMutationDetails(event),
			) ?? false
		);
	}
	return !result.isError && !!PREWALK_ACTION_TOOLS[result.toolName] && isPrewalkMutationDetails(result.details);
}

function isPrewalkMutationDetails(details: unknown): boolean {
	// A direct filesystem edit/write carries no `xd://` dispatch metadata.
	if (!details || typeof details !== "object" || !("xdev" in details) || !details.xdev) return true;
	const xdev = details.xdev;
	// Device dispatch: switch only on a genuine mutation tier. An absent tier
	// (help lookup, unresolved approval) declines the switch, matching the
	// reporter's "stay on the large model a couple turns longer" preference.
	if (typeof xdev !== "object" || !("tier" in xdev)) return false;
	return xdev.tier === "write" || xdev.tier === "exec";
}

/** Capabilities the prewalk coordinator borrows from its owning session. */
export interface PrewalkCoordinatorHost {
	agent: Agent;
	sessionManager: SessionManager;
	settings: Pick<Settings, "get">;
	model(): Model | undefined;
	configuredThinkingLevel(): ConfiguredThinkingLevel | undefined;
	hasRunningEvalJobs?(): boolean;
	resolveAutomaticPolicy?():
		| { source: Model; thinkingLevel: ConfiguredThinkingLevel | undefined; target: Prewalk }
		| undefined;
	emitNotice(level: "info" | "warning" | "error", message: string, source?: string): void;
	setModelTemporary(
		model: Model,
		thinkingLevel?: ConfiguredThinkingLevel,
		options?: { ephemeral?: boolean },
	): Promise<void>;
	setActiveToolsByName(names: string[]): Promise<void>;
	restoreNonMCPToolPresentation(nonMCPToolNames: string[], nonMCPMountedToolNames: string[]): Promise<void>;
	getActiveToolNames(): string[];
	getEnabledToolNames(): string[];
	getMountedXdevToolNames(): string[];
	hasBuiltInTool(name: string): boolean;
	getPlanModeState(): PlanModeState | undefined;
	setPlanModeState(state: PlanModeState | undefined): void;
	getPlanReferencePath(): string;
	setPlanProposalHandler(handler: PlanProposalHandler | null): void;
	waitForSessionMessagePersistence(message: AgentMessage): Promise<void>;
	localProtocolOptions(): LocalProtocolOptions;
}

/** In-memory cycle state retained when a parked agent session is recreated. */
export interface PrewalkSnapshot {
	readonly policy: { source: Model; thinkingLevel: ConfiguredThinkingLevel | undefined; target: Prewalk } | undefined;
	readonly armed: Prewalk | undefined;
	readonly standingTarget: Pick<Model, "provider" | "id"> | undefined;
	readonly planInjected: boolean;
	readonly continuePending: boolean;
	readonly todoSeen: boolean;
	readonly completedActions: number;
	readonly lastCountedMessage: AssistantMessage | undefined;
	readonly rearmPending: boolean;
	readonly automaticDisabled: boolean;
	readonly disabledByToggle: boolean;
}

/** Initial state for prewalk and plan-yolo startup flows. */
export interface PrewalkCoordinatorOptions {
	prewalk?: Prewalk | false;
	prewalkSnapshot?: PrewalkSnapshot;
	planYolo?: PlanYolo;
}

/** Coordinates one-way model prewalks and automatic plan-yolo handoffs. */

export type PrewalkRestartResult = "armed" | "reset" | "rejected";
export class PrewalkCoordinator {
	readonly #host: PrewalkCoordinatorHost;
	#prewalk: Prewalk | undefined;
	readonly #automaticDisabled: boolean;
	#policy: { source: Model; thinkingLevel: ConfiguredThinkingLevel | undefined; target: Prewalk } | undefined;
	#rearmPending = false;
	#disabledByToggle = false;
	#standingTarget: Pick<Model, "provider" | "id"> | undefined;
	#planInjected = false;
	#continuePending = false;
	#todoSeen = false;
	#completedActions = 0;
	#cycleGeneration = 0;
	#lastCountedMessage: AssistantMessage | undefined;
	#planYolo: PlanYolo | undefined;
	#planYoloPreviousNonMCPPresentation: { enabled: string[]; mounted: string[] } | undefined;
	#planYoloArmed = false;

	constructor(host: PrewalkCoordinatorHost, options: PrewalkCoordinatorOptions = {}) {
		this.#host = host;
		const snapshot = options.prewalkSnapshot;
		this.#automaticDisabled = snapshot?.automaticDisabled ?? options.prewalk === false;
		if (snapshot) {
			this.#policy = snapshot.policy;
			this.#prewalk = snapshot.armed;
			this.#standingTarget = snapshot.standingTarget;
			this.#planInjected = snapshot.planInjected;
			this.#continuePending = snapshot.continuePending;
			this.#todoSeen = snapshot.todoSeen;
			this.#completedActions = snapshot.completedActions;
			this.#lastCountedMessage = snapshot.lastCountedMessage;
			this.#rearmPending = snapshot.rearmPending;
			this.#disabledByToggle = snapshot.disabledByToggle;
		} else {
			this.#prewalk = options.prewalk || undefined;
			if (this.#prewalk) this.#capturePolicy(this.#prewalk);
		}
		this.#planYolo = options.planYolo;
	}

	get snapshot(): PrewalkSnapshot {
		return {
			policy: this.#policy,
			armed: this.#prewalk,
			standingTarget: this.#standingTarget,
			planInjected: this.#planInjected,
			continuePending: this.#continuePending,
			todoSeen: this.#todoSeen,
			completedActions: this.#completedActions,
			lastCountedMessage: this.#lastCountedMessage,
			rearmPending: this.#rearmPending,
			automaticDisabled: this.#automaticDisabled,
			disabledByToggle: this.#disabledByToggle,
		};
	}

	/** Count a nested execution on settlement, never in a later planning cycle. */
	beginToolCall(): (() => void) | undefined {
		if (!this.#prewalk) return undefined;
		const generation = this.#cycleGeneration;
		let completed = false;
		return () => {
			if (completed) return;
			completed = true;
			if (this.#prewalk && this.#cycleGeneration === generation) this.#completedActions++;
		};
	}

	/** Current prewalk target, if the one-way switch remains armed. */
	get state(): Prewalk | undefined {
		return this.#prewalk;
	}
	get status(): "walking" | "standing" | undefined {
		if (this.willHandoff) return "walking";
		const target = this.#standingTarget;
		const active = this.#host.model();
		return target && active?.provider === target.provider && active.id === target.id ? "standing" : undefined;
	}

	#capturePolicy(target: Prewalk): void {
		const source = this.#host.model();
		this.#policy = source ? { source, thinkingLevel: this.#host.configuredThinkingLevel(), target } : undefined;
		this.#standingTarget = undefined;
	}

	setEnabled(enabled: boolean): void {
		this.#disabledByToggle = !enabled;
		if (!enabled) this.disable();
	}

	disable(): void {
		this.#scrubPlanNudge();
		this.#clearPrewalkState();
		this.#standingTarget = undefined;
		this.#rearmPending = false;
	}

	reset(): void {
		this.disable();
		this.#policy = undefined;
	}

	/** Restores planning only at a safe request boundary after newly delivered user input. */
	async beforeModelCall(hasNewUserInput: boolean): Promise<void> {
		if (
			this.#automaticDisabled ||
			this.#disabledByToggle ||
			(!this.#policy && !this.#host.settings.get("prewalk.enabled")) ||
			!this.#host.settings.get("prewalk.afterEveryUserMessage")
		) {
			this.#rearmPending = false;
			return;
		}
		if (hasNewUserInput) this.#rearmPending = true;
		if (!this.#rearmPending || this.#host.hasRunningEvalJobs?.()) return;
		const policy = this.#policy ?? this.#host.resolveAutomaticPolicy?.();
		if (!policy) {
			this.#rearmPending = false;
			this.#host.emitNotice(
				"warning",
				"Prewalk: skipped user input because the configured planning or target model is unavailable or has no configured auth.",
				"prewalk",
			);
			return;
		}
		await this.#host.setModelTemporary(policy.source, policy.thinkingLevel, { ephemeral: true });
		const retainPlanNudge =
			this.#prewalk !== undefined && this.#planInjected && !this.#host.settings.get("prewalk.repeatPlanNudge");
		if (!retainPlanNudge) this.#scrubPlanNudge();
		this.#clearPrewalkState();
		this.#planInjected = retainPlanNudge;
		this.#standingTarget = undefined;
		this.#policy = policy;
		this.#rearmPending = false;
		if (!this.#isNoop(policy.target)) this.#prewalk = policy.target;
	}
	/** Whether the armed prewalk would perform a model or thinking-level handoff. */
	get willHandoff(): boolean {
		const prewalk = this.#prewalk;
		return prewalk !== undefined && !this.#isNoop(prewalk);
	}

	#isNoop(prewalk: Prewalk): boolean {
		return prewalkWouldBeNoop(
			this.#host.model(),
			this.#host.configuredThinkingLevel(),
			prewalk.target,
			prewalk.thinkingLevel,
		);
	}

	#clearPrewalkState(): void {
		this.#cycleGeneration++;
		this.#prewalk = undefined;
		this.#planInjected = false;
		this.#continuePending = false;
		this.#todoSeen = false;
		this.#completedActions = 0;
		this.#lastCountedMessage = undefined;
	}

	#disarmNoop(prewalk: Prewalk): void {
		this.#standingTarget = undefined;
		this.#clearPrewalkState();
		this.#host.emitNotice(
			"info",
			`Prewalk: target ${prewalk.target.provider}/${prewalk.target.id} already matches the active model and thinking level; nothing to switch.`,
			"prewalk",
		);
	}

	/** Advances the one-way prewalk switch at a completed assistant-turn boundary. */
	async advanceAtTurnEnd(liveMessages: AgentMessage[], context: AgentTurnEndContext | undefined): Promise<void> {
		const prewalk = this.#prewalk;
		if (!prewalk || context?.message.role !== "assistant") return;
		const message = context.message;
		const completed = message.stopReason !== "error" && message.stopReason !== "aborted";
		if (completed) {
			const previous = this.#lastCountedMessage;
			// Tool replay closes the same response again, including after hot revival.
			// Distinct live responses still count when their timestamps/content collide.
			const alreadyCounted =
				previous === message ||
				(previous !== undefined &&
					previous.timestamp === message.timestamp &&
					!liveMessages.includes(previous) &&
					sessionMessagePersistenceKey(previous) === sessionMessagePersistenceKey(message) &&
					sameMessageContent(previous, message));
			if (!alreadyCounted) this.#completedActions++;
			this.#lastCountedMessage = message;
		}
		// Results here are new executions, including a replayed missing result.
		// Synthetic pairing placeholders never executed a tool. Nested Eval calls
		// settle through beginToolCall, not through their UI status summaries.
		for (const result of context.toolResults) {
			if (!isSyntheticToolResultMessage(result)) this.#completedActions++;
		}
		if (this.#isNoop(prewalk)) {
			this.#scrubPlanNudge(liveMessages);
			this.#disarmNoop(prewalk);
			return;
		}
		if (
			context.toolResults.some(
				result =>
					(result.toolName === "todo" && !result.isError) ||
					evalStatusEvents(result)?.some(
						event =>
							isRecord(event) &&
							event.op === "todo" &&
							event.completed === true &&
							event.hasError !== true &&
							event.error === undefined,
					),
			)
		)
			this.#todoSeen = true;

		const hasToolResults = context.toolResults.length > 0;
		if (this.#host.settings.get("prewalk.planNudge") && this.#planInjected && hasToolResults) {
			this.#continuePending = true;
		} else if (this.#host.settings.get("prewalk.planNudge") && this.#continuePending) {
			this.#continuePending = false;
			this.#host.agent.steer({
				role: "custom",
				customType: PREWALK_CONTINUE_MESSAGE_TYPE,
				content: prewalkContinuePrompt,
				attribution: "agent",
				display: false,
				timestamp: Date.now(),
			});
		}

		const minimum = Math.max(0, Math.trunc(Number(this.#host.settings.get("prewalk.minMessages")) || 0));
		const maximum = Math.max(0, Math.trunc(Number(this.#host.settings.get("prewalk.maxMessages")) || 0));
		const minimumReached = this.#completedActions >= minimum;
		const maximumReached = completed && maximum > 0 && this.#completedActions >= maximum;
		const todoGateOpen = this.#todoSeen || !this.#host.getActiveToolNames().includes("todo");
		const action =
			minimumReached && todoGateOpen
				? context.toolResults.find(result => isPrewalkImplementationAction(result))
				: undefined;
		if (!minimumReached || (!maximumReached && !action) || (maximumReached && this.#host.hasRunningEvalJobs?.())) {
			if (this.#host.settings.get("prewalk.planNudge") && !this.#planInjected) {
				this.#planInjected = true;
				this.#continuePending = true;
				this.#host.agent.steer({
					role: "custom",
					customType: PREWALK_PLAN_MESSAGE_TYPE,
					content: prewalkPlanPrompt,
					display: false,
					attribution: "agent",
					timestamp: Date.now(),
				});
				this.#host.emitNotice("info", "Prewalk: injected deep-plan nudge.", "prewalk");
			}
			return;
		}

		await this.#host.waitForSessionMessagePersistence(context.message);
		for (const toolResult of context.toolResults) {
			await this.#host.waitForSessionMessagePersistence(toolResult);
		}
		if (this.#prewalk !== prewalk || (maximumReached && this.#host.hasRunningEvalJobs?.())) return;
		this.#scrubPlanNudge(liveMessages);
		const target = prewalk.target;
		if (this.#isNoop(prewalk)) {
			this.#disarmNoop(prewalk);
			return;
		}
		const reason = maximumReached ? `${this.#completedActions} thread actions` : `first ${action!.toolName} call`;
		await this.#host.setModelTemporary(target, prewalk.thinkingLevel, { ephemeral: true });
		if (this.#prewalk !== prewalk) return;
		this.#clearPrewalkState();
		this.#standingTarget = { provider: target.provider, id: target.id };
		this.#host.emitNotice("info", `Prewalk: switched to ${target.provider}/${target.id} after ${reason}.`, "prewalk");
		this.#host.agent.steer({
			role: "custom",
			customType: PREWALK_CHECKLIST_MESSAGE_TYPE,
			content: prewalkChecklistPrompt,
			attribution: "agent",
			display: false,
			timestamp: Date.now(),
		});
	}

	/** Arms a prewalk immediately for an explicit slash-command request. */
	arm(target: Model, thinkingLevel?: ConfiguredThinkingLevel): boolean {
		const active = this.#prewalk;
		if (active) {
			this.#host.emitNotice(
				"info",
				`Prewalk: already armed for ${active.target.provider}/${active.target.id}, waiting for the first edit/write.`,
				"prewalk",
			);
			return (
				active.target.provider === target.provider &&
				active.target.id === target.id &&
				active.thinkingLevel === thinkingLevel
			);
		}
		const candidate = { target, thinkingLevel };
		if (this.#isNoop(candidate)) {
			this.#disarmNoop(candidate);
			return false;
		}
		this.#capturePolicy(candidate);
		this.#rearmPending = false;
		this.#prewalk = candidate;
		this.#planInjected = this.#host.settings.get("prewalk.planNudge");
		this.#continuePending = this.#planInjected;
		this.#todoSeen = false;
		this.#completedActions = 0;
		this.#cycleGeneration++;
		this.#lastCountedMessage = undefined;
		if (this.#planInjected) {
			this.#host.agent.steer({
				role: "custom",
				customType: PREWALK_PLAN_MESSAGE_TYPE,
				content: prewalkPlanPrompt,
				display: false,
				attribution: "agent",
				timestamp: Date.now(),
			});
		}
		this.#host.emitNotice(
			"info",
			`Prewalk: armed for ${target.provider}/${target.id} — will switch at the first edit/write once the todo list exists.`,
			"prewalk",
		);
		return true;
	}

	/**
	 * Restores the planning model and reuses or creates the requested one-shot handoff.
	 * A different active arm rejects the restart before the current model changes.
	 */
	async restart(
		source: Model,
		sourceThinkingLevel: ConfiguredThinkingLevel | undefined,
		target: Model,
		targetThinkingLevel: ConfiguredThinkingLevel | undefined,
	): Promise<PrewalkRestartResult> {
		const active = this.#prewalk;
		if (
			active &&
			(active.target.provider !== target.provider ||
				active.target.id !== target.id ||
				active.thinkingLevel !== targetThinkingLevel)
		) {
			this.arm(target, targetThinkingLevel);
			return "rejected";
		}

		await this.#host.setModelTemporary(source, sourceThinkingLevel, { ephemeral: true });
		this.#capturePolicy({ target, thinkingLevel: targetThinkingLevel });
		this.#rearmPending = false;
		this.#scrubPlanNudge();
		this.#clearPrewalkState();
		return this.arm(target, targetThinkingLevel) ? "armed" : "reset";
	}

	/** Lazily enables plan-yolo's plan phase before the first prompt is built. */
	async armPlanYoloIfNeeded(): Promise<void> {
		if (!this.#planYolo || this.#planYoloArmed) return;
		this.#planYoloArmed = true;
		const previousEnabledTools = this.#host.getEnabledToolNames();
		const previousMountedTools = this.#host.getMountedXdevToolNames();
		const previousPlanModeState = this.#host.getPlanModeState();
		const planModeState: PlanModeState = {
			enabled: true,
			planFilePath: this.#host.getPlanReferencePath() || "local://PLAN.md",
			workflow: "parallel",
		};
		// PlanYolo's injected write is a plan transport, not a user grant. Publish
		// plan mode before applying the tool set so SessionTools keeps an existing
		// device-only write restricted.
		this.#host.setPlanModeState(planModeState);
		const augmentations = this.#host.hasBuiltInTool("write") ? ["write"] : [];
		try {
			await this.#host.setActiveToolsByName([...new Set([...previousEnabledTools, ...augmentations])]);
		} catch (error) {
			this.#host.setPlanModeState(previousPlanModeState);
			this.#planYoloArmed = false;
			throw error;
		}
		this.#planYoloPreviousNonMCPPresentation = {
			enabled: previousEnabledTools.filter(name => !isMCPToolName(name)),
			mounted: previousMountedTools.filter(name => !isMCPToolName(name)),
		};
		this.#host.setPlanProposalHandler(title => this.#finalizePlanYoloProposal(title));
	}

	#scrubPlanNudge(liveMessages?: AgentMessage[]): void {
		if (!this.#planInjected) return;
		const isPlanNudge = isPrewalkPlanNudge;
		if (liveMessages) {
			for (let index = liveMessages.length - 1; index >= 0; index--) {
				if (!isPlanNudge(liveMessages[index])) continue;
				invalidateMessageCache(liveMessages[index]);
				liveMessages.splice(index, 1);
			}
		}
		const stateMessages = this.#host.agent.state.messages;
		const filtered = stateMessages.filter(message => !isPlanNudge(message));
		if (filtered.length !== stateMessages.length) this.#host.agent.replaceMessages(filtered);
	}

	async #finalizePlanYoloProposal(title: string): Promise<AgentToolResult<unknown>> {
		const planYolo = this.#planYolo;
		const state = this.#host.getPlanModeState();
		if (!planYolo || !state?.enabled) throw new ToolError("Plan mode is not active.");
		const {
			planFilePath,
			planContent,
			title: resolvedTitle,
		} = await resolveApprovedPlan({
			suppliedTitle: title,
			statePlanFilePath: state.planFilePath,
			readPlan: url =>
				readPlanFile(url, {
					localProtocolOptions: this.#host.localProtocolOptions(),
					cwd: this.#host.sessionManager.getCwd(),
				}),
			listPlanFiles: () => listPlanFiles({ localProtocolOptions: this.#host.localProtocolOptions() }),
		});
		let autosavedPlan: string | null = null;
		try {
			autosavedPlan = await autosaveApprovedPlan({
				settings: this.#host.settings,
				cwd: this.#host.sessionManager.getCwd(),
				title: resolvedTitle,
				planContent,
			});
			if (autosavedPlan) {
				const displayPath = truncateToWidth(replaceTabs(shortenPath(autosavedPlan)), TRUNCATE_LENGTHS.CONTENT);
				this.#host.emitNotice("info", `Plan autosaved to ${displayPath}.`, "plan-yolo");
			}
		} catch (error) {
			logger.warn("Failed to autosave approved plan", { error });
			const detail = truncateToWidth(
				shortenEmbeddedPaths(
					replaceTabs(error instanceof Error ? error.message : String(error))
						.replace(/[\r\n]+/g, " ")
						.trim(),
				),
				TRUNCATE_LENGTHS.CONTENT,
			);
			this.#host.emitNotice(
				"warning",
				`Plan autosave failed: ${detail} Continuing with implementation.`,
				"plan-yolo",
			);
		}
		this.#host.setPlanModeState(undefined);
		const previousPresentation = this.#planYoloPreviousNonMCPPresentation;
		try {
			if (previousPresentation) {
				await this.#host.restoreNonMCPToolPresentation(previousPresentation.enabled, previousPresentation.mounted);
			}
		} catch (error) {
			this.#host.setPlanModeState(state);
			throw error;
		}
		this.#host.setPlanProposalHandler(null);
		this.#planYolo = undefined;
		this.#planYoloPreviousNonMCPPresentation = undefined;
		await this.#host.setModelTemporary(planYolo.target, planYolo.thinkingLevel, { ephemeral: true });
		this.#host.emitNotice(
			"info",
			`Plan-yolo: plan approved, switched to ${planYolo.target.provider}/${planYolo.target.id} to implement "${resolvedTitle}".`,
			"plan-yolo",
		);
		this.#host.agent.steer({
			role: "custom",
			customType: PLAN_YOLO_HANDOFF_MESSAGE_TYPE,
			content: prompt.render(planYoloHandoffPrompt, { planFilePath, title: resolvedTitle }),
			attribution: "agent",
			display: false,
			timestamp: Date.now(),
		});
		return {
			content: [{ type: "text", text: `Plan approved. Implementing now with ${planYolo.target.id}.` }],
			details: { planFilePath, title: resolvedTitle, planExists: true },
		};
	}
}
