import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { PrewalkSnapshot } from "../../src/session/prewalk";

/** Spread first in a session fake; keep state and behavior overrides on the fake itself. */
export function createSessionDefaults() {
	return {
		configuredThinkingLevel: () => undefined,
		getPrewalkSnapshot: (): PrewalkSnapshot => ({
			policy: undefined,
			armed: undefined,
			standingTarget: undefined,
			planInjected: false,
			continuePending: false,
			todoSeen: false,
			completedActions: 0,
			lastCountedMessage: undefined,
			rearmPending: false,
			automaticDisabled: false,
			disabledByToggle: false,
		}),
		setActiveToolsByName: async (_toolNames: string[]) => {},
		waitForIdle: async () => {},
		prepareForHeadlessAdvisorDrain: () => {},
		waitForAdvisorCatchup: async () => true,
		getToolByName: () => undefined,
		getLastAssistantMessage: () => undefined,
		abort: async () => {},
		dispose: async () => {},
		setIrcWakeTurnObserver: () => {},
		isAdvisorActive: () => false,
		subscribeRunState: () => () => {},
	} satisfies Partial<AgentSession>;
}
