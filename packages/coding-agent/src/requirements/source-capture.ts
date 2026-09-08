import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { ImageContent, TextContent } from "@oh-my-pi/pi-ai";
import type { SessionManager } from "../session/session-manager";
import type { RequirementsObservation, RequirementsSource, RequirementsUnit } from "./types";

export type { UserMessageProducer } from "@oh-my-pi/pi-ai";
export const REQUIREMENTS_OPERATOR_DECISION_ENTRY = "requirements_operator_decision";
export interface ResolvedRequirementsSource {
	operatorTargetRevisionIds?: string[];
	source: RequirementsSource;
	units: { id: string; text?: string; image?: ImageContent }[];
	context: AgentMessage[];
	/** Actual message position within the shared frozen evidence context. */
	contextIndex?: number;
	referents: ResolvedRequirementsSource[];
	/** Unavailable originals in chronological context; no fabricated replacement bodies. */
	unavailableContext?: RequirementsSource[];
}
export function requirementsHash(value: string | Uint8Array): string {
	return new Bun.SHA256().update(value).digest("hex");
}
export function requirementsUnits(content: (TextContent | ImageContent)[]): RequirementsUnit[] {
	return content.map((part, index) => {
		const bytes = part.type === "text" ? Buffer.from(part.text) : Buffer.from(part.data, "base64");
		return { id: String(index), kind: part.type, byteLength: bytes.byteLength, sha256: requirementsHash(bytes) };
	});
}
export async function captureRequirementsSources(manager: SessionManager): Promise<{
	observations: (RequirementsObservation & { fromIntegrity: string })[];
	sources: RequirementsSource[];
	context: AgentMessage[];
}> {
	return manager.getRequirementsSources();
}
export async function resolveRequirementsSource(
	manager: SessionManager,
	key: string,
	descriptor?: RequirementsSource,
	options?: { context?: boolean },
): Promise<ResolvedRequirementsSource | undefined> {
	return manager.resolveRequirementsEvidence(key, descriptor, options);
}
