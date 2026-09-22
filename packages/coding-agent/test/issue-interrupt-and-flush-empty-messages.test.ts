import { describe, expect, it, vi } from "bun:test";
import type { ImageContent } from "@oh-my-pi/pi-ai";
import { CustomEditor } from "@oh-my-pi/pi-tui/prompt/custom-editor";
import { getEditorTheme, initTheme } from "@oh-my-pi/pi-tui/theme";
import { InputController } from "@oh-my-pi/pi-coding-agent/modes/controllers/input-controller";
import type { InteractiveModeContext } from "@oh-my-pi/pi-coding-agent/modes/types";
import { USER_INTERRUPT_LABEL } from "@oh-my-pi/pi-coding-agent/session/messages";

function createContext(options?: {
	queuedMessageCount?: number;
	pendingImages?: ImageContent[];
	pendingImageLinks?: (string | undefined)[];
}) {
	let editorText = "";
	const abort = vi.fn(async () => {});
	const prompt = vi.fn(async () => {});
	const updatePendingMessagesDisplay = vi.fn();
	const requestRender = vi.fn();
	const showError = vi.fn();
	const ctx = {
		editor: {
			imageLinks: undefined as (string | undefined)[] | undefined,
			setText(text: string) {
				editorText = text;
			},
			// The stub skips chip collapsing so assertions read the wire-format text.
			setCollapsedText(text: string) {
				editorText = text;
			},
			getText() {
				return editorText;
			},
			addToHistory: vi.fn(),
			clearDraft() {
				editorText = "";
				ctx.editor.pendingImages = [];
				ctx.editor.pendingImageLinks = [];
				ctx.editor.imageLinks = undefined;
			},
			pendingImages: options?.pendingImages ? [...options.pendingImages] : ([] as ImageContent[]),
			pendingImageLinks:
				options?.pendingImageLinks ??
				options?.pendingImages?.map(() => undefined) ??
				([] as (string | undefined)[]),
		},
		ui: { requestRender },
		session: {
			isStreaming: true,
			isCompacting: false,
			isBashRunning: false,
			isEvalRunning: false,
			queuedMessageCount: options?.queuedMessageCount ?? 1,
			extensionRunner: undefined,
			abort,
			prompt,
		},
		get viewSession() {
			return (this as typeof ctx).session;
		},
		compactionQueuedMessages: [],
		locallySubmittedUserSignatures: new Set<string>(),
		isBashMode: false,
		isPythonMode: false,
		loopModeEnabled: false,
		updatePendingMessagesDisplay,
		showError,
		hasActiveBtw: () => false,
		withLocalSubmission: async (_text: string, fn: () => Promise<unknown>) => fn(),
		hasActiveOmfg: () => false,
	} as unknown as InteractiveModeContext;
	return { ctx, abort, prompt, updatePendingMessagesDisplay, requestRender, showError };
}

describe("empty submit with queued messages", () => {
	it("aborts the active stream instead of eagerly prompting a drained queue", async () => {
		const { ctx, abort, prompt, updatePendingMessagesDisplay, requestRender, showError } = createContext();
		const controller = new InputController(ctx);
		controller.setupEditorSubmitHandler();

		await ctx.editor.onSubmit?.("");

		expect(abort).toHaveBeenCalledWith({ reason: USER_INTERRUPT_LABEL });
		expect(prompt).not.toHaveBeenCalled();
		expect(showError).not.toHaveBeenCalled();
		expect(updatePendingMessagesDisplay).toHaveBeenCalledTimes(1);
		expect(requestRender).toHaveBeenCalledTimes(1);
	});

	it("resubmits an original image-only draft without injecting display markers or resurrecting deleted chips", async () => {
		await initTheme(false);
		const image: ImageContent = { type: "image", mimeType: "image/png", data: "aW1hZ2U=" };
		const { ctx, abort, prompt } = createContext({ queuedMessageCount: 1 });
		const editor = new CustomEditor(getEditorTheme());
		ctx.editor = editor;
		new InputController(ctx).setupEditorSubmitHandler();
		editor.setDraft("", [image]);
		expect(editor.composerChips().map(chip => (chip.kind === "image" ? chip.image : undefined))).toEqual([image]);
		await editor.onSubmit?.(editor.getExpandedText());
		expect(prompt).toHaveBeenCalledWith(
			"",
			expect.objectContaining({
				images: [image],
				originalSubmission: expect.objectContaining({ text: "", images: [image] }),
			}),
		);
		expect(abort).not.toHaveBeenCalled();
		prompt.mockClear();
		editor.setDraft("", [image]);
		editor.setText("");
		await editor.onSubmit?.(editor.getExpandedText());
		expect(prompt).not.toHaveBeenCalled();
		expect(abort).toHaveBeenCalledWith({ reason: USER_INTERRUPT_LABEL });
		expect(editor.pendingImages).toEqual([]);
	});

	it("restores an image-only steer when streaming dispatch rejects", async () => {
		const image: ImageContent = { type: "image", mimeType: "image/png", data: "aW1hZ2U=" };
		const { ctx, abort, prompt, showError, updatePendingMessagesDisplay, requestRender } = createContext({
			queuedMessageCount: 0,
			pendingImages: [image],
			pendingImageLinks: ["local://draft.png"],
		});
		prompt.mockImplementationOnce(async () => {
			throw new Error("queue rejected");
		});
		const controller = new InputController(ctx);
		controller.setupEditorSubmitHandler();

		await ctx.editor.onSubmit?.("[Image #1]");

		expect(abort).not.toHaveBeenCalled();
		expect(showError).toHaveBeenCalledWith("queue rejected");
		expect(ctx.editor.getText()).toBe("[Image #1]");
		expect(ctx.editor.pendingImages).toEqual([image]);
		expect(ctx.editor.pendingImageLinks).toEqual(["local://draft.png"]);
		expect(ctx.editor.imageLinks).toEqual(["local://draft.png"]);
		expect(updatePendingMessagesDisplay).toHaveBeenCalledTimes(1);
		expect(requestRender).toHaveBeenCalledTimes(1);
	});

	it("drops a pending image whose marker was deleted and aborts as an empty submit", async () => {
		// Deleting the chip token removes the attachment: an empty submit with a
		// token-less pending image behaves like a plain empty submit (abort path).
		const image: ImageContent = { type: "image", mimeType: "image/png", data: "aW1hZ2U=" };
		const { ctx, abort, prompt } = createContext({ queuedMessageCount: 1, pendingImages: [image] });
		const controller = new InputController(ctx);
		controller.setupEditorSubmitHandler();

		await ctx.editor.onSubmit?.("");

		expect(prompt).not.toHaveBeenCalled();
		expect(abort).toHaveBeenCalledWith({ reason: USER_INTERRUPT_LABEL });
		expect(ctx.editor.pendingImages).toEqual([]);
	});
});
