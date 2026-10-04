/** Exercise the read CLI's vision consent boundary against a local provider. */
import { describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { TempDir } from "@oh-my-pi/pi-utils";

// 1x1 PNG so the read tool's image loader accepts the file.
const PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

// A local provider accepts the same wire request as an external vision API.
const MODELS_YML = `providers:
  testvision:
    api: openai-completions
    baseUrl: LOCAL_BASE_URL
    apiKey: "test-key"
    models:
      - id: vmodel
        name: Vision Test
        input:
          - text
          - image
        contextWindow: 128000
        maxTokens: 4096
        cost:
          input: 0
          output: 0
          cacheRead: 0
          cacheWrite: 0
`;

const READ_CLI_URL = new URL("../src/cli/read-cli.ts", import.meta.url).href;

describe("omp read <image>?q=", () => {
	it("sends image bytes only after explicit vision-question opt-in", async () => {
		const tempDir = TempDir.createSync("@pi-read-cli-imgq-");
		const requests: Array<{ messages: Array<{ content: unknown }> }> = [];
		const server = Bun.serve({
			hostname: "127.0.0.1",
			port: 0,
			async fetch(request) {
				requests.push((await request.json()) as (typeof requests)[number]);
				const chunk = { id: "local-vision", object: "chat.completion.chunk", created: 1, model: "vmodel" };
				return new Response(
					[
						"data: " +
							JSON.stringify({
								...chunk,
								choices: [
									{
										index: 0,
										delta: { role: "assistant", content: "A one-pixel image." },
										finish_reason: null,
									},
								],
							}),
						"data: " +
							JSON.stringify({
								...chunk,
								choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
								usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
							}),
						"data: [DONE]",
					].join("\n\n") + "\n\n",
					{ headers: { "content-type": "text/event-stream" } },
				);
			},
		});
		try {
			const agentDir = tempDir.join("agent");
			const home = tempDir.join("home");
			const project = tempDir.join("project");
			fs.mkdirSync(agentDir, { recursive: true });
			fs.mkdirSync(home, { recursive: true });
			fs.mkdirSync(path.join(project, ".omp"), { recursive: true });
			const configPath = path.join(project, ".omp", "config.yml");
			const roleConfig = "modelRoles:\n  vision: testvision/vmodel\n";
			fs.writeFileSync(configPath, roleConfig);
			fs.writeFileSync(
				path.join(agentDir, "models.yml"),
				MODELS_YML.replace("LOCAL_BASE_URL", "http://127.0.0.1:" + server.port + "/v1"),
			);
			const pngPath = path.join(project, "test.png");
			fs.writeFileSync(pngPath, Buffer.from(PNG_1X1, "base64"));
			const runnerPath = tempDir.join("read-cli-runner.ts");
			fs.writeFileSync(
				runnerPath,
				"import { runReadCommand } from " +
					JSON.stringify(READ_CLI_URL) +
					";\nawait runReadCommand({ path: process.argv[2] });\n",
			);
			async function readQuestion() {
				const child = Bun.spawn(["bun", runnerPath, pngPath + "?q=describe this image"], {
					cwd: project,
					env: { ...process.env, HOME: home, PI_CODING_AGENT_DIR: agentDir, PI_TEST_RUNTIME: "1" },
					stdout: "pipe",
					stderr: "pipe",
				});
				const [stdout, stderr, exitCode] = await Promise.all([
					new Response(child.stdout).text(),
					new Response(child.stderr).text(),
					child.exited,
				]);
				return { stdout, stderr, exitCode };
			}
			const denied = await readQuestion();
			expect(denied.exitCode).not.toBe(0);
			expect(requests).toHaveLength(0);
			fs.writeFileSync(configPath, roleConfig + "images:\n  questionEnabled: true\n  questionTimeoutMs: 3000\n");
			const allowed = await readQuestion();
			expect(allowed).toMatchObject({ exitCode: 0 });
			expect(allowed.stdout).toContain("A one-pixel image.");
			expect(requests).toHaveLength(1);
			const wire = JSON.stringify(requests[0]!.messages);
			expect(wire).toMatch(new RegExp("data:image/[a-z]+;base64,"));
			expect(wire).toContain("describe this image");
			fs.writeFileSync(configPath, "modelRoles:\n  vision: none\nimages:\n  questionEnabled: true\n");
			const disabledRole = await readQuestion();
			expect(disabledRole.exitCode).not.toBe(0);
			expect(requests).toHaveLength(1);
		} finally {
			await server.stop(true);
			tempDir.removeSync();
		}
	}, 60_000);
});
