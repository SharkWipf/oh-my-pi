# Prewalk

Prewalk hands off from a planning model to a faster or cheaper execution model after planning reaches implementation. By default it runs once; an optional setting restarts it for each new user message. The planning model inspects the repository, creates a todo list, and begins the change before the target model continues.

Prewalk is off by default. Its default target is the model assigned to the `@smol` role.

## Enable prewalk

Enable prewalk persistently in the global config:

```bash
omp config set prewalk.enabled true
```

The equivalent YAML in `~/.omp/agent/config.yml` or a project `.omp/config.yml` is:

```yaml
prewalk:
  enabled: true
```

Session flags override the configured value:

| Flag | Effect |
| --- | --- |
| `--prewalk` | Arm prewalk for the new session. |
| `--no-prewalk` | Leave prewalk disabled for the session, even when `prewalk.enabled` is `true`. |
| `--prewalk-into <model-or-role>` | Arm prewalk and use the supplied model pattern or role instead of `@smol`. |

For example:

```bash
omp --prewalk
omp --prewalk-into @smol
omp --prewalk-into openai/gpt-5-mini
```

At startup, OMP resolves the target with the normal model-role and model-matching rules. If the target cannot be resolved or has no configured credentials, OMP prints a warning and starts with prewalk unarmed.

## Planning nudges

Under **Model → Prewalk**, **Deep Plan Nudge** (`prewalk.planNudge`) defaults on and injects the deep-plan reminder during each prewalk cycle, along with its related continuation reminder. Turning it off changes only these nudges: prewalk, action counting, limits and handoff remain enabled.

**Repeat Plan Nudge** (`prewalk.repeatPlanNudge`) defaults off and is shown only while **Deep Plan Nudge** is on. Enable it to repeat the reminder when new user input arrives during an unfinished prewalk. With repeat off, an already-sent reminder remains in the live thread rather than being removed or injected again.

Repeat does not control ordinary fresh cycles: each fresh cycle receives its initial reminder while **Deep Plan Nudge** is on. With **Prewalk Every User Message**, new input during an unfinished prewalk still resets the action counter and restarts the cycle, but preserves the already-sent reminder unless **Repeat Plan Nudge** is on. New input after handoff and an explicit `/prewalk restart` start fresh cycles with their initial reminders.

## Handoff trigger

When the `todo` tool is active, any successful `todo` call—including the read-only `view` operation—opens the handoff gate. OMP then switches models after the first successful completed `edit` or `write` call.

Calls to other tools do not trigger this first-mutation handoff, but their executions count toward the action limits below. A read-only `xd://` device request routed through `write`, such as LSP navigation, does not qualify as a mutation; only device operations classified as workspace writes or execution qualify.

After each handoff, the current prewalk disarms itself. Without automatic restarts, it stays disarmed until explicitly armed again. A target already matching the model and configured thinking level needs no handoff.

## Minimum and maximum actions

Under **Model → Prewalk**, **Minimum Prewalk Actions** and **Maximum Prewalk Actions** bound each planning cycle. Every successful completed assistant response counts once; every actual primary-thread tool execution counts once, including failed executions and host tools called inside JavaScript or Python Eval. An assistant response with three executed tools therefore contributes four actions. Failed or aborted assistant responses, user input, advisor activity, injections, summaries, compaction, streaming chunks and synthetic results for tools that never executed do not count. Reading retained history contributes nothing; actually reexecuting a tool whose result is missing contributes a new action without recounting its original assistant response.

- The minimum defaults to **No minimum** (`0`). Before the minimum is reached, edit/write handoff triggers are ignored, not saved for later. Afterward, a new qualifying action can trigger the normal handoff.
- The maximum defaults to **Unlimited** (`0`). A finite maximum switches to the execution model even without a todo list or edit/write.
- Limits are evaluated at safe completed-response boundaries, after the response’s tool batch settles. A batch can exceed the maximum; the minimum sees the completed batch count.
- The minimum takes precedence: with minimum `25` and maximum `10`, forced handoff waits for a safe boundary with at least `25` actions.
- A running background Eval cell delays a maximum-forced handoff until a later safe boundary; this limit never switches models mid-cell.

The existing persisted setting keys retain their names (`minMessages`, `maxMessages`, `prewalkMinMessages`, `prewalkMaxMessages`), but their values count actions.

For example:

```yaml
prewalk:
  minMessages: 15
  maxMessages: 150
```

The count starts at zero for every new cycle, including `/prewalk restart` and automatic restarts after user input. Parking and reviving a live subagent retains its current cycle count. The default limits leave the existing handoff behavior unchanged.

## Restart for every user message

In `/settings`, open **Model → Prewalk** and enable **Prewalk Every User Message**:

```yaml
prewalk:
  enabled: true
  afterEveryUserMessage: true
```

Each newly delivered user prompt, steering injection, follow-up, aside or user-authored custom message starts a fresh planning cycle. The original planning model and configured thinking level are restored before the next safe model request. Internal notices and restored history do not start cycles. An Eval cell still running in the background defers restoration until a later safe request boundary; it is never switched mid-cell.

On a resumed session without an existing cycle, the next new user message resolves the planning model from `@default` and the execution target from `@smol`. `--no-prewalk` disables automatic restarts. Disabling **Enable Prewalk** in settings cancels the armed handoff and clears its status without switching the running model.

The status line shows a walking person while a handoff is armed and a standing person after a successful handoff while its execution model remains active. Unicode uses 🚶 and 🧍; Nerd Font and ASCII presets have corresponding alternatives. Other modes remain visible alongside it. Custom status layouts need the `mode` segment to display these icons.

## Arm from an active session

Run either slash command without restarting OMP:

```text
/prewalk
/prewalk restart
```

`/prewalk` arms a one-shot handoff from the active model to the current `@smol` assignment.

After a handoff, `/prewalk restart` immediately returns the session to the current `@default` assignment and re-arms the handoff to `@smol`. Both roles are resolved when the command runs, so the cycle is independent of concrete model names and does not alter either role's persisted configuration.

If prewalk is already armed, the command leaves the existing target in place. To choose a different target at startup, use `--prewalk-into`.

## Subagent prewalk

Task subagents have separate controls: agent frontmatter, `task.prewalk`, per-agent `task.agentPrewalk` overrides, and **Tasks → Subagents → Prewalk Unpinned Subagents** (`task.prewalkWithoutModelOverride`). The explicit per-agent setting overrides frontmatter; `task.prewalk` retains its existing behavior for the bundled task agent.

The unpinned-subagent setting defaults off. When enabled, eligible launches start on `@slow` with its configured thinking level, then hand off to the existing prewalk target or otherwise the model and thinking level the child would normally use. Explicit agent models, configured role choices, per-agent model overrides, launch overrides and extension-selected models are exempt. An unconfigured bundled `@task` default and ordinary parent-model inheritance are not overrides. Explicit per-agent prewalk off stays off; plan-mode launches do not receive this automatic fallback. An unavailable planning model is reported and skipped rather than failing the child launch.

**Tasks → Subagents → Minimum/Maximum Prewalk Actions** set separate child limits. Both default to **Inherit** (`-1`), using the parent’s effective limits. An explicit minimum `0` removes the child minimum; an explicit maximum `0` makes the child maximum unlimited. These limits do not enable prewalk by themselves.

```yaml
task:
  prewalkMinMessages: 10
  prewalkMaxMessages: 100
```
