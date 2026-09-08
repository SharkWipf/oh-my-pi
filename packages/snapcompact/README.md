# @oh-my-pi/snapcompact

Bitmap-frame context compression for vision-capable LLMs.

Instead of asking an LLM to summarize discarded conversation history, snapcompact serializes it and renders the text into dense PNG frames of pixel-font glyphs that vision models read back directly. The whole pass is local and deterministic — no LLM call, no API key, no latency beyond rendering. Rasterization and PNG encoding happen in native code (`@oh-my-pi/pi-natives`).

Built for [oh-my-pi](https://github.com/can1357/oh-my-pi)'s compaction pipeline, but the rendering API works on arbitrary text.

## How it works

1. Discarded history is serialized to compact text (`serializeConversation`), with per-tool-result and per-argument character caps.
2. Text is normalized for the selected native font (`normalize`): ANSI sequences stripped, whitespace collapsed, newline runs folded into a single full-block glyph, box drawing and compatibility symbols folded to ASCII, semantic emoji folded to ASCII labels, decorative emoji dropped, and non-Latin glyphs preserved when either the selected font or the embedded Silver fallback can render them.
3. Pages of text are rasterized into PNG frames (`render` / `renderMany`). Frame width is fixed per shape; height hugs the rows actually printed, so a partially filled frame never bills blank pixel rows.
4. Frames persist in the compaction entry's `preserveData` and are re-attached to the summary message on every context rebuild.

Frame shapes are provider-aware, chosen by SQuAD recall evals (see `research/`) against real provider billing:

| Reader | Default shape | Notes |
| --- | --- | --- |
| Anthropic | `11on16-bw` | X.org 8x13 glyphs on an 11px advance; high-res Claude lines get 1932px frames |
| Google | `8on22-bw` @2048 | X.org 8x13 glyphs on a 22px pitch; Gemini bills a fixed per-image budget, so larger frames are free chars |
| OpenAI | `8on22-bw` | X.org 8x13 glyphs on a 22px pitch, sent at `detail: "original"` |
| Unknown | Anthropic shape | Per-provider image-count budgets guard against gateways that silently drop frames |

`resolveShape({ api, id })` matches the model id, not just the wire API — a Claude routed through Vertex or OpenRouter keeps its Claude shape, priced for the gateway actually carrying the request.

Bitmap shapes keep their provider-tuned geometry and draw missing glyphs through the embedded Silver TrueType fallback one character at a time; East Asian (CJK/Kana/Hangul) glyphs render full-width across two cells so they stay legible in the narrow ASCII grid. Selecting `silver16-bw` uses Silver for the whole frame.

## Install

```sh
bun add @oh-my-pi/snapcompact
```

Ships TypeScript source directly (no build step); requires Bun ≥ 1.3.14.

## Usage

Render arbitrary text into LLM image blocks:

```ts
import { renderMany, frames, resolveShape } from "@oh-my-pi/snapcompact";

const images = renderMany(longText, { model }); // ImageContent[], first page first
const count = frames(longText, { model });      // frame count without rendering
const shape = resolveShape(model);              // eval-optimal Shape for the reader
```

Run a full compaction pass over prepared messages:

```ts
import { compact } from "@oh-my-pi/snapcompact";

const result = await compact(preparation, { model });
// result.summary        — short "resume prior conversation" lead-in, reading guide, and FILES section
// result.preserveData   — archive text, rendered frames, and optional source representation
```

### Source-aware compaction

Hosts can provide `sourcesToSummarize`, `turnPrefixSources`, `recentSources`, and already-admitted `selectedSources` alongside the ordinary preparation. Each source carries its durable entry ID, active-branch order, original message, and optional raw block intervals. Selection and ordinary retained-budget allocation remain the host's responsibility.

When an original submission differs from its expanded delivery, the host marks only that original source with `projection: "original"`. Absence names delivered/legacy coordinates. Union and coverage match entry ID plus projection, so selected original intervals cannot truncate or replace ordinarily retained delivery. Both projections remain adjacent in their chronological source position; identical originals need no projection marker. Reconstruction resolves flagged original parts through the host's normal original-submission reader.

The normal serializer and retention planner run once, with optional source capture; there is no alternate source compactor. Ordinary archive coverage comes from the actual previously committed `Archive.text`, including formerly selected content. Only missing selected intervals are added in source order, and overlaps never replace fuller ordinary spans, including partially retained recent sources. The vanilla text-edge boundaries and planned frame shapes define a fixed slot schedule. The first frame-count or measured PNG-base64 byte nonfit emits the complete rejected candidate and all subsequent content as a chronological text suffix, whether ordinary or selected. Zero selection uses the ordinary pages unchanged through the same writer. Complete admitted tool atoms bypass lossy tool-content caps but retain the standard serializer fold. Repeated rescue applies actual ordinary retention rather than keeping an unbounded prior archive as text.

The single `preserveData.sourceRepresentation` descriptor records raw/normalized positions, ordered text and frame locations, original-image references, and positive whole-message omissions. It does not duplicate image bytes or archive text. Original images are hydrated from the host's existing source storage:

```ts
const archive = getPreservedArchive(result.preserveData);
const blocks = archive && historyBlocks(archive, {
  sourceRepresentation: result.preserveData?.sourceRepresentation,
  resolveSourceImage: reference => readOriginalImage(reference),
  onEmit: (layoutIndex, blockIndex, block) => recordEmittedLocation(layoutIndex, blockIndex, block),
});
```

`historyBlocks` emits mapped text, raster frames, and original images in descriptor order; `source` parts are emitted by the host as original messages. A missing referenced original image is reported rather than silently discarded. Original images are not raster-frame slots and still pass through normal provider image/capacity checks. Reload reuses the committed descriptor and PNGs; a later real compaction re-renders actual committed source, not a protection-stripped shadow archive. Unmapped legacy archives require host-owned original-source rematerialization before entering the source-aware path.


## API surface

- **Compaction**: `compact`, `CompactionPreparation`, `CompactionResult`, `getPreservedArchive`, `images`, `historyBlocks`
- **Rendering**: `render`, `renderMany`, `frames`, `geometry`
- **Shapes**: `SHAPES`, `SHAPE_VARIANTS`, `resolveShape`, `resolveShapeForText`, `idealShapeVariant`, `isShape`, `isShapeVariantName`
- **Text**: `serializeConversation`, `normalize`, `scanRenderability`, `renderabilityProbeText`, `dimStopwords`, `wrap`
- **Budgets**: `providerImageBudget`, `MAX_FRAMES_DEFAULT`, `FRAME_TOKEN_ESTIMATE`, `HQ_EDGE_FRAMES`
- **File ops**: `createFileOps`, `computeFileLists`, `upsertFileOperations`

## References

- [Monorepo README](https://github.com/can1357/oh-my-pi#readme)
- [Compaction architecture](../../docs/compaction.md)
- [CHANGELOG](./CHANGELOG.md)
