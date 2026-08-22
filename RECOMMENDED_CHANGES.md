# Recommended Changes

This review prioritizes correctness and operational reliability. No source
changes are included in this document.

## Priority 0 — Correctness

### Make `--denoise` produce one speaker label

**Issue:** `--denoise` uses a one-source model, but the pipeline still clusters
with the default `num_speakers=2`. A single-speaker recording can therefore be
split between `SPEAKER_00` and `SPEAKER_01`.

**Relevant code:** `speechsep/main.py:107`, `speechsep/cli.py:44`,
`speechsep/cli.py:47`.

**Recommendation:** When `denoise_only=True`, skip clustering and label every
valid segment as speaker `0`. Reject combining this mode with
`--auto-speakers`, rather than silently using the normal two-speaker default.

**Acceptance criteria:** `speechsep --file call.wav --denoise` emits only
`SPEAKER_00` labels.

### Handle segments that are too short to embed

**Issue:** VAD accepts 250 ms segments, while embedding discards segments under
500 ms. If every VAD segment is discarded, `np.stack([])` raises an unhelpful
exception.

**Relevant code:** `speechsep/main.py:101`, `speechsep/pipeline/embed.py:83`,
`speechsep/pipeline/cluster.py:93`.

**Recommendation:** Check `valid_segments` immediately after embedding. Return
a clear no-transcribable-speech result, or use a documented fallback such as
padding short segments. Validate empty clustering input defensively too.

**Acceptance criteria:** Audio containing only 250–499 ms speech regions exits
cleanly and explains why no transcript was produced.

## Priority 1 — Reliable Labels and Execution

### Correct the Whisper model cache key

**Issue:** The cache considers only model size. A later CPU/CUDA call, or a call
with a different compute type, can reuse a model configured for an earlier run.

**Relevant code:** `speechsep/pipeline/transcribe.py:7`,
`speechsep/pipeline/transcribe.py:10`.

**Recommendation:** Key the cache by `(model_size, device, compute_type)` or
use a dictionary keyed by that tuple.

### Make speaker-count behavior robust for sparse data

**Issue:** Auto-clustering labels two segments as two different speakers by
default, despite insufficient evidence. Known speaker counts greater than the
number of segments are also not validated.

**Relevant code:** `speechsep/pipeline/cluster.py:30`,
`speechsep/pipeline/cluster.py:100`.

**Recommendation:** For one or two segments, use a conservative one-speaker
fallback or require an explicit speaker count. Validate non-empty embeddings,
`1 <= k <= number_of_segments`, and `max_speakers >= 1`.

### Centralize configuration validation

**Issue:** Invalid combinations, such as `--speakers 1` without `--denoise`,
non-positive `--max-speakers`, or unavailable CUDA, fail deep in the pipeline.

**Relevant code:** `speechsep/cli.py:44`, `speechsep/cli.py:57`,
`speechsep/schemas.py:73`, `speechsep/pipeline/separate.py:62`.

**Recommendation:** Add `PipelineConfig.validate()` and call it before loading
models. Use actionable messages and check `torch.cuda.is_available()` when
CUDA is requested.

## Priority 2 — Reproducibility and Operations

### Use a deterministic model-loading and cache strategy

**Issue:** VAD uses `torch.hub`, which can download repository code at runtime,
despite `silero-vad` being a declared dependency. SpeechBrain models are cached
under the current working directory.

**Relevant code:** `speechsep/pipeline/vad.py:15`,
`speechsep/pipeline/separate.py:22`, `speechsep/pipeline/embed.py:29`.

**Recommendation:** Use the installed VAD package API, pin compatible model and
library versions, and expose a cache directory via configuration or an
environment variable. Default to a user cache, not the source checkout.

### Improve failure reporting and output handling

**Issue:** File decoding, model downloads, invalid output parents, and write
failures surface as low-level exceptions. Output parent directories are not
created.

**Relevant code:** `speechsep/main.py:45`, `speechsep/output.py:115`,
`speechsep/output.py:143`.

**Recommendation:** Wrap external boundaries with concise domain errors, create
requested output parents, and derive the RTTM filename from the input path.

### Support library-friendly inputs consistently

**Issue:** `run()` accepts `str` file paths or NumPy arrays, but not
`pathlib.Path`; array shape, dtype, finiteness, and sample rate are unchecked.

**Relevant code:** `speechsep/main.py:25`, `speechsep/main.py:45`.

**Recommendation:** Accept `str | os.PathLike`, normalize inputs to finite mono
`float32`, and validate positive sample rates at the API boundary.

## Priority 3 — Tests, Tooling, and Documentation

### Add pipeline-level tests using mocks

**Issue:** Pure-logic tests exist, but no tests cover CLI parsing, configuration
validation, model-cache selection, or `run()` orchestration.

**Recommendation:** Mock all model stages to test control flow without
downloads. Cover denoise mode, no valid embeddings, sparse auto-speaker mode,
and changing Whisper device or compute type.

### Add automated quality checks

**Issue:** Pytest is the only declared development tool.

**Relevant code:** `pyproject.toml:28`.

**Recommendation:** Add formatting/linting, type checking, and CI. Keep
dependencies in one canonical source instead of manually syncing
`requirements.txt` with `pyproject.toml`.

### Update stale documentation

**Issue:** The README lists cross-source overlap resolution as future work even
though it is implemented.

**Relevant code:** `README.md:120`, `README.md:131`,
`speechsep/pipeline/overlap.py:1`.

**Recommendation:** Remove that future item and document the current overlap
heuristic, including its limitation: it removes only duplicates that receive the
same speaker label.

## Suggested Delivery Order

1. Fix denoise labeling, empty-embedding handling, and configuration checks.
2. Fix the Whisper cache key and sparse-data clustering behavior.
3. Add mocked end-to-end regression tests for those cases.
4. Move model loading and caching to a reproducible, configurable strategy.
5. Add CI, quality tooling, and documentation cleanup.

## Validation Note

During review, 31 tests completed successfully. Five output tests could not
create pytest temporary directories because this sandbox denied access to the
Windows temp location; those were environment setup failures rather than test
assertion failures.
