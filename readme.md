# LangFeat analysis pipelines

This project is organized around one installable, YAML-driven preprocessing
pipeline. Audio and text extractors are implementation components; batch
discovery, failure isolation, reports, and event logging are owned by the
pipeline package.

## Layout

```text
configs/preproc.yaml                         # inputs, outputs, models, batch policy
src/langfeat_analysis/cli.py                 # installed command
src/langfeat_analysis/pipeline/              # config validation and batch runner
src/langfeat_analysis/preprocessing/         # audio/text processing components
src/langfeat_analysis/registry.py             # JSONL event registry
scripts/                                      # compatibility entry points and fixtures
tests/                                        # lightweight pipeline/registry tests
```

## Installation

### Install a released wheel

The recommended server installation uses `pipx`, which keeps `lafa` in an
isolated environment and makes the command available from any repository.
Release 0.2.1 targets Python 3.11 because upstream OpenL3 0.4.2 does not build
on Python 3.12 or newer.
Run exactly one of the installation commands below. Every installation includes
`huggingface-hub` and OpenL3 so model caching works in the pipx environment;
choose extras only for the processing backends you intend to run:

```bash
python3.11 -m pip install --user pipx
python3.11 -m pipx ensurepath

# Core pipeline only
pipx install \
  'langfeat-analysis @ https://github.com/dukeofnemours/langfeat-analysis/releases/download/v0.2.1/langfeat_analysis-0.2.1-py3-none-any.whl'

# Audio and text processes
pipx install \
  'langfeat-analysis[audio,text] @ https://github.com/dukeofnemours/langfeat-analysis/releases/download/v0.2.1/langfeat_analysis-0.2.1-py3-none-any.whl'
```

Open a new shell after `pipx ensurepath`, or follow the path instructions it
prints. Verify the installation with:

```bash
lafa --help
```

To keep the installation inside a project instead, use a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install \
  'langfeat-analysis[audio,text] @ https://github.com/dukeofnemours/langfeat-analysis/releases/download/v0.2.1/langfeat_analysis-0.2.1-py3-none-any.whl'
```

Download the example configuration into the repository where the analysis will
run, then adjust its input and output paths:

```bash
curl -fL \
  https://raw.githubusercontent.com/dukeofnemours/langfeat-analysis/v0.2.1/configs/preproc.yaml.example \
  -o preproc.yaml
lafa --config preproc.yaml --dry-run
```

Released wheels are immutable. To install a later release with `pipx`, replace
the version and wheel filename in the URL and run `pipx install --force ...`.
The `audio` extra obtains `natural-features` from its own versioned wheel
release, so installing `lafa` does not clone either source repository.

### Development installation

For an editable source checkout, install only the dependencies required by the
processes you intend to run:

```bash
python -m pip install -e .
python -m pip install -e '.[audio]'
python -m pip install -e '.[text]'
python -m pip install -e '.[audio,text]'
```

The local development environment can also use its existing dependencies with
`python -m pip install -e . --no-deps`.

## Running the pipeline

```bash
lafa --config configs/preproc.yaml --dry-run
lafa --config configs/preproc.yaml
lafa --only audio_features --only audio_embeddings
lafa --report outputs/preproc-report.json
```

Prepare model weights for an offline environment without initializing the
pipeline or touching stimulus inputs:

```bash
lafa --config configs/preproc.yaml --cache-models
lafa --config configs/preproc.yaml --verify-model-cache
```

The cache location comes from `models.cache_directory` and can be overridden
with `--model-cache-dir`. Copy that directory to the offline machine, retain
the same YAML path (or override it), and run:

```bash
lafa --config configs/preproc.yaml --offline
```

Offline mode verifies every selected model before pipeline initialization and
sets the Hugging Face offline flags. Cache preparation covers the CTC model,
the platform-selected Qwen ASR/aligner pair, and SentenceTransformer. OpenL3
weights are bundled with its package, so preparation verifies their presence
rather than downloading a second copy. `--only` limits caching and verification
to the named processes.

The legacy invocation remains available:

```bash
python scripts/pipelines/preproc.py --config configs/preproc.yaml
```

Inputs are processed one at a time. With `batch.continue_on_error: true`, a bad
stimulus receives a failed item result while later stimuli continue. The CLI
still exits with status 1 for partial failure so schedulers detect it; pass
`--allow-partial-success` only when a zero exit status is explicitly desired.
Use `--fail-fast` to override the YAML for a particular run.

### Existing transcript matching

The transcription process first compares every audio filename with existing
CSV, TSV, TXT, TextGrid, and EAF filenames. Collector names are parsed as
`safe_alias_kind_safe_relative_path-checksum` (the literal `*kind*` separator
is also supported). The `kind` and path checksum are identifiers, not stimulus
content, so they are excluded from similarity scoring. `safe_alias` must agree;
only the two `safe_relative` source stems are scored after normalizing case,
separators, and letter/number boundaries. A transcript is accepted only when
its score meets `transcripts.input.matching.threshold` and it leads the
runner-up by `min_margin`. Explicitly conflicting language tags or section
numbers are rejected. Each transcript can satisfy at most one audio file.

As a fallback, filenames are grouped by their parsed `safe_alias`. When the
original audio and text counts for an alias are equal,
the matcher computes the highest-scoring one-to-one assignment for the
remaining files. Those pairs may fall below `threshold`, but must still meet
`alias_min_score`. A singleton alias group (exactly one audio and one text) may
match below that floor when it has no explicit language or number conflict.
This permits generic names such as `alice_text_annotations_...` to pair with
Alice's sole audio file. Set `alias_count_heuristic: false` to disable this
fallback. The same matcher is used by `text_embeddings` when its strategy is
`auto`.

Validated matches are emitted as `[SKIPPED]` and linked in the JSONL event. Only
unmatched or ambiguous audio is sent to ASR. When every audio has a transcript,
the transcription and alignment models are not loaded or added to model-cache
preparation.

### Terminal feedback

During a run, `lafa` writes human-readable status updates to stderr while
leaving the final JSON report on stdout. Each pipeline and process begins with
`[PENDING]`; every item ends as `[SUCCESS]` or `[FAILED]`; and each batch item
advances a `[PROGRESS]` bar. This keeps the terminal informative without
breaking shell scripts that capture stdout. Pass `--quiet` to suppress these
terminal updates.

### Custom input and output directories

The short form follows the storage layout created by `collect-stimuli.sh`:

```bash
lafa \
  --input-dir /path/to/data-storage/features \
  --output-dir /path/to/preprocessed-results
```

When present, `INPUT/audio` supplies WAV stimuli and `INPUT/text` supplies CSV,
TSV, and TextGrid annotations. If these modality folders are absent, `INPUT` is
treated as a flat directory. Collector names such as
`alias_audio_story_123.wav` and `alias_text_story_456.csv` are paired by alias
and source stem; their different path checksums do not prevent matching.
Ambiguous matches fail explicitly. Results are routed below the output root:

```text
OUTPUT/audio/features
OUTPUT/audio/phonetics
OUTPUT/audio/embeddings
OUTPUT/transcripts
OUTPUT/text/embeddings
OUTPUT/logs
```

Ambiguous or split layouts can be specified directly:

```bash
lafa \
  --audio-input-dir /data/stimuli \
  --annotation-input-dir /data/annotations \
  --output-dir /scratch/lafa-results \
  --log-dir /project/registry/logs
```

The collector accepts MP3, TXT, and EAF for storage, but current preprocessing
supports WAV audio and CSV, TSV, or TextGrid annotations. Unsupported files are
excluded by the CLI override patterns rather than failing unrelated batch items.

## Failure messages and safeguards

Configuration validation happens before model loading. Errors identify the
process, input path, concrete cause, and usually a corrective hint. Output
JSON and CSV files are written through an adjacent temporary file and moved
atomically into place, preventing truncated final files. Unsafe output templates
that try to create directories are rejected.

The current runner intentionally uses one worker. ASR and embedding models are
reused within a batch but are not safe to share across worker processes. A
configuration requesting more than one worker fails validation with this
explanation instead of silently risking accelerator errors.

When `acoustic_phonetics` uses the CTC backend, its processor and model are
loaded once before any stimulus is processed. `ctc_device: auto` prefers CUDA,
then Apple MPS, and otherwise uses CPU. Each stimulus is divided according to
`ctc_chunk_seconds`; `ctc_batch_size: null` selects four concurrent chunks on
CUDA, two on MPS, and sequential inference on CPU. Lower either setting if an
accelerator reports insufficient memory. Model loading and per-stimulus chunk
progress are displayed in the terminal and recorded in the JSONL registry.

Every enabled model-backed process is preloaded during pipeline startup. The
order is acoustic CTC, OpenL3, transcription/forced alignment, then the
SentenceTransformer model. Signal-only `audio_features` needs no model.

OpenL3 has an independent TensorFlow device policy: `tensorflow_device: auto`
uses a TensorFlow GPU when one is visible and otherwise uses CPU. It does not
inherit `ctc_device` or PyTorch MPS/CUDA state. `inference_batch_size` controls
Keras prediction batches; `stimulus_batch_size` and
`max_stimulus_batch_seconds` bound multi-stimulus groups. If TensorFlow reports
an out-of-memory error and `memory_fallback` is enabled, the pipeline retries
stimuli sequentially and halves the inference batch until it reaches one.

## JSONL event registry

`logging` in `configs/preproc.yaml` controls the directory, optional filename,
and traceback inclusion. Every event is exactly one JSON object followed by one
newline. Records include a shared `run_id`, unique `event_id`, UTC timestamp,
event/status, process and input/output paths, duration, and exception details.

Typical lifecycle events are:

```text
model_cache_started
model_cache_completed | model_cache_failed
pipeline_started
model_load_started
model_load_completed | model_load_failed
model_load_skipped
process_started
item_started
file_created
item_completed | item_skipped | item_failed
process_completed | process_failed
pipeline_completed | pipeline_failed
```

Writes are flushed and guarded by thread plus advisory process locks, so
concurrent runs can safely append to the same daily JSONL file. Set
`logging.fsync_on_write: true` when power-loss durability matters more than
performance on a synced or network folder. The compatibility API in
`utils/registry.py` re-exports the new registry for older scripts.

## Tests

```bash
python -m pytest
```
