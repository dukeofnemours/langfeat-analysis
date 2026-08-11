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

Create an editable installation with only the dependencies required by the
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

The legacy invocation remains available:

```bash
python scripts/pipelines/preproc.py --config configs/preproc.yaml
```

Inputs are processed one at a time. With `batch.continue_on_error: true`, a bad
stimulus receives a failed item result while later stimuli continue. The CLI
still exits with status 1 for partial failure so schedulers detect it; pass
`--allow-partial-success` only when a zero exit status is explicitly desired.
Use `--fail-fast` to override the YAML for a particular run.

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

Configuration and input discovery happen before model loading. Errors identify
the process, input path, concrete cause, and usually a corrective hint. Output
JSON and CSV files are written through an adjacent temporary file and moved
atomically into place, preventing truncated final files. Unsafe output templates
that try to create directories are rejected.

The current runner intentionally uses one worker. ASR and embedding models are
reused within a batch but are not safe to share across worker processes. A
configuration requesting more than one worker fails validation with this
explanation instead of silently risking accelerator errors.

## JSONL event registry

`logging` in `configs/preproc.yaml` controls the directory, optional filename,
and traceback inclusion. Every event is exactly one JSON object followed by one
newline. Records include a shared `run_id`, unique `event_id`, UTC timestamp,
event/status, process and input/output paths, duration, and exception details.

Typical lifecycle events are:

```text
pipeline_started
process_started
item_started
file_created
item_completed | item_failed
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
