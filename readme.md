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
langfeat-preproc --config configs/preproc.yaml --dry-run
langfeat-preproc --config configs/preproc.yaml
langfeat-preproc --only audio_features --only audio_embeddings
langfeat-preproc --report outputs/preproc-report.json
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
