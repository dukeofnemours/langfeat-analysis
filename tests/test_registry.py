import json
from concurrent.futures import ThreadPoolExecutor

from langfeat_analysis.registry import EventRegistry


def test_registry_writes_one_valid_event_per_line(tmp_path):
    registry = EventRegistry(tmp_path, run_id="test-run", filename="events.jsonl")
    registry.emit("pipeline_started", "started")
    error = ValueError("bad setting")
    registry.emit(
        "item_failed",
        "failed",
        process="audio_features",
        input_path="bad.wav",
        error=error,
    )

    lines = registry.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert {event["run_id"] for event in events} == {"test-run"}
    assert events[1]["error_type"] == "ValueError"
    assert events[1]["error_message"] == "bad setting"
    assert all(line.startswith("{") and line.endswith("}") for line in lines)


def test_concurrent_events_do_not_interleave_json(tmp_path):
    registry = EventRegistry(tmp_path, filename="concurrent.jsonl")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: registry.emit(
                    "item_completed",
                    "success",
                    process="test",
                    details={"index": index},
                ),
                range(20),
            )
        )
    lines = registry.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert sorted(json.loads(line)["details"]["index"] for line in lines) == list(
        range(20)
    )
