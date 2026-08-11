import pytest

from langfeat_analysis.io import atomic_text_writer, safe_output_path


def test_failed_atomic_write_preserves_existing_file(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_text_writer(destination) as file:
            file.write("partial")
            raise RuntimeError("simulated crash")
    assert destination.read_text(encoding="utf-8") == "original"


def test_output_template_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError, match="unsafe path"):
        safe_output_path(tmp_path, "../escape.json")
