import pytest

from langfeat_analysis.preprocessing.text import TextEmbedderGrid


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_textgrid_reader_supports_utf16_with_bom(tmp_path, encoding):
    textgrid = tmp_path / "french.TextGrid"
    content = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3.53
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "text words"
        xmin = 0
        xmax = 3.53
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 3.05
            text = ""
        intervals [2]:
            xmin = 3.05
            xmax = 3.42
            text = "lorsque"
        intervals [3]:
            xmin = 3.42
            xmax = 3.53
            text = "j'"
'''
    byte_order_mark = b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff"
    textgrid.write_bytes(byte_order_mark + content.encode(encoding))

    reader = TextEmbedderGrid.__new__(TextEmbedderGrid)
    reader.annotation_path = textgrid

    assert reader._read_textgrid() == [
        ["lorsque", "j'"],
        [3.05, 3.42],
        [3.42, 3.53],
    ]
