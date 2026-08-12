from langfeat_analysis.preprocessing.text import TextEmbedderGrid


def test_textgrid_reader_supports_utf16_with_bom(tmp_path):
    textgrid = tmp_path / "french.TextGrid"
    textgrid.write_text(
        '''File type = "ooTextFile"
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
''',
        encoding="utf-16",
    )

    reader = TextEmbedderGrid.__new__(TextEmbedderGrid)
    reader.annotation_path = textgrid

    assert reader._read_textgrid() == [
        ["lorsque", "j'"],
        [3.05, 3.42],
        [3.42, 3.53],
    ]
