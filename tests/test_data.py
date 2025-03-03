import pytest

from encoders.data import parse_textgrid


@pytest.fixture
def textgrid_data_long():
    textgrid_long = """
    File type = "ooTextFile"
    Object class = "TextGrid"

    xmin = 0.0124716553288
    xmax = 729.993423837
    tiers? <exists>
    size = 2
    item []:
        item [1]:
            class = "IntervalTier"
            name = "phone"
            xmin = 0.0124716553288
            xmax = 729.993423837
            intervals: size = 3
            intervals [1]:
                xmin = 0.0124716553288
                xmax = 2.82607699773
                text = "phoneme1"
            intervals [2]:
                xmin = 2.82607709751
                xmax = 2.9465576552097734
                text = "phoneme2"
            intervals [3]:
                xmin = 2.9465576552097734
                xmax = 3.348726528529025
                text = "phoneme3"
        item [2]:
            class = "IntervalTier"
            name = "word"
            xmin = 0.0124716553288
            xmax = 729.993423837
            intervals: size = 3
            intervals [1]:
                xmin = 0.0124716553288
                xmax = 2.82607699773
                text = "word1"
            intervals [2]:
                xmin = 2.82607709751
                xmax = 2.9465576552097734
                text = "word2"
            intervals [3]:
                xmin = 2.9465576552097734
                xmax = 3.348726528529025
                text = "word3"

    """

    return textgrid_long


def test_parse_textgrid_long(textgrid_data_long):
    lines = [e.strip() for e in textgrid_data_long.split("\n") if e]
    lines = lines[0:2] + ["\n"] + lines[2::]

    output_dict = parse_textgrid(lines)

    assert isinstance(output_dict, dict)

    df1 = output_dict["phone"]
    df2 = output_dict["word"]

    assert df1.shape == (3, 3)
    assert df2.shape == (3, 3)

    assert set(df1.columns) == set(["start", "stop", "text"])
    assert set(df1.columns) == set(["start", "stop", "text"])

    assert set(df1.text) == set(["phoneme1", "phoneme2", "phoneme3"])
    assert set(df2.text) == set(["word1", "word2", "word3"])
