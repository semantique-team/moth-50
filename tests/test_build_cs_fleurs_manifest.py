from build_cs_fleurs_manifest import build_rows


def test_keeps_only_the_requested_language_and_builds_the_schema():
    metadata = [
        {
            "id": "ita_0001_SC",
            "file_name": "audio/ita/ita_0001_SC.wav",
            "text": "  Le scenes vengono displayed.  ",
            "duration": 6.5,
            "fluency": "0",
            "language": "ita-eng",
            "speaker": "SC",
        },
        {
            "id": "ara_0001_BT",
            "file_name": "audio/ara/ara_0001_BT.wav",
            "text": "نص",
            "duration": 5.0,
            "fluency": "1",
            "language": "ara-eng",
            "speaker": "BT",
        },
    ]
    rows = build_rows(metadata)

    assert [row["id"] for row in rows] == ["cs-fleurs-ita_0001_SC"]
    row = rows[0]
    assert row["audio"] == "audio/ita/ita_0001_SC.wav"
    assert row["text"] == "Le scenes vengono displayed."
    assert row["norm"] == "le scenes vengono displayed"
    assert row["source"] == "cs-fleurs"
    assert row["split"] == "test"
    assert row["lang"] == "mix"
    assert row["speaker"] == "SC"
    assert row["en_spans"] == []
    assert row["duration"] == 6.5
