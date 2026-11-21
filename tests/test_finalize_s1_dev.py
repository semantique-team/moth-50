import json

from finalize_s1_dev import is_finalized, main, problems, rewrite, summarize


def row(index: int, split: str = "train", prefix: str = "audio/R-S1/") -> dict:
    return {
        "id": f"R-S1-DEV-{index:04d}",
        "audio": f"{prefix}S1/S1_{index:04d}.wav",
        "text": "una frase di prova",
        "split": split,
        "speaker": "S1",
        "duration": 2.5,
    }


def test_rewrite_sets_dev_and_the_session_prefix():
    rows = rewrite([row(1), row(2)])
    assert [item["split"] for item in rows] == ["dev", "dev"]
    assert rows[0]["audio"] == "audio/R-S1-dev/S1/S1_0001.wav"
    assert is_finalized(rows)


def test_rewrite_accepts_a_calibration_prefix_and_split():
    rows = rewrite(
        [row(1)],
        target_prefix="audio/R-S1-cal-norm/",
        split="dev",
    )
    assert rows[0]["audio"] == "audio/R-S1-cal-norm/S1/S1_0001.wav"
    assert is_finalized(rows, target_prefix="audio/R-S1-cal-norm/", split="dev")
    assert not is_finalized(rows)


def test_rewrite_is_idempotent():
    once = rewrite([row(1)])
    assert rewrite(once) == once


def test_rewrite_rejects_unexpected_paths():
    try:
        rewrite([row(1, prefix="audio/T/S1/")])
    except SystemExit as exc:
        assert "unexpected audio path" in str(exc)
    else:
        raise AssertionError("mixed fragments must be rejected")


def test_problems_finds_duplicates_missing_audio_and_bad_rows(tmp_path):
    rows = rewrite([row(1), row(1), row(2)])
    rows[2]["split"] = "train"
    rows.append(
        {
            "id": "R-S1-DEV-0003",
            "audio": "audio/R-S1-dev/S1/S1_0003.wav",
            "split": "dev",
            "speaker": "S2",
            "duration": 31.0,
        }
    )
    (tmp_path / "audio/R-S1-dev/S1").mkdir(parents=True)
    (tmp_path / "audio/R-S1-dev/S1/S1_0001.wav").touch()
    found = problems(rows, tmp_path)
    assert any("duplicate id" in item for item in found)
    assert any("not found" in item for item in found)
    assert any("speaker 'S2'" in item for item in found)
    assert any("outside" in item for item in found)


def test_summarize_reports_the_budget():
    text = summarize(rewrite([row(1), row(2)]))
    assert "2 rows, 0.08 min" in text
    assert "outside" in text


def test_main_checks_then_writes(tmp_path):
    (tmp_path / "audio/R-S1-dev/S1").mkdir(parents=True)
    for index in (1, 2):
        (tmp_path / f"audio/R-S1-dev/S1/S1_{index:04d}.wav").touch()
    fragment = tmp_path / "fragment.jsonl"
    fragment.write_text(
        "\n".join(json.dumps(row(i), ensure_ascii=False) for i in (1, 2)) + "\n",
        encoding="utf-8",
    )

    assert main(["--fragment", str(fragment), "--dataset-root", str(tmp_path), "--check"]) == 0
    assert json.loads(fragment.read_text(encoding="utf-8").splitlines()[0])["split"] == "train"

    assert main(["--fragment", str(fragment), "--dataset-root", str(tmp_path)]) == 0
    written = [json.loads(line) for line in fragment.read_text(encoding="utf-8").splitlines()]
    assert [item["split"] for item in written] == ["dev", "dev"]
    assert written[0]["audio"] == "audio/R-S1-dev/S1/S1_0001.wav"
