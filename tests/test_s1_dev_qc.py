import json

from s1_dev_qc import cue_index, qc, render, word_count


def take(index: int, duration: float, peak: float = -19.0, clipping: float = 0.0) -> dict:
    return {
        "name": f"S1_{index:04d}.wav",
        "index": index,
        "duration": duration,
        "peak_dbfs": peak,
        "rms_dbfs": peak - 16.0,
        "clipping": clipping,
    }


CUES = [
    {"id": f"R-S1-DEV-{i:04d}", "text": "una frase breve di prova", "category": "phrase"}
    for i in range(1, 11)
]


def test_cue_index_reads_the_take_number():
    assert cue_index("S1_0007.wav") == 7
    assert cue_index("S1_1234.wav") == 1234
    assert cue_index("notes.txt") == 0


def test_word_count_ignores_punctuation():
    assert word_count("il download si è bloccato,") == 5


def test_clean_partial_session_has_no_errors_and_projects_the_total():
    report = qc([take(i, 3.0) for i in range(1, 6)], CUES)
    assert report["clips"] == 5
    assert report["expected"] == 10
    assert report["gaps"] == []
    assert report["remaining"] == 5
    assert report["errors"] == []
    assert report["script"]["seconds_per_word"] == 0.6  # 3.0 s / 5 words
    assert report["script"]["estimated_minutes"] == 0.5  # 10 cues * 3.0 s
    assert abs(report["script"]["estimated_remaining_minutes"] - 0.25) < 0.05


def test_out_of_range_and_clipped_takes_are_errors():
    report = qc([take(1, 0.4), take(2, 31.0), take(3, 2.0, clipping=0.05)], CUES)
    assert len(report["errors"]) == 3
    assert any("outside" in item for item in report["errors"])
    assert any("clipped" in item for item in report["errors"])


def test_quiet_takes_and_holes_inside_the_session_are_warnings():
    takes = [take(i, 3.0, peak=-30.0) for i in (1, 2, 3, 5, 6, 7, 8, 9, 10)]
    report = qc(takes, CUES)
    assert report["gaps"] == [4]
    assert any("quieter" in item for item in report["warnings"])
    assert any("skipped inside" in item for item in report["warnings"])


def test_short_form_script_is_flagged():
    report = qc([take(i, 3.0) for i in range(1, 6)], CUES)
    assert report["script"]["estimated_over_8s"] == 0
    assert any("short-form" in item for item in report["warnings"])


def test_level_drift_is_reported():
    takes = [take(i, 3.0, peak=-24.0) for i in range(1, 11)]
    takes += [take(i, 3.0, peak=-15.0) for i in range(11, 21)]
    report = qc(takes, CUES + CUES)
    assert report["levels"]["drift_db"] == 9.0
    assert any("drift" in item for item in report["warnings"])


def test_render_is_a_short_summary(tmp_path):
    report = qc([take(i, 3.0) for i in range(1, 6)], CUES)
    text = render(report)
    assert "5/10 takes" in text
    assert "0.25 min recorded" in text
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["clips"] == 5
