import json

import pytest
from renotice_release import renotice
from test_release_budget import _release


def _setup(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    notice = tmp_path / "NOTICE.new"
    notice.write_text("corrected notice\n", encoding="utf-8")
    return _release(old), notice


def _bytes(report, key, when):
    return report[key][when]["bytes"]


def test_only_the_notice_and_version_change(tmp_path):
    release, notice = _setup(tmp_path)
    new = tmp_path / "new"
    report = renotice(release, notice, "0.0.2", new)

    assert (new / "NOTICE").read_text(encoding="utf-8") == "corrected notice\n"
    old_config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    new_config = json.loads((new / "config.json").read_text(encoding="utf-8"))
    assert new_config == {**old_config, "version": "0.0.2"}
    for name in old_config["files"]:
        assert (new / name).read_bytes() == (release / name).read_bytes()
    assert report["source_version"] == "0.0.1"
    assert report["version"] == "0.0.2"
    assert report["changed"] == ["NOTICE", "config.json"]
    assert set(report["unchanged_assets"]) == set(old_config["files"])
    growth = report["total_bytes"]["after"] - report["total_bytes"]["before"]
    assert growth == sum(
        _bytes(report, key, "after") - _bytes(report, key, "before") for key in ("notice", "config")
    )


def test_refuses_same_version_same_notice_and_non_empty_output(tmp_path):
    release, notice = _setup(tmp_path)
    with pytest.raises(ValueError, match="must differ"):
        renotice(release, notice, "0.0.1", tmp_path / "a")
    with pytest.raises(ValueError, match="already in"):
        renotice(release, release / "NOTICE", "0.0.2", tmp_path / "b")
    busy = tmp_path / "c"
    busy.mkdir()
    (busy / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        renotice(release, notice, "0.0.2", busy)
