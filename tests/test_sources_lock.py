import json
import re
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "data" / "sources.lock.json"


def test_sources_lock_is_well_formed():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    sources = payload["sources"]
    assert sources
    ids = [source["id"] for source in sources]
    assert len(ids) == len(set(ids))
    for source in sources:
        assert source["role"]
        assert source["license"]
        assert source["status"]
        for key in ("md5",):
            if key in source:
                assert re.fullmatch(r"[0-9a-f]{32}", source[key])
        if "sha256" in source:
            assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        for download in source.get("downloads", []):
            assert re.fullmatch(r"[0-9a-f]{32}", download["md5"])


def test_sources_lock_includes_the_training_sources():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    ids = {source["id"] for source in payload["sources"]}
    assert {"moth-cs", "common-voice-it", "mls-italian", "librispeech", "musan"} <= ids
    assert {"fleurs", "cs-fleurs"} <= ids
