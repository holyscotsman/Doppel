import json

import pytest

from doppel.jobs import now
from doppel.stages.brand import correct_brand, run_brand
from tests.fakes import FakeImageFetcher, FakeVlm, insert_photo


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "brand_v1.txt").write_text("brand prompt v1")
    return d


@pytest.fixture
def fetcher(config):
    return FakeImageFetcher(config.cache_dir)


def tags(conn) -> dict[int, dict]:
    return {
        row["photo_id"]: dict(row)
        for row in conn.execute("SELECT * FROM tags WHERE kind = 'brand'")
    }


def test_brand_tags_non_trashed_photos_at_original_resolution(
    conn, config, fetcher, prompts_dir
) -> None:
    keeper = insert_photo(conn, "keeper", md5="k")
    trashed = insert_photo(conn, "trashed", md5="t")
    conn.execute(
        "INSERT INTO decisions (photo_id, action, decided_at) VALUES (?, 'trash', ?)",
        (trashed, now()),
    )
    conn.commit()
    vlm = FakeVlm([{"brand": "Patagonia", "evidence": "chest logo", "confidence": 0.9}])

    run_brand(conn, fetcher, vlm, config, prompts_dir=prompts_dir)

    assert len(vlm.calls) == 1
    assert vlm.calls[0]["prompt"] == "brand prompt v1"
    assert vlm.calls[0]["n_images"] == 1
    assert ("keeper", "orig") in fetcher.calls  # original resolution, per spec
    assert ("trashed", "orig") not in fetcher.calls

    tag = tags(conn)[keeper]
    assert tag["value"] == "Patagonia"
    assert tag["confidence"] == 0.9
    assert tag["source"] == "vlm"
    assert trashed not in tags(conn)

    result = conn.execute("SELECT * FROM vlm_results WHERE task = 'brand'").fetchone()
    assert result["photo_id"] == keeper
    assert result["verdict"] == "Patagonia"
    assert result["prompt_version"] == "v1"
    assert json.loads(result["response"])["evidence"] == "chest logo"


def test_brand_resumes_without_recalling_vlm(
    conn, config, fetcher, prompts_dir
) -> None:
    insert_photo(conn, "keeper", md5="k")
    run_brand(
        conn,
        fetcher,
        FakeVlm([{"brand": "Nike", "evidence": "swoosh", "confidence": 0.8}]),
        config,
        prompts_dir=prompts_dir,
    )

    run_brand(conn, fetcher, FakeVlm([]), config, prompts_dir=prompts_dir)

    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "done"
    assert scan["total"] == 0


def test_human_correction_persists_across_reruns(
    conn, config, fetcher, prompts_dir
) -> None:
    pid = insert_photo(conn, "keeper", md5="k")
    run_brand(
        conn,
        fetcher,
        FakeVlm([{"brand": "Adidas", "evidence": "stripes", "confidence": 0.4}]),
        config,
        prompts_dir=prompts_dir,
    )

    correct_brand(conn, pid, "Puma")
    tag = tags(conn)[pid]
    assert (tag["value"], tag["source"]) == ("Puma", "human")

    # bump the prompt version: the job re-runs but must skip human-corrected
    # photos entirely and never overwrite their tags
    (prompts_dir / "brand_v2.txt").write_text("brand prompt v2")
    run_brand(conn, fetcher, FakeVlm([]), config, prompts_dir=prompts_dir)

    tag = tags(conn)[pid]
    assert (tag["value"], tag["source"]) == ("Puma", "human")


def test_prompt_version_bump_retags_uncorrected_photos(
    conn, config, fetcher, prompts_dir
) -> None:
    pid = insert_photo(conn, "keeper", md5="k")
    run_brand(
        conn,
        fetcher,
        FakeVlm([{"brand": "unknown", "evidence": "no logo", "confidence": 0.2}]),
        config,
        prompts_dir=prompts_dir,
    )

    (prompts_dir / "brand_v2.txt").write_text("brand prompt v2")
    vlm2 = FakeVlm(
        [{"brand": "Carhartt", "evidence": "sleeve label", "confidence": 0.7}]
    )
    run_brand(conn, fetcher, vlm2, config, prompts_dir=prompts_dir)

    assert vlm2.calls[0]["prompt"] == "brand prompt v2"
    assert tags(conn)[pid]["value"] == "Carhartt"
    versions = [
        row["prompt_version"]
        for row in conn.execute(
            "SELECT prompt_version FROM vlm_results WHERE task = 'brand' ORDER BY id"
        )
    ]
    assert versions == ["v1", "v2"]  # v1 result preserved


def test_brand_interrupt_resumes(conn, config, fetcher, prompts_dir) -> None:
    insert_photo(conn, "p1", md5="a")
    insert_photo(conn, "p2", md5="b")

    class InterruptingVlm(FakeVlm):
        def chat_json(self, prompt, images, schema):
            if len(self.calls) >= 1:
                raise KeyboardInterrupt
            return super().chat_json(prompt, images, schema)

    with pytest.raises(KeyboardInterrupt):
        run_brand(
            conn,
            fetcher,
            InterruptingVlm(
                [{"brand": "Nike", "evidence": "swoosh", "confidence": 0.9}] * 2
            ),
            config,
            prompts_dir=prompts_dir,
        )

    assert len(tags(conn)) == 1
    scan = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    assert scan["status"] == "failed"

    resumed = FakeVlm([{"brand": "Puma", "evidence": "logo", "confidence": 0.9}])
    run_brand(conn, fetcher, resumed, config, prompts_dir=prompts_dir)
    assert len(resumed.calls) == 1
    assert len(tags(conn)) == 2


def test_missing_photos_are_not_tagged(conn, config, fetcher, prompts_dir) -> None:
    insert_photo(conn, "gone", md5="g", status="missing")

    run_brand(conn, fetcher, FakeVlm([]), config, prompts_dir=prompts_dir)

    assert tags(conn) == {}
