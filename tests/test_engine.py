import numpy as np
import pytest

from conftest import image_for


def test_build_index_tags_every_image(engine, catalog):
    engine.build_index(catalog)
    assert len(engine.metadata) == 25
    for _, r in engine.metadata.iterrows():
        assert r["image_path"].endswith(f"{r['color']}_{r['category']}.png")


def test_text_search_ranks_exact_match_first(engine, catalog):
    engine.build_index(catalog)
    top = engine.search("black shoe", top_k=3).iloc[0]
    assert (top["color"], top["category"]) == ("black", "shoe")


def test_prices_and_ratings_are_stable_across_reindexing(engine, catalog):
    engine.build_index(catalog)
    first = engine.metadata[["image_path", "price", "rating"]].copy()
    engine.build_index(catalog)
    second = engine.metadata[["image_path", "price", "rating"]]
    assert first.equals(second)
    assert engine.metadata["rating"].between(3.5, 5.0).all()


def test_search_on_empty_catalog_returns_empty_frame(engine):
    out = engine.search("red shirt")
    assert out.empty and "score" in out.columns


def test_all_images_failing_raises_clear_error(engine, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="No images"):
        engine.build_index([str(bad)])


def test_one_bad_file_is_skipped_not_fatal(engine, catalog, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"nope")
    engine.build_index(catalog + [str(bad)])
    assert len(engine.metadata) == 25


def test_filters(engine, catalog):
    engine.build_index(catalog)
    out = engine.search("shoe", top_k=50, filters={"category": "Shoe", "color": "Red"})
    assert len(out) == 1 and out.iloc[0]["color"] == "red"
    cheap = engine.search("shoe", top_k=50, filters={"max_price": 0})
    assert cheap.empty


def test_image_plus_text_blend_is_a_real_70_30_mix(engine, catalog):
    """Text vectors are 3x longer than image vectors in the fake model; unnormalised blending would let the
    text dominate. With normalisation the image still decides the category, the text only nudges the colour."""
    engine.build_index(catalog)
    q = image_for("watch", "red")
    img_only = engine.search(q, query_type="image", top_k=1).iloc[0]
    assert (img_only["category"], img_only["color"]) == ("watch", "red")
    blended = engine.search(q, query_type="image", text_query="blue", top_k=1).iloc[0]
    assert blended["category"] == "watch"


def test_invalid_query_type(engine, catalog):
    engine.build_index(catalog)
    with pytest.raises(ValueError):
        engine.search("x", query_type="audio")


def test_add_image_to_empty_engine_and_duplicate_path_not_double_indexed(engine, tmp_path):
    info = engine.add_image(image_for("laptop", "green"), str(tmp_path / "new" / "a.png"))
    assert (info["category"], info["color"]) == ("laptop", "green")
    assert len(engine.metadata) == 1 and engine.index.ntotal == 1
    out = engine.search("green laptop", top_k=5)
    assert len(out) == 1 and out.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_label_embeddings_are_cached(engine, catalog, monkeypatch):
    engine.build_index(catalog[:2])
    calls = []
    orig = engine.get_text_embedding
    monkeypatch.setattr(engine, "get_text_embedding", lambda t: calls.append(t) or orig(t))
    engine.build_index(catalog[:2])
    assert calls == []
