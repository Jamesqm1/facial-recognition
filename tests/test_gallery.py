import numpy as np

from facerec.engine import Gallery


def test_identify_returns_closest_match():
    gallery = Gallery()
    gallery.add("alice", np.array([1.0, 0.0, 0.0]))
    gallery.add("bob", np.array([0.0, 1.0, 0.0]))

    name, score = gallery.identify(np.array([0.9, 0.1, 0.0]))

    assert name == "alice"
    assert score > 0.9


def test_identify_rejects_below_threshold():
    gallery = Gallery()
    gallery.add("alice", np.array([1.0, 0.0, 0.0]))

    name, _ = gallery.identify(np.array([0.0, 0.0, 1.0]))

    assert name is None


def test_save_and_load_roundtrip(tmp_path):
    gallery = Gallery()
    gallery.add("alice", np.array([1.0, 2.0, 3.0]))
    path = tmp_path / "gallery.npz"

    gallery.save(path)
    loaded = Gallery.load(path)

    assert loaded.names == ["alice"]
    np.testing.assert_array_equal(loaded.embeddings[0], [1.0, 2.0, 3.0])
