"""Fast, offline tests: torch/transformers are replaced by tiny fakes so no 600 MB CLIP download is needed.
The fake "CLIP" embeds category and colour into separate dimensions, which is enough to check the engine's
own logic (indexing, tagging, blending, filtering). It says nothing about real CLIP accuracy."""
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CATS = ["shoe", "watch", "laptop", "backpack", "shirt"]
COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "black": (0, 0, 0), "white": (255, 255, 255)}
COLOR_NAMES = list(COLORS)


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a, dtype="float32")

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeInputs(dict):
    def to(self, _device):
        return self


def _vec(cat=None, color=None):
    v = np.zeros(10, dtype="float32")
    if cat is not None:
        v[CATS.index(cat)] = 1
    if color is not None:
        v[5 + COLOR_NAMES.index(color)] = 1
    return v


def image_for(cat, color):
    """A solid-colour image whose width encodes the category."""
    return Image.new("RGB", (10 + CATS.index(cat), 8), COLORS[color])


class FakeModel:
    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def to(self, _device):
        return self

    def get_text_features(self, text):
        words = text[0].lower().split()  # whole words: "colored" must not match "red"
        cat = next((c for c in CATS if c in words), None)
        color = next((c for c in COLOR_NAMES if c in words), None)
        return FakeTensor([_vec(cat, color) * 3.0])  # text vectors deliberately longer than image ones

    def get_image_features(self, img):
        im = img[0]
        cat = CATS[im.size[0] - 10]
        color = min(COLOR_NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(COLORS[n], im.getpixel((0, 0)))))
        return FakeTensor([_vec(cat, color)])


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def __call__(self, text=None, images=None, **_k):
        return FakeInputs(text=text) if text is not None else FakeInputs(img=[images])


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


@pytest.fixture(scope="session", autouse=True)
def _stub_ml_libs():
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: _NoGrad()
    tf = types.ModuleType("transformers")
    tf.CLIPModel, tf.CLIPProcessor = FakeModel, FakeProcessor
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers", "models.engine")}
    sys.modules.update(torch=torch, transformers=tf)
    sys.modules.pop("models.engine", None)
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture
def catalog(tmp_path):
    """One image per (category, colour) pair, saved as PNGs."""
    paths = []
    for c in CATS:
        for col in COLOR_NAMES:
            p = tmp_path / f"{col}_{c}.png"
            image_for(c, col).save(p)
            paths.append(str(p))
    return paths


@pytest.fixture
def engine():
    from models.engine import OmniSearchEngine
    return OmniSearchEngine()
