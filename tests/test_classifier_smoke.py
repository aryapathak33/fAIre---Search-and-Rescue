"""
Smoke test for the classifier model construction.

Skips automatically if torch/torchvision aren't installed, so the test suite still
passes on a bare machine. Once you have torch (or run in Colab), this verifies the
classifier head is wired correctly and produces the right output shape.

    pytest -v
"""

import pytest

torch = pytest.importorskip("torch")          # skip whole file if torch missing
pytest.importorskip("torchvision")


def test_classifier_output_shape():
    from torch import nn
    from torchvision.models import mobilenet_v2

    num_classes = 2
    model = mobilenet_v2(weights=None)         # no download; just structure
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.eval()

    dummy = torch.randn(4, 3, 224, 224)        # batch of 4 fake images
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (4, num_classes)       # 4 images -> 4 rows of class scores
