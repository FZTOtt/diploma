import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from attention.models.generator import Generator

def test_generator_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator().to(device)
    cover = torch.randn(2, 3, 256, 256).to(device)   # батч из 2 картинок
    secret = torch.randn(2, 3, 256, 256).to(device)

    output = model(cover, secret)

    assert output.shape == (2, 3, 256, 256), f"Unexpected output shape: {output.shape}"
    print("✅ Generator forward pass OK. Output shape:", output.shape)

if __name__ == "__main__":
    test_generator_forward()
