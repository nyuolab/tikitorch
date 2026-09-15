"""Checked integer error-feedback carry/resume example; no model or hardware."""
import torch

from tikitorch.optim import fixed_point_ef_step


def main():
    weights = torch.tensor([10, -10], dtype=torch.int16)
    remainder = torch.zeros(2, dtype=torch.int32)
    gradient = torch.tensor([1, -1], dtype=torch.int32)
    k = 2
    # Three quarter-LSB contributions remain in optimizer state.
    for _ in range(3):
        weights, remainder = fixed_point_ef_step(weights, remainder, gradient, k=k)
    assert weights.tolist() == [10, -10]
    assert remainder.tolist() == [3, -3]
    checkpoint = {"weights": weights.clone(), "remainder": remainder.clone(), "k": k}
    # A checkpoint must preserve the remainder and its fixed denominator.
    weights, remainder = fixed_point_ef_step(
        checkpoint["weights"], checkpoint["remainder"], gradient, k=checkpoint["k"]
    )
    assert weights.tolist() == [9, -9]
    assert remainder.tolist() == [0, 0]
    print("Four fractional contributions preserved:", weights.tolist(), remainder.tolist())
    return weights, remainder


if __name__ == "__main__":
    main()
