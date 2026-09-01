"""TikiTorch quickstart — a tropical model in a familiar PyTorch loop.

Run:  python examples/quickstart.py

This default soft model and cross-entropy loss are an interoperability example,
not a multiplier-free training certificate.
"""
import torch
import tikitorch as tt
import tikitorch.optim as optim

model = tt.models.MorphSSM(d_model=64, n_layers=4, task_type="classification",
                           input_dim=1, num_classes=10)        # tropical SSM
opt = optim.PureTropicalSGD(model.parameters())                # shift-oriented update
x = torch.randn(8, 784, 1)                                     # (batch, seq_len, feat)
y = torch.randint(0, 10, (8,))                                 # class labels
loss = model.compute_loss(x, y)                               # cross-entropy
loss.backward(); opt.step()                                    # bit-shift update
print(f"loss={loss.item():.4f}  params={model.param_count}")
