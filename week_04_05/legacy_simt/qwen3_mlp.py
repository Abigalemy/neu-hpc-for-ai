import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

if __name__ == "__main__":
    torch.manual_seed(42)
    
    hidden_size = 1024
    intermediate_size = 2816
    batch_size = 2
    seq_len = 8

    mlp = Qwen3MLP(hidden_size, intermediate_size)
    mlp.eval()

    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    with torch.no_grad():
        output = mlp(dummy_input)

    print("Qwen3MLP input shape:", dummy_input.shape)
    print("Qwen3MLP output shape:", output.shape)
    assert output.shape == dummy_input.shape
    print("PyTorch baseline sanity check passed.")