import torch
import torch.nn as nn
import math

class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)

if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    hidden_size = 1024

    attn = Qwen3Attention(hidden_size=hidden_size, num_heads=16)
    attn.eval()

    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    with torch.no_grad():
        output = attn(dummy_input)

    print("Qwen3Attention input shape:", dummy_input.shape)
    print("Qwen3Attention output shape:", output.shape)
    assert output.shape == dummy_input.shape
    print("PyTorch Attention baseline sanity check passed.")