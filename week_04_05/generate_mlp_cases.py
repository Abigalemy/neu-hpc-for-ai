import sqlite3
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=256, intermediate_size=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

torch.manual_seed(42)
model = SimpleMLP()
model.eval()

batch_size = 2
seq_len = 4
hidden_size = 256
input_tensor = torch.randn(batch_size, seq_len, hidden_size)

with torch.no_grad():
    output_tensor = model(input_tensor)

conn = sqlite3.connect("test_cases.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS mlp_test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_data BLOB,
    output_data BLOB
)
""")

cursor.execute(
    "INSERT INTO mlp_test_cases (input_data, output_data) VALUES (?, ?)",
    (input_tensor.numpy().tobytes(), output_tensor.numpy().tobytes())
)

conn.commit()
conn.close()