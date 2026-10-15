import sqlite3
import torch
import torch.nn as nn
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=256, intermediate_size=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

conn = sqlite3.connect("test_cases.db")
cursor = conn.cursor()

cursor.execute("SELECT input_data, output_data FROM mlp_test_cases")
row = cursor.fetchone()

if row:
    input_bytes, expected_output_bytes = row
    
    input_tensor = torch.from_numpy(np.frombuffer(input_bytes, dtype=np.float32).copy().reshape(2, 4, 256))
    expected_output = torch.from_numpy(np.frombuffer(expected_output_bytes, dtype=np.float32).copy().reshape(2, 4, 256))

    model = SimpleMLP()
    model.eval()

    with torch.no_grad():
        actual_output = model(input_tensor)

    max_diff = torch.max(torch.abs(actual_output - expected_output)).item()
    print("Verification max difference:", max_diff)
    
    if max_diff < 1e-5:
        print("Test passed successfully!")
    else:
        print("Test failed!")

conn.close()