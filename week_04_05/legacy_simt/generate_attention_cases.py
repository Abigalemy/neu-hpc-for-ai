import sqlite3
import torch
import numpy as np
import pickle
import math
from qwen3_attention import Qwen3Attention

torch.manual_seed(42)

batch_size = 2
seq_len = 8
hidden_size = 1024
num_heads = 16

model = Qwen3Attention(hidden_size=hidden_size, num_heads=num_heads)
model.eval()

input_tensor = torch.randn(batch_size, seq_len, hidden_size)

with torch.no_grad():
    output_tensor = model(input_tensor)

weights_dict = {k: v.cpu() for k, v in model.state_dict().items()}
weights_bytes = pickle.dumps(weights_dict)

conn = sqlite3.connect("test_cases.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attention_test_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    weights_data BLOB,
    input_data BLOB,
    output_data BLOB,
    batch_size INTEGER,
    seq_len INTEGER,
    hidden_size INTEGER,
    num_heads INTEGER
)
""")

cursor.execute(
    """
    INSERT INTO attention_test_cases 
    (weights_data, input_data, output_data, batch_size, seq_len, hidden_size, num_heads) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (
        weights_bytes,
        input_tensor.numpy().tobytes(),
        output_tensor.numpy().tobytes(),
        batch_size,
        seq_len,
        hidden_size,
        num_heads
    )
)

conn.commit()
conn.close()

print("Attention test cases generated and saved to test_cases.db successfully.")