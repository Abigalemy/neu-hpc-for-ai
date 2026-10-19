import sqlite3
import torch
import numpy as np
import pickle
from qwen3_attention import Qwen3Attention

conn = sqlite3.connect("test_cases.db")
cursor = conn.cursor()

cursor.execute(
    "SELECT weights_data, input_data, output_data, batch_size, seq_len, hidden_size, num_heads FROM attention_test_cases LIMIT 1"
)
row = cursor.fetchone()
conn.close()

assert row is not None, "No attention test cases found in test_cases.db!"

weights_bytes, input_bytes, expected_output_bytes, batch_size, seq_len, hidden_size, num_heads = row

weights_dict = pickle.loads(weights_bytes)
input_tensor = torch.from_numpy(np.frombuffer(input_bytes, dtype=np.float32).copy().reshape(batch_size, seq_len, hidden_size))
expected_output = torch.from_numpy(np.frombuffer(expected_output_bytes, dtype=np.float32).copy().reshape(batch_size, seq_len, hidden_size))

model = Qwen3Attention(hidden_size=hidden_size, num_heads=num_heads)
model.load_state_dict(weights_dict)
model.eval()

with torch.no_grad():
    actual_output = model(input_tensor)

max_diff = torch.max(torch.abs(actual_output - expected_output)).item()
print("Attention verification max difference:", max_diff)
assert max_diff < 1e-5, f"Verification failed with max diff {max_diff}"
print("Attention local test passed successfully!")