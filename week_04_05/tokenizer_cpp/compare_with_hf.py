"""Compare the compiled C++ tokenizer against Hugging Face's tokenizer.json implementation."""

import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer


CASES = [
    "Hello, world!",
    "你好，HPC！",
    "Qwen3 on H100: 16 concurrent requests, 4-bit KV cache.",
    "Line one.\nLine two with a tab:\tend.",
    "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: compare_with_hf.py PROBE_BINARY TOKENIZER_JSON PYTHON_MODULE_DIR"
        )
    binary, tokenizer_path, module_dir = sys.argv[1:]
    sys.path.insert(0, str(Path(module_dir).resolve()))
    from _qwen3_cpp_tokenizer import Qwen3Tokenizer as CppTokenizer

    result = subprocess.run(
        [binary, tokenizer_path], check=True, capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    if len(lines) != len(CASES) + 1:
        raise SystemExit(f"C++ probe returned {len(lines) - 1} rows, expected {len(CASES)}")

    hf = Tokenizer.from_file(tokenizer_path)
    cpp_vocab_size = int(lines[0])
    hf_vocab_size = hf.get_vocab_size(with_added_tokens=True)
    if cpp_vocab_size != hf_vocab_size:
        raise SystemExit(f"vocabulary size differs: C++={cpp_vocab_size}, HF={hf_vocab_size}")

    cpp = CppTokenizer(tokenizer_path)
    if cpp.vocab_size() != hf_vocab_size:
        raise SystemExit(
            f"Python extension vocabulary differs: C++={cpp.vocab_size()}, HF={hf_vocab_size}"
        )
    batch_ids = cpp.encode_batch(CASES)
    if len(batch_ids) != len(CASES):
        raise SystemExit(f"C++ Python extension returned {len(batch_ids)} batch rows")

    for i, (text, row) in enumerate(zip(CASES, lines[1:])):
        cpp_ids = [int(v) for v in row.split(",") if v]
        module_ids = cpp.encode(text)
        if module_ids != cpp_ids or batch_ids[i] != cpp_ids:
            raise SystemExit(f"C++ executable, extension, and batch encoding disagree at probe {i}")
        with_specials = hf.encode(text, add_special_tokens=True).ids
        without_specials = hf.encode(text, add_special_tokens=False).ids
        if cpp_ids not in (with_specials, without_specials):
            raise SystemExit(
                f"token IDs differ at probe {i}: C++={cpp_ids}, "
                f"HF(add_special_tokens=True)={with_specials}, "
                f"HF(add_special_tokens=False)={without_specials}"
            )
        if cpp_ids == with_specials == without_specials:
            mode = "both"
        elif cpp_ids == with_specials:
            mode = "with specials"
        else:
            mode = "without specials"
        print(f"probe {i}: exact HF token IDs ({mode})")
    print(f"PASS: C++ executable, Python extension, and HF agree on vocab and {len(CASES)} probes")


if __name__ == "__main__":
    main()
