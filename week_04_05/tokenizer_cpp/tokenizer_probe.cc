#include <iostream>
#include <string>
#include <vector>

#include "tokenizer.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: qwen3_tokenizer_probe /path/to/tokenizer.json\n";
    return 2;
  }

  try {
    Qwen3Tokenizer tokenizer(argv[1]);
    const std::vector<std::string> probes = {
        "Hello, world!",
        "你好，HPC！",
        "Qwen3 on H100: 16 concurrent requests, 4-bit KV cache.",
        "Line one.\nLine two with a tab:\tend.",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
    };

    const auto batch_ids = tokenizer.encode_batch(probes);
    if (batch_ids.size() != probes.size()) {
      std::cerr << "EncodeBatch returned the wrong number of rows\n";
      return 1;
    }

    std::cout << tokenizer.vocab_size() << '\n';
    for (std::size_t i = 0; i < probes.size(); ++i) {
      const auto single_ids = tokenizer.encode(probes[i]);
      if (single_ids != batch_ids[i]) {
        std::cerr << "single and batch encoders disagree at probe " << i << '\n';
        return 1;
      }
      for (std::size_t j = 0; j < single_ids.size(); ++j) {
        if (j != 0) std::cout << ',';
        std::cout << single_ids[j];
      }
      std::cout << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "tokenizer probe failed: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
