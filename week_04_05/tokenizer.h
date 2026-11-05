#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tokenizers_cpp.h>

// Thin Qwen3 adapter around MLC's C++ binding for Hugging Face tokenizers.
// Load the tokenizer.json shipped with the exact Qwen3 model revision; a byte
// to integer mapping is not equivalent to Qwen3's BPE tokenizer.
class Qwen3Tokenizer {
 public:
  explicit Qwen3Tokenizer(const std::string& tokenizer_json_path) {
    std::ifstream input(tokenizer_json_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Cannot open tokenizer JSON: " + tokenizer_json_path);
    }
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json);
    if (!tokenizer_) {
      throw std::runtime_error("Failed to load Hugging Face tokenizer JSON");
    }
  }

  std::vector<int32_t> encode(const std::string& text) {
    return tokenizer_->Encode(text);
  }

  std::vector<std::vector<int32_t>> encode_batch(
      const std::vector<std::string>& texts) {
    return tokenizer_->EncodeBatch(texts);
  }

  std::string decode(const std::vector<int32_t>& ids) {
    return tokenizer_->Decode(ids);
  }

  size_t vocab_size() { return tokenizer_->GetVocabSize(); }

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};
