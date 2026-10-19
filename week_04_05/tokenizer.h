#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

class SimpleQwenTokenizer {
public:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    int eos_token_id = 151643;
    int pad_token_id = 151643;

    SimpleQwenTokenizer() {
        vocab["<|endoftext|>"] = 151643;
        vocab["<|im_start|>"] = 151644;
        vocab["<|im_end|>"] = 151645;
        inv_vocab[151643] = "<|endoftext|>";
        inv_vocab[151644] = "<|im_start|>";
        inv_vocab[151645] = "<|im_end|>";

        for (int i = 0; i < 256; ++i) {
            std::string s(1, (char)i);
            vocab[s] = i;
            inv_vocab[i] = s;
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        for (size_t i = 0; i < text.size(); ++i) {
            std::string ch(1, text[i]);
            auto it = vocab.find(ch);
            if (it != vocab.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back((unsigned char)text[i]);
            }
        }
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string text = "";
        for (int token : tokens) {
            auto it = inv_vocab.find(token);
            if (it != inv_vocab.end()) {
                text += it->second;
            } else if (token >= 0 && token < 256) {
                text += (char)token;
            }
        }
        return text;
    }
};