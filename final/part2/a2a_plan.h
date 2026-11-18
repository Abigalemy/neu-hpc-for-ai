// Part 2 (all-to-all mock): the host-side plan of an expert-parallel MoE layer, plain C++ so it can be tested without a
// GPU (test_plan.cpp). The tokens and the routing are hard-wired to the figure of the Week 8 handout:
//
//   worker 0: "Today" "is" "hot"      experts 0, 1 live on worker 0
//   worker 1: "I" "feel" "happy"      experts 2, 3 live on worker 1
//   gate (top-1): Today->3  is->3  hot->1  |  I->2  feel->0  happy->3
//
// Steps (the figure's columns): routing -> permutation (sort the local tokens by expert, so that the tokens for one
// worker are contiguous) -> all-to-all dispatch -> expert computation -> all-to-all combine (the inverse) ->
// un-permutation -> scale by the gate weight. The experts are mocked as e_k(x) = (k + 1) * x + k, elementwise.
#pragma once
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

constexpr int WORLD = 2, EXPERTS = 4, EXPERTS_PER_RANK = EXPERTS / WORLD, D = 128;   // D: hidden size of a token

struct Token { std::string word; int expert; float weight; };

// the hard-wired batch of each worker; weights are binary fractions so that every result is exact in fp32
inline std::vector<Token> tokens_of(int rank) {
    if (rank == 0) return {{"Today", 3, 0.75f}, {"is", 3, 0.5f}, {"hot", 1, 0.25f}};
    return {{"I", 2, 0.5f}, {"feel", 0, 0.75f}, {"happy", 3, 1.0f}};
}

// token embedding: row of D values that identifies the token (word code + position in the row)
inline float embed(const std::string& w, int d) {
    int code = 0;
    for (char c : w) code = (code * 31 + c) % 97;
    return (float)code + d / 128.0f;
}

inline float expert_fn(int k, float x) { return (k + 1) * x + k; }   // the mocked expert k

struct Plan {
    std::vector<int> perm;                  // permuted row j comes from local token perm[j] (stable sort by expert)
    std::vector<int> send_count, send_off;  // rows for each destination rank, contiguous after the permutation
    std::vector<int> perm_expert;           // expert of permuted row j (sent along with the rows)
    std::vector<float> perm_weight;         // gate weight of permuted row j (stays local, used when scaling)
};

inline Plan make_plan(const std::vector<Token>& t) {
    Plan p;
    p.perm.resize(t.size());
    std::iota(p.perm.begin(), p.perm.end(), 0);
    std::stable_sort(p.perm.begin(), p.perm.end(), [&](int a, int b) { return t[a].expert < t[b].expert; });
    p.send_count.assign(WORLD, 0);
    for (int i : p.perm) {
        p.send_count[t[i].expert / EXPERTS_PER_RANK]++;
        p.perm_expert.push_back(t[i].expert);
        p.perm_weight.push_back(t[i].weight);
    }
    p.send_off.assign(WORLD, 0);
    for (int r = 1; r < WORLD; ++r) p.send_off[r] = p.send_off[r - 1] + p.send_count[r - 1];
    return p;
}

// what the layer must return for local token i: weight * expert(embedding)
inline float expected(const Token& t, int d) { return t.weight * expert_fn(t.expert, embed(t.word, d)); }
