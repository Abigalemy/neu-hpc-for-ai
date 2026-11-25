// CPU-only check of the Part 2 plan: the whole dispatch / compute / combine pipeline simulated on the host with the
// same plan the GPU program uses (the all-to-all is a copy between the two simulated workers). No CUDA needed:
//   c++ -std=c++20 -O2 test_plan.cpp -o test_plan && ./test_plan
#include <cassert>
#include <cstdio>
#include "a2a_plan.h"

int main() {
    std::vector<std::vector<Token>> tok(WORLD);
    std::vector<Plan> plan(WORLD);
    std::vector<std::vector<float>> send(WORLD);   // permuted rows, [n, D]
    for (int r = 0; r < WORLD; ++r) {
        tok[r] = tokens_of(r);
        plan[r] = make_plan(tok[r]);
        for (int i : plan[r].perm)
            for (int d = 0; d < D; ++d) send[r].push_back(embed(tok[r][i].word, d));
    }
    // the figure's permutation column
    const char* want_perm[WORLD][3] = {{"hot", "Today", "is"}, {"feel", "I", "happy"}};
    for (int r = 0; r < WORLD; ++r)
        for (int j = 0; j < 3; ++j) assert(tok[r][plan[r].perm[j]].word == want_perm[r][j]);

    // all-to-all dispatch: worker dst receives, from each src in rank order, src's block for dst
    std::vector<std::vector<float>> recv(WORLD);
    std::vector<std::vector<int>> recv_expert(WORLD), recv_count(WORLD, std::vector<int>(WORLD));
    for (int dst = 0; dst < WORLD; ++dst)
        for (int src = 0; src < WORLD; ++src) {
            int n = plan[src].send_count[dst], o = plan[src].send_off[dst];
            recv_count[dst][src] = n;
            recv[dst].insert(recv[dst].end(), send[src].begin() + (size_t)o * D, send[src].begin() + (size_t)(o + n) * D);
            recv_expert[dst].insert(recv_expert[dst].end(), plan[src].perm_expert.begin() + o, plan[src].perm_expert.begin() + o + n);
        }
    // computation: every received row goes through its (local) expert
    for (int r = 0; r < WORLD; ++r)
        for (size_t j = 0; j < recv_expert[r].size(); ++j) {
            assert(recv_expert[r][j] / EXPERTS_PER_RANK == r);   // only local experts arrive here
            for (int d = 0; d < D; ++d) recv[r][j * D + d] = expert_fn(recv_expert[r][j], recv[r][j * D + d]);
        }
    // all-to-all combine (the inverse): src gets back its block from every dst, in the order it sent them
    std::vector<std::vector<float>> back(WORLD);
    for (int src = 0; src < WORLD; ++src) {
        back[src].resize(send[src].size());
        for (int dst = 0; dst < WORLD; ++dst) {
            int skip = 0;
            for (int s = 0; s < src; ++s) skip += recv_count[dst][s];
            std::copy(recv[dst].begin() + (size_t)skip * D, recv[dst].begin() + (size_t)(skip + plan[src].send_count[dst]) * D,
                      back[src].begin() + (size_t)plan[src].send_off[dst] * D);
        }
    }
    // un-permutation + scale, then compare with the expected output of every token
    int bad = 0;
    for (int r = 0; r < WORLD; ++r) {
        std::vector<float> out(tok[r].size() * D);
        for (size_t j = 0; j < plan[r].perm.size(); ++j)
            for (int d = 0; d < D; ++d) out[plan[r].perm[j] * D + d] = plan[r].perm_weight[j] * back[r][j * D + d];
        for (size_t i = 0; i < tok[r].size(); ++i)
            for (int d = 0; d < D; ++d) bad += out[i * D + d] != expected(tok[r][i], d);
    }
    std::printf(bad ? "FAIL: %d values differ\n" : "PLAN OK: the simulated pipeline gives the expected output exactly\n", bad);
    return bad != 0;
}
