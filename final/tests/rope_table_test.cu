// rope_table_test <theta> <yarn_factor (0 = default RoPE)> <orig_max> <n_pos> <out.bin>
// Host-only: writes the engine's cos/sin tables (cos then sin, [n_pos, 64] bf16) so they can be compared with HF.
#include <cstdio>
#include <cstdlib>
#include "../engine/qwen3_rope_table.h"

int main(int argc, char** argv) {
    if (argc != 6) { std::fprintf(stderr, "usage: %s theta yarn_factor orig_max n_pos out.bin\n", argv[0]); return 1; }
    RopeParams p;
    p.theta = std::atof(argv[1]);
    double f = std::atof(argv[2]);
    p.yarn = f > 0;
    p.factor = f;
    p.orig_max = std::atoi(argv[3]);
    int n = std::atoi(argv[4]);
    auto t = rope_tables_bf16(p, n);
    FILE* o = std::fopen(argv[5], "wb");
    std::fwrite(t.data(), sizeof(__nv_bfloat16), t.size(), o);
    std::fclose(o);
    std::printf("wrote %zu bf16 values (%d positions, yarn=%d, factor=%g)\n", t.size(), n, (int)p.yarn, p.factor);
    return 0;
}
