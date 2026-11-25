// Weight loading without PyTorch: parse safetensors files and config.json, copy every tensor to the GPU ONCE.
//
// safetensors layout: 8-byte little-endian header size N, then N bytes of JSON
//   {"name": {"dtype": "BF16", "shape": [..], "data_offsets": [begin, end]}, ..., "__metadata__": {...}},
// then the raw tensor bytes (offsets are relative to the end of the header).
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

// ---- a minimal JSON reader (objects, arrays, strings, numbers, true/false/null), enough for the two file types ----
struct JV {
    enum Kind { NUL, NUM, STR, ARR, OBJ } t = NUL;
    double num = 0;
    std::string str;
    std::vector<JV> arr;
    std::vector<std::pair<std::string, JV>> obj;
    const JV* get(const char* k) const {
        for (auto& p : obj) if (p.first == k) return &p.second;
        return nullptr;
    }
};

static void json_ws(const char*& p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p; }

static JV json_parse(const char*& p) {
    json_ws(p);
    JV v;
    if (*p == '{') {
        v.t = JV::OBJ; ++p; json_ws(p);
        if (*p == '}') { ++p; return v; }
        while (true) {
            json_ws(p); JV k = json_parse(p); json_ws(p); ++p;  // ':'
            JV val = json_parse(p);
            v.obj.emplace_back(k.str, std::move(val));
            json_ws(p);
            if (*p == ',') { ++p; continue; }
            ++p; break;  // '}'
        }
    } else if (*p == '[') {
        v.t = JV::ARR; ++p; json_ws(p);
        if (*p == ']') { ++p; return v; }
        while (true) {
            v.arr.push_back(json_parse(p)); json_ws(p);
            if (*p == ',') { ++p; continue; }
            ++p; break;  // ']'
        }
    } else if (*p == '"') {
        v.t = JV::STR; ++p;
        while (*p != '"') { if (*p == '\\') ++p; v.str += *p++; }
        ++p;
    } else if (!std::strncmp(p, "true", 4)) { v.t = JV::NUM; v.num = 1; p += 4; }
    else if (!std::strncmp(p, "false", 5)) { v.t = JV::NUM; v.num = 0; p += 5; }
    else if (!std::strncmp(p, "null", 4)) { v.t = JV::NUL; p += 4; }
    else { char* end; v.t = JV::NUM; v.num = std::strtod(p, &end); p = end; }
    return v;
}

struct DevTensor {
    void* ptr = nullptr;
    std::vector<int64_t> shape;
    size_t bytes = 0;
};

struct Qwen3Config {
    int hidden = 0, inter = 0, layers = 0, hq = 0, hkv = 0, head_dim = 128, vocab = 0;
    float eps = 1e-6f;
    double rope_theta = 1e6;
    bool tie_embeddings = true;
    bool unsupported_rope_scaling = false;   // a rope_scaling type other than YaRN: the engine refuses to run
    bool yarn = false;                       // YaRN from the config (or the QWEN3_YARN override, see the engine)
    double yarn_factor = 0, yarn_beta_fast = 32, yarn_beta_slow = 1, yarn_attention_factor = -1;
    int yarn_orig_max = 32768;
    // Qwen3-MoE (Qwen3-30B-A3B: 128 experts, top-8, expert width 768). experts == 0: dense model. Note that the MoE
    // config also has intermediate_size (6144, for dense MLP layers, which Qwen3-30B-A3B does not have): the experts
    // use moe_intermediate_size.
    int experts = 0, top_k = 0, moe_inter = 0, sparse_step = 1;
    bool norm_topk = true;
    bool has_mlp_only_layers = false;        // dense layers inside an MoE model: not supported (none in 30B-A3B)
    bool is_moe_layer(int l) const { return experts > 0 && (l + 1) % sparse_step == 0; }
};

static std::string slurp(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::string s(n, '\0');
    if (std::fread(&s[0], 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f);
    return s;
}

static Qwen3Config load_config(const std::string& dir) {
    std::string text = slurp(dir + "/config.json");
    const char* p = text.c_str();
    JV j = json_parse(p);
    auto num = [&](const char* k, double dflt) { const JV* v = j.get(k); return v && v->t == JV::NUM ? v->num : dflt; };
    Qwen3Config c;
    c.hidden = (int)num("hidden_size", 0);
    c.inter = (int)num("intermediate_size", 0);
    c.layers = (int)num("num_hidden_layers", 0);
    c.hq = (int)num("num_attention_heads", 0);
    c.hkv = (int)num("num_key_value_heads", 0);
    c.head_dim = (int)num("head_dim", c.hidden / c.hq);
    c.vocab = (int)num("vocab_size", 0);
    c.eps = (float)num("rms_norm_eps", 1e-6);
    c.rope_theta = num("rope_theta", 1e6);
    c.tie_embeddings = num("tie_word_embeddings", 1) != 0;
    c.experts = (int)num("num_experts", 0);
    c.top_k = (int)num("num_experts_per_tok", 0);
    c.moe_inter = (int)num("moe_intermediate_size", 0);
    c.sparse_step = (int)num("decoder_sparse_step", 1);
    c.norm_topk = num("norm_topk_prob", 1) != 0;   // json_parse reads true / false as 1 / 0
    if (const JV* ml = j.get("mlp_only_layers")) c.has_mlp_only_layers = ml->t == JV::ARR && !ml->arr.empty();
    const JV* rs = j.get("rope_scaling");
    if (!rs || rs->t != JV::OBJ) rs = j.get("rope_parameters");   // newer config layout
    if (rs && rs->t == JV::OBJ) {
        const JV* ty = rs->get("rope_type") ? rs->get("rope_type") : rs->get("type");
        std::string type = ty && ty->t == JV::STR ? ty->str : "default";
        if (type == "yarn") {
            c.yarn = true;
            auto rnum = [&](const char* k, double d) { const JV* v = rs->get(k); return v && v->t == JV::NUM ? v->num : d; };
            c.yarn_factor = rnum("factor", 1.0);
            c.yarn_orig_max = (int)rnum("original_max_position_embeddings", 32768);
            c.yarn_beta_fast = rnum("beta_fast", 32);
            c.yarn_beta_slow = rnum("beta_slow", 1);
            c.yarn_attention_factor = rnum("attention_factor", -1);
        } else if (type != "default") {
            c.unsupported_rope_scaling = true;
        }
    }
    return c;
}

// Where a tensor goes while loading: DEFAULT = its own device allocation in the returned map; PLACED = copied to the
// device address the callback stored in *dst (e.g. its slot in a packed [experts, ...] buffer; the callback checks the
// size); SKIP = not loaded at all.
enum class Placement { DEFAULT, PLACED, SKIP };
using PlaceFn = std::function<Placement(const std::string& name, const std::vector<int64_t>& shape, size_t bytes, void** dst)>;

// Loads every *.safetensors file in `dir` to the GPU (BF16 only), once at start-up, straight from the mmap'ed file
// (no second copy on the host or on the device).
static std::map<std::string, DevTensor> load_weights_to_gpu(const std::string& dir, const PlaceFn& place = nullptr) {
    std::vector<std::string> files;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) files.push_back(dir + "/" + n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "no .safetensors files in %s\n", dir.c_str()); std::exit(2); }

    std::map<std::string, DevTensor> out;
    for (auto& path : files) {
        int fd = open(path.c_str(), O_RDONLY);
        struct stat st; fstat(fd, &st);
        if (fd < 0 || fstat(fd, &st) != 0 || st.st_size < 8) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
        void* mp = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mp == MAP_FAILED) { std::fprintf(stderr, "mmap of %s failed\n", path.c_str()); std::exit(2); }
        const uint8_t* base = (const uint8_t*)mp;
        uint64_t hlen; std::memcpy(&hlen, base, 8);
        if (hlen > (uint64_t)st.st_size - 8) { std::fprintf(stderr, "%s: bad header length\n", path.c_str()); std::exit(2); }
        std::string header((const char*)base + 8, hlen);
        const char* p = header.c_str();
        JV j = json_parse(p);
        const uint8_t* data = base + 8 + hlen;
        for (auto& kv : j.obj) {
            if (kv.first == "__metadata__") continue;
            const JV& t = kv.second;
            if (t.get("dtype")->str != "BF16") { std::fprintf(stderr, "%s: only BF16 supported\n", kv.first.c_str()); std::exit(2); }
            size_t b = (size_t)t.get("data_offsets")->arr[0].num, e = (size_t)t.get("data_offsets")->arr[1].num;
            if (b > e || e > (size_t)st.st_size - 8 - hlen) {
                std::fprintf(stderr, "%s: data_offsets of %s outside the file\n", path.c_str(), kv.first.c_str());
                std::exit(2);
            }
            DevTensor dt;
            for (auto& s : t.get("shape")->arr) dt.shape.push_back((int64_t)s.num);
            dt.bytes = e - b;
            void* dst = nullptr;
            Placement where = place ? place(kv.first, dt.shape, dt.bytes, &dst) : Placement::DEFAULT;
            if (where == Placement::SKIP) continue;
            if (where == Placement::PLACED) {
                if (cudaMemcpy(dst, data + b, dt.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                    std::fprintf(stderr, "copy of %s to its packed slot failed\n", kv.first.c_str());
                    std::exit(2);
                }
                continue;
            }
            if (cudaMalloc(&dt.ptr, dt.bytes) != cudaSuccess) { std::fprintf(stderr, "out of GPU memory loading %s\n", kv.first.c_str()); std::exit(2); }
            if (cudaMemcpy(dt.ptr, data + b, dt.bytes, cudaMemcpyHostToDevice) != cudaSuccess) std::exit(2);
            out[kv.first] = dt;
        }
        munmap((void*)base, st.st_size);
        close(fd);
    }
    return out;
}
