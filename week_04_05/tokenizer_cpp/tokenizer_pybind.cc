#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_qwen3_cpp_tokenizer, m) {
  m.doc() = "C++ Qwen3 tokenizer backed by MLC tokenizers-cpp";
  py::class_<Qwen3Tokenizer>(m, "Qwen3Tokenizer")
      .def(py::init<const std::string&>())
      .def("encode", &Qwen3Tokenizer::encode)
      .def("encode_batch", &Qwen3Tokenizer::encode_batch)
      .def("decode", &Qwen3Tokenizer::decode)
      .def("vocab_size", &Qwen3Tokenizer::vocab_size);
}
