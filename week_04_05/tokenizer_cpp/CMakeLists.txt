cmake_minimum_required(VERSION 3.19)
project(qwen3_cpp_tokenizer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Git REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
include(FetchContent)

# Upstream commit identified as building successfully in mlc-ai/tokenizers-cpp#94.
# The v0.1.1 release tag has a known Rust dependency mismatch. Fetch pinned submodules too.
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "Build HF JSON tokenizer support only" FORCE)
FetchContent_Declare(tokenizers_cpp_upstream
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG acbdc5a2
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp_upstream)

add_executable(qwen3_tokenizer_probe tokenizer_probe.cc)
target_include_directories(qwen3_tokenizer_probe PRIVATE "${CMAKE_CURRENT_LIST_DIR}/..")
target_link_libraries(qwen3_tokenizer_probe PRIVATE tokenizers_cpp)

pybind11_add_module(_qwen3_cpp_tokenizer tokenizer_pybind.cc)
target_include_directories(_qwen3_cpp_tokenizer PRIVATE "${CMAKE_CURRENT_LIST_DIR}/..")
target_link_libraries(_qwen3_cpp_tokenizer PRIVATE tokenizers_cpp)
set(QWEN3_TOKENIZER_PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Python extension output directory")
set_target_properties(_qwen3_cpp_tokenizer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${QWEN3_TOKENIZER_PYTHON_OUTPUT_DIR}"
)
