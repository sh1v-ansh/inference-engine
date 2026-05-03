cmake_minimum_required(VERSION 3.26)
project(inference_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# ── CUDA extension ────────────────────────────────────────────────────────────
set(CUDA_SRCS
    src/cuda/paged_attention.cu
    src/cuda/attention_kernels.cu
    src/cuda/cache_ops.cu
)

pybind11_add_module(_kernels ${CUDA_SRCS} src/cuda/bindings.cpp)

target_include_directories(_kernels PRIVATE
    src/cuda/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(_kernels PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

target_compile_options(_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --ptxas-options=-v
        -Xcompiler=-fPIC
    >
)

set_target_properties(_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/engine
)
