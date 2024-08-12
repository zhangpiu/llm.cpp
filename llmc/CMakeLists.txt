
# OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
    add_compile_definitions(OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif ()

# llm.c
add_executable(train_gpt2 ../train_gpt2.c)
target_link_libraries(train_gpt2 m ${OpenMP_CXX_LIBRARIES})

add_executable(test_gpt2 ../test_gpt2.c)
target_link_libraries(test_gpt2 m ${OpenMP_CXX_LIBRARIES})

if (CUDA_FOUND)
    add_compile_definitions(ENABLE_FP32)
    add_executable(train_gpt2cu train_gpt2.cu)
    set_target_properties(train_gpt2cu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "61;70;75"
    )
    target_link_libraries(train_gpt2cu ${CUDA_LIBRARIES} cublas cublasLt)
endif ()
