# llm.cpp
add_library(nn nn.hpp)
target_link_libraries(nn
        absl::strings absl::log absl::check pthread)

add_library(gpt gpt.hpp)
target_link_libraries(gpt nn)

add_library(gpt2 gpt2.hpp)
target_link_libraries(gpt2 gpt)

add_library(optim optim.hpp)
target_link_libraries(optim nn)

add_executable(test_gpt2_cpu test_gpt2.cpp)
target_link_libraries(test_gpt2_cpu gpt2 optim)
target_compile_options(test_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(train_gpt2_cpu train_gpt2.cpp)
target_link_libraries(train_gpt2_cpu
        gpt2 optim
        profiler
)
target_compile_options(train_gpt2_cpu PRIVATE -Ofast -march=native)

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test nn GTest::gtest_main)

add_executable(optim_test optim_test.cpp)
target_link_libraries(optim_test nn GTest::gtest_main)

add_executable(gpt_test gpt_test.cpp)
target_link_libraries(gpt_test gpt GTest::gtest_main)

add_executable(gpt_optim gpt_optim.cpp)
target_link_libraries(gpt_optim gpt)

add_executable(test_eigen_cpu test_eigen_cpu.cpp)
target_link_libraries(test_eigen_cpu nn)
target_compile_options(test_eigen_cpu PRIVATE -Ofast -march=native)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)
find_package(CUDA)
if (CUDA_FOUND)
    add_library(nn_gpu nn.hpp)
    target_compile_definitions(nn_gpu PUBLIC EIGEN_USE_GPU)
    target_link_libraries(nn_gpu
            absl::strings absl::log absl::check
            ${CUDA_LIBRARIES}
    )

    add_library(gpt_gpu gpt.hpp)
    target_link_libraries(gpt_gpu
            nn_gpu
    )

    add_executable(test_eigen_gpu test_eigen_gpu.cu)
    target_compile_definitions(test_eigen_gpu PRIVATE EIGEN_USE_GPU)
    target_link_libraries(test_eigen_gpu
            nn_gpu
    )
    #    target_compile_options(test_eigen_gpu PRIVATE -Xcompiler=-Ofast,-march=native)

    # nn_test_gpu
    add_executable(nn_test_gpu nn_test.cu)
    target_link_libraries(nn_test_gpu
            nn_gpu
            GTest::gtest_main
    )

    # gpt_test_gpu
    add_executable(gpt_test_gpu gpt_test.cu)
    target_link_libraries(gpt_test_gpu
            gpt_gpu
            GTest::gtest_main
    )

    # gpt_optim_gpu
    add_executable(gpt_optim_gpu gpt_optim.cu)
    target_link_libraries(gpt_optim_gpu
            gpt_gpu
    )

    # train_gpt2_gpu
    add_executable(train_gpt2_gpu train_gpt2.cu)
    target_link_libraries(train_gpt2_gpu
            gpt_gpu
    )
    target_compile_options(train_gpt2_gpu PRIVATE -O3)
endif ()
