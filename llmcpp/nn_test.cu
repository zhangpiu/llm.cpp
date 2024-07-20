#include "gtest/gtest.h"
#include "nn.hpp"

using nn::DT_FLOAT;
using nn::Parameter;

TEST(Random, UniformFill) {
  nn::ManualSeed(42);
  std::vector<float> expected_num = {0.882269, 0.915004, 0.382864, 0.959306,
                                     0.390448, 0.600895, 0.256572, 0.793641,
                                     0.940771, 0.133186};
  auto& gpu_device = nn::g_device;
  size_t length = expected_num.size();
  float* d_num = (float*)gpu_device.allocate(sizeof(float) * length);
  nn::UniformFill(absl::MakeSpan(d_num, length));
  std::vector<float> num(length);
  gpu_device.memcpyDeviceToHost(num.data(), d_num, sizeof(float) * length);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }
  gpu_device.deallocate(d_num);
}

TEST(Random, NormalFill) {
  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  std::vector<float> expected_num = {0.336690,  0.128809,  0.234462, 0.230333,
                                     -1.122856, -0.186328, 2.208201, -0.637997,
                                     0.461657,  0.267351};
  size_t length = expected_num.size();
  float* d_num = (float*)gpu_device.allocate(sizeof(float) * length);
  nn::NormalFill(absl::MakeSpan(d_num, length));
  std::vector<float> num(length);
  gpu_device.memcpyDeviceToHost(num.data(), d_num, sizeof(float) * length);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }

  expected_num = {-0.758131, 1.078318,  0.800801,  1.680621,  0.355860,
                  -0.686623, -0.493356, 0.241488,  -0.231624, 0.041759,
                  -0.251575, 0.859859,  -0.309727, -0.395710, 0.803409,
                  -0.621595, 0.318880,  -0.424519, 0.305721,  -0.774593,
                  0.034912,  0.321103,  1.573600,  -0.845467, -1.274151,
                  2.122785,  -1.234653, -0.487914, -1.418060, 0.896268,
                  0.049905,  2.266718};
  length = expected_num.size();
  float* d_num2 = (float*)gpu_device.allocate(sizeof(float) * length);
  nn::NormalFill(absl::MakeSpan(d_num2, length));
  num.resize(length);
  gpu_device.memcpyDeviceToHost(num.data(), d_num2, sizeof(float) * length);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }

  gpu_device.deallocate(d_num);
  gpu_device.deallocate(d_num2);
}

TEST(Random, KaimingUniformFill) {
  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  int in_features = 4, out_features = 3, num_samples = 12;
  std::vector<float> expected_num = {0.382269,  0.415004,  -0.117136, 0.459306,
                                     -0.109552, 0.100895,  -0.243428, 0.293641,
                                     0.440771,  -0.366814, 0.434598,  0.093580};
  float* d_num = (float*)gpu_device.allocate(sizeof(float) * num_samples);
  nn::KaimingUniformFill(absl::MakeSpan(d_num, num_samples), in_features);
  std::vector<float> num(num_samples);
  gpu_device.memcpyDeviceToHost(num.data(), d_num, sizeof(float) * num_samples);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }

  gpu_device.deallocate(d_num);
}

TEST(MatMul, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 2)
x1 = nn.Parameter(x1)
x2 = nn.Parameter(x2)
y = torch.matmul(x1, x2)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  int M = 4, N = 3, K = 2;
  auto x1_gpu = nn::Parameter(nn::DT_FLOAT, M * N);
  auto x2_gpu = nn::Parameter(nn::DT_FLOAT, N * K);
  nn::NormalFill(x1_gpu.span<float>());
  nn::NormalFill(x2_gpu.span<float>());

  // forward
  auto y_gpu = nn::Parameter(nn::DT_FLOAT, M * K);
  auto x1_2d = x1_gpu.const_matrix<float>(M, N);
  auto x2_2d = x2_gpu.const_matrix<float>(N, K);
  auto y_2d = y_gpu.matrix<float>(M, K);
  nn::MatMul<float>::Forward(x1_2d, x2_2d, y_2d);

  std::vector<float> expected_y = {0.556428, -0.253943, 1.119845, -1.617147,
                                   3.693071, -3.965338, 0.837917, 0.722053};
  std::vector<float> y_cpu(expected_y.size(), 100.f);
  gpu_device.memcpyDeviceToHost(y_cpu.data(), y_gpu.data<float>(),
                                sizeof(float) * y_cpu.size());
  gpu_device.synchronize();

  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y_cpu[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad_cpu(y_2d.size(), 1.0);
  std::vector<float> x1_grad_cpu(x1_2d.size(), 0.0),
      x2_grad_cpu(x2_2d.size(), 0.0);
  auto y_grad_gpu = Parameter(DT_FLOAT, y_2d.size());
  auto x1_grad_gpu = Parameter(DT_FLOAT, x1_2d.size());
  auto x2_grad_gpu = Parameter(DT_FLOAT, x2_2d.size());
  gpu_device.memcpyHostToDevice(y_grad_gpu.data<float>(), y_grad_cpu.data(),
                                sizeof(float) * y_grad_cpu.size());

  auto y_grad_2d = y_grad_gpu.const_matrix<float>(M, K);
  auto x1_grad_2d = x1_grad_gpu.matrix<float>(M, N);
  auto x2_grad_2d = x2_grad_gpu.matrix<float>(N, K);
  nn::MatMul<float>::Backward(x1_2d, x2_2d, y_grad_2d, x1_grad_2d, x2_grad_2d);

  std::vector<float> expected_x1_grad = {
      -0.579509, -0.030988, 2.139325, -0.579509, -0.030988, 2.139325,
      -0.579509, -0.030988, 2.139325, -0.579509, -0.030988, 2.139325};
  std::vector<float> expected_x2_grad = {3.042576,  3.042576, -1.097139,
                                         -1.097139, 1.319149, 1.319149};

  gpu_device.memcpyDeviceToHost(x1_grad_cpu.data(), x1_grad_gpu.data<float>(),
                                sizeof(float) * x1_grad_cpu.size());
  gpu_device.memcpyDeviceToHost(x2_grad_cpu.data(), x2_grad_gpu.data<float>(),
                                sizeof(float) * x2_grad_cpu.size());
  gpu_device.synchronize();
  for (size_t i = 0; i < expected_x1_grad.size(); ++i) {
    EXPECT_NEAR(expected_x1_grad[i], x1_grad_cpu[i], 1e-5);
  }
  for (size_t i = 0; i < expected_x2_grad.size(); ++i) {
    EXPECT_NEAR(expected_x2_grad[i], x2_grad_cpu[i], 1e-5);
  }
}
