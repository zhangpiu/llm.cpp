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
  gpu_device.synchronize();
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
  gpu_device.synchronize();
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
  gpu_device.synchronize();
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
  gpu_device.synchronize();
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
  gpu_device.synchronize();
  nn::MatMul::Forward(x1_2d, x2_2d, y_2d);

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
  gpu_device.synchronize();

  auto y_grad_2d = y_grad_gpu.const_matrix<float>(M, K);
  auto x1_grad_2d = x1_grad_gpu.matrix<float>(M, N);
  auto x2_grad_2d = x2_grad_gpu.matrix<float>(N, K);
  nn::MatMul::Backward(x1_2d, x2_2d, y_grad_2d, x1_grad_2d, x2_grad_2d);

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

TEST(Linear, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Linear(3, 2)
x = torch.randn(4, 3)
x = nn.Parameter(x)
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  int B = 4, in_features = 3, out_features = 2;
  nn::Linear m(in_features, out_features, true);
  std::vector<float> x = {-1.122856, -0.186328, 2.208201,  -0.637997,
                          0.461657,  0.267351,  0.534905,  0.809357,
                          1.110290,  -1.689799, -0.988960, 0.957972};
  Parameter d_x(DT_FLOAT, x.size());
  gpu_device.memcpyHostToDevice(d_x.data<float>(), x.data(),
                                sizeof(float) * x.size());
  gpu_device.synchronize();

  // forward
  Parameter d_y(DT_FLOAT, 8);
  auto xm = d_x.const_matrix<float>(B, in_features);
  auto ym = d_y.matrix<float>(B, out_features);
  m.Forward(xm, ym);

  std::vector<float> expected_y = {-1.164687, 0.024384, -0.377635, -0.026553,
                                   0.192698,  0.649730, -1.630462, -0.320424};
  std::vector<float> y(8);
  gpu_device.memcpyDeviceToHost(y.data(), d_y.data<float>(),
                                sizeof(float) * y.size());
  gpu_device.synchronize();
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-4);
  }

  // backward
  Parameter d_y_grad(DT_FLOAT, y.size());
  Parameter d_x_grad(DT_FLOAT, x.size());
  std::vector<float> y_grad(y.size(), 1.0f);
  std::vector<float> x_grad(x.size(), 0.f);
  gpu_device.memcpyHostToDevice(d_y_grad.data<float>(), y_grad.data(),
                                sizeof(float) * y_grad.size());
  gpu_device.memcpyHostToDevice(d_x_grad.data<float>(), x_grad.data(),
                                sizeof(float) * x_grad.size());
  gpu_device.synchronize();
  auto y_gradm = d_y_grad.const_matrix<float>(B, out_features);
  auto x_gradm = d_x_grad.matrix<float>(B, in_features);
  m.Backward(xm, y_gradm, x_gradm);
  gpu_device.memcpyDeviceToHost(x_grad.data(), d_x_grad.data<float>(),
                                sizeof(float) * x_grad.size());
  std::vector<float> weight_grad(m.weight_->size()), bias_grad(m.bias_->size());
  gpu_device.memcpyDeviceToHost(weight_grad.data(), m.weight_->grad<float>(),
                                sizeof(float) * weight_grad.size());
  gpu_device.memcpyDeviceToHost(bias_grad.data(), m.bias_->grad<float>(),
                                sizeof(float) * bias_grad.size());
  gpu_device.synchronize();

  std::vector<float> expected_w_grad = {-2.915748, 0.095726, 4.543815,
                                        -2.915748, 0.095726, 4.543815};
  std::vector<float> expected_b_grad = {4, 4};
  std::vector<float> expected_x_grad = {
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753,
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753};

  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], weight_grad[i], 1e-5);
  }
  for (size_t i = 0; i < expected_b_grad.size(); ++i) {
    EXPECT_NEAR(expected_b_grad[i], bias_grad[i], 1e-5);
  }
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(Embedding, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Embedding(10, 3)
idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
emb = m(idx)
loss = torch.sum(emb)
loss.backward()
*/

  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  int vocab_size = 10, dim = 3;
  nn::Embedding m(vocab_size, dim);

  std::vector<int> idx = {1, 2, 4, 5, 4, 3, 2, 9};
  nn::Parameter embedding_gpu(DT_FLOAT, idx.size() * dim);
  m.Forward(idx, embedding_gpu.span<float>());

  std::vector<float> embedding(idx.size() * dim);
  gpu_device.memcpyDeviceToHost(embedding.data(), embedding_gpu.data<float>(),
                                sizeof(float) * embedding.size());
  gpu_device.synchronize();
  std::vector<float> expected_embedding = {
      -2.105521, 0.678418,  -1.234545, -0.043067, -1.604667, -0.752135,
      -0.727881, -0.559430, -2.316923, -0.216805, -1.384674, -0.871236,
      -0.727881, -0.559430, -2.316923, 1.648723,  -0.392479, -1.403607,
      -0.043067, -1.604667, -0.752135, -0.601142, -1.274151, 2.122785};
  for (size_t i = 0; i < expected_embedding.size(); ++i) {
    EXPECT_NEAR(expected_embedding[i], embedding[i], 1e-5);
  }

  std::vector<float> expected_w_grad = {0., 0., 0., 1., 1., 1., 2., 2., 2., 1.,
                                        1., 1., 2., 2., 2., 1., 1., 1., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 1., 1., 1.};
  std::vector<float> grad_embedding(idx.size() * dim, 1.0f);
  nn::Parameter grad_embedding_gpu(DT_FLOAT, idx.size() * dim);
  gpu_device.memcpyHostToDevice(grad_embedding_gpu.data<float>(),
                                grad_embedding.data(),
                                sizeof(float) * grad_embedding.size());
  gpu_device.synchronize();
  m.Backward(idx, grad_embedding_gpu.span<float>());

  std::vector<float> weight_grad(m.weight_->size());
  gpu_device.memcpyDeviceToHost(weight_grad.data(), m.weight_->grad<float>(),
                                sizeof(float) * weight_grad.size());
  gpu_device.synchronize();
  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], weight_grad[i], 1e-5);
  }
}

TEST(LayerNorm, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
batch, sentence_length, embedding_dim = 4, 16, 4
x = torch.randn(batch, sentence_length, embedding_dim)
x = nn.Parameter(x)
m = nn.LayerNorm(embedding_dim)
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  auto& gpu_device = nn::g_device;
  int batch = 4, sentence_length = 16, embedding_dim = 4;
  Parameter x(DT_FLOAT, batch * sentence_length * embedding_dim);
  nn::NormalFill(x.span<float>());
  auto x_m =
      MakeConstMatrix(x.data<float>(), batch * sentence_length, embedding_dim);
  int row_size = batch * sentence_length;
  Parameter y(DT_FLOAT, x.size()), mean(DT_FLOAT, row_size),
      rstd(DT_FLOAT, row_size);
  auto y_m = MakeMatrix(y.data<float>(), row_size, embedding_dim);
  auto mean_m = MakeFlat(mean.data<float>(), row_size);
  auto rstd_m = MakeFlat(rstd.data<float>(), row_size);
  auto m = nn::LayerNorm(embedding_dim);
  m.Forward(x_m, y_m, mean_m, rstd_m);
  std::vector<float> expected_y = {
      0.871568,  0.592812,  0.220889,  -1.685270, 1.343979,  -0.747299,
      0.555241,  -1.151922, -0.462176, 1.642320,  -0.146915, -1.033229,
      -0.640134, -0.373525, -0.704957, 1.718616,  1.578360,  -0.633050,
      -1.047617, 0.102306,  -1.620320, 0.419836,  0.111535,  1.088949,
      0.495214,  0.552731,  -1.728073, 0.680128,  -0.745194, -0.139286,
      -0.789412, 1.673892,  -1.015600, -0.578948, -0.027969, 1.622518,
      0.978930,  -0.594526, 0.951077,  -1.335481, -1.117631, 1.608192,
      -0.394021, -0.096539, -0.760813, 1.687337,  -0.732347, -0.194177,
      -1.170991, -0.725308, 0.557885,  1.338414,  -0.337432, 1.713225,
      -0.735528, -0.640265, -1.605659, 0.297183,  0.165659,  1.142816,
      -0.485914, 1.129540,  -1.389725, 0.746099,  0.286472,  -0.627944,
      1.487023,  -1.145551, 0.299241,  -0.342977, -1.355051, 1.398787,
      -1.357315, 1.123290,  -0.541784, 0.775808,  1.349958,  0.115885,
      -1.471093, 0.005251,  1.157446,  0.452900,  -1.566235, -0.044112,
      -0.070412, -0.100323, 1.494292,  -1.323556, 1.054618,  -0.581938,
      -1.340275, 0.867594,  -1.308845, -0.548262, 1.290411,  0.566697,
      -1.694622, 0.606525,  0.834207,  0.253890,  0.323692,  0.220598,
      1.087690,  -1.631981, -1.386152, 1.363097,  0.343145,  -0.320090,
      -0.073245, 1.629092,  -0.522729, -1.033118, -1.461255, 0.448287,
      -0.252015, 1.264983,  1.134724,  -0.344502, -1.463118, 0.672896,
      -1.003026, -0.137854, -0.507443, 1.648324,  1.520410,  -1.011541,
      0.262874,  -0.771743, 1.294280,  0.646568,  -1.078416, -0.862432,
      -1.298251, 1.511276,  -0.044074, -0.168950, 0.922166,  -0.044200,
      -1.614188, 0.736221,  -0.747085, 1.248320,  -1.188131, 0.686896,
      -1.683398, 0.188602,  0.829112,  0.665684,  1.078050,  -0.178072,
      -1.545627, 0.645649,  -0.346409, 0.462202,  -1.410598, 1.294804,
      -1.387675, 0.336611,  -0.313711, 1.364774,  -1.678569, 0.870049,
      0.626194,  0.182327,  1.068842,  -0.664288, 0.878082,  -1.282636,
      -1.562189, 0.899704,  0.846327,  -0.183842, -1.588379, 0.196610,
      0.211188,  1.180581,  0.690463,  0.924564,  -1.633341, 0.018314,
      0.294812,  -1.418109, -0.235498, 1.358795,  0.540153,  -0.677221,
      -1.204288, 1.341356,  0.911414,  -1.692396, 0.385871,  0.395112,
      1.325757,  -1.218455, 0.559515,  -0.666817, -1.570161, 0.114548,
      0.247170,  1.208442,  0.195450,  1.544783,  -0.695173, -1.045060,
      -1.262950, 0.695054,  1.222140,  -0.654243, -0.411368, -1.424810,
      1.156788,  0.679390,  0.571627,  0.834783,  0.293628,  -1.700038,
      -0.385801, 0.754057,  -1.451946, 1.083690,  -0.396792, -1.388408,
      0.491757,  1.293443,  -0.198960, 1.519554,  -0.036059, -1.284534,
      -1.112192, -0.415548, -0.079726, 1.607466,  1.359567,  0.172076,
      -1.454633, -0.077011, -1.049470, -0.644490, 0.123547,  1.570413,
      -1.304491, 0.800871,  -0.622824, 1.126445,  -1.064767, -0.092131,
      -0.467621, 1.624519,  1.057049,  -1.618052, 0.511932,  0.049071,
      0.590086,  0.777018,  0.343934,  -1.711037};
  std::vector<float> y_cpu(y.size());
  gpu_device.memcpyDeviceToHost(y_cpu.data(), y.data<float>(),
                                sizeof(float) * y.size());
  gpu_device.synchronize();
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y_cpu[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad(x.size(), 1.0), x_grad(x.size(), 0);
  Parameter y_grad_gpu(DT_FLOAT, x.size()), x_grad_gpu(DT_FLOAT, x.size());
  gpu_device.memcpyHostToDevice(y_grad_gpu.data<float>(), y_grad.data(),
                                sizeof(float) * y_grad.size());
  gpu_device.memcpyHostToDevice(x_grad_gpu.data<float>(), x_grad.data(),
                                sizeof(float) * x_grad.size());
  auto y_grad_m =
      MakeConstMatrix(y_grad_gpu.data<float>(), row_size, embedding_dim);
  auto x_grad_m = MakeMatrix(x_grad_gpu.data<float>(), row_size, embedding_dim);
  auto mean_const = MakeConstFlat(mean.data<float>(), row_size);
  auto rstd_const = MakeConstFlat(rstd.data<float>(), row_size);
  m.Backward(x_m, y_grad_m, mean_const, rstd_const, x_grad_m);

  std::vector<float>
      expected_w_grad = {-13.071318, 11.136637, -13.703045, 15.637726},
      expected_b_grad = {64., 64., 64., 64.},
      expected_x_grad = {
          0.000000e+00,  0.000000e+00,  0.000000e+00,  5.960464e-08,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          1.192093e-07,  1.192093e-07,  1.192093e-07,  -1.192093e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  -2.384186e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          1.192093e-07,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  5.960464e-08,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          2.384186e-07,  0.000000e+00,  -4.768372e-07, 0.000000e+00,
          0.000000e+00,  0.000000e+00,  -7.152557e-07, 4.768372e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          -5.960464e-08, 0.000000e+00,  0.000000e+00,  0.000000e+00,
          5.960464e-08,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  1.192093e-07,  -2.384186e-07,
          4.768372e-07,  -2.384186e-07, 0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          1.192093e-07,  -2.384186e-07, 0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          -1.192093e-07, 0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  2.384186e-07,  0.000000e+00,
          -1.192093e-07, 0.000000e+00,  -2.384186e-07, 1.192093e-07,
          -5.960464e-08, 0.000000e+00,  -5.960464e-08, 0.000000e+00,
          1.192093e-07,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          5.960464e-08,  0.000000e+00,  0.000000e+00,  -5.960464e-08,
          0.000000e+00,  0.000000e+00,  2.384186e-07,  0.000000e+00,
          2.384186e-07,  -1.192093e-06, -2.384186e-07, 9.536743e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  2.384186e-07,  0.000000e+00,  0.000000e+00,
          -5.960464e-08, 0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  5.960464e-08,  0.000000e+00,  -5.960464e-08,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          -1.192093e-07, -1.192093e-07, 0.000000e+00,  1.192093e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          1.192093e-07,  0.000000e+00,  1.192093e-07,  -1.192093e-07,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
          1.192093e-07,  0.000000e+00,  1.192093e-07,  0.000000e+00,
          0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00};
  std::vector<float> weight_grad(m.weight_->size()), bias_grad(m.bias_->size());
  gpu_device.memcpyDeviceToHost(weight_grad.data(), m.weight_->grad<float>(),
                                weight_grad.size() * sizeof(float));
  gpu_device.memcpyDeviceToHost(bias_grad.data(), m.bias_->grad<float>(),
                                bias_grad.size() * sizeof(float));
  gpu_device.synchronize();
  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], weight_grad[i], 1e-5);
  }
  for (size_t i = 0; i < expected_b_grad.size(); ++i) {
    EXPECT_NEAR(expected_b_grad[i], bias_grad[i], 1e-5);
  }
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(NewGELU, ForwardAndBackward) {
  /*
class NewGELU(nn.Module):
  def forward(self, input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input +
0.044715 * torch.pow(input, 3.0))))

torch.manual_seed(42)
batch, dim = 4, 3
x = torch.randn(batch, dim)
x = nn.Parameter(x)
m = NewGELU()
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  auto& gpu_device = nn::g_device;
  // forward
  std::vector<float> x = {0.336690,  0.128809,  0.234462, 0.230333,
                          -1.122856, -0.186328, 2.208201, -0.637997,
                          0.461657,  0.267351,  0.534905, 0.809357};
  std::vector<float> expected_y = {0.212725,  0.071006,  0.138962, 0.136145,
                                   -0.147006, -0.079394, 2.178409, -0.167029,
                                   0.312915,  0.161853,  0.376359, 0.639989};
  std::vector<float> y(x.size(), 0);
  Parameter x_gpu(DT_FLOAT, x.size()), y_gpu(DT_FLOAT, y.size());
  gpu_device.memcpyHostToDevice(x_gpu.data<float>(), x.data(),
                                sizeof(float) * x.size());
  gpu_device.synchronize();
  nn::NewGELU m;
  m.Forward(x_gpu.const_flat<float>(), y_gpu.flat<float>());
  gpu_device.memcpyDeviceToHost(y.data(), y_gpu.data<float>(),
                                sizeof(float) * y.size());
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad(x.size(), 1.0f), x_grad(x.size(), 0.f);
  Parameter y_grad_gpu(DT_FLOAT, y_grad.size()),
      x_grad_gpu(DT_FLOAT, x_grad.size());
  gpu_device.memcpyHostToDevice(y_grad_gpu.data<float>(), y_grad.data(),
                                sizeof(float) * y_grad.size());
  gpu_device.memcpyHostToDevice(x_grad_gpu.data<float>(), x_grad.data(),
                                sizeof(float) * x_grad.size());
  m.Backward(x_gpu.const_flat<float>(), y_grad_gpu.const_flat<float>(),
             x_grad_gpu.flat<float>());
  gpu_device.memcpyDeviceToHost(x_grad.data(), x_grad_gpu.data<float>(),
                                sizeof(float) * x_grad.size());
  gpu_device.synchronize();
  std::vector<float> expected_x_grad = {
      0.758699, 0.602206, 0.683672, 0.680552, -0.107435, 0.353047,
      1.064087, 0.054299, 0.843291, 0.708290, 0.888446,  1.023232};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}
