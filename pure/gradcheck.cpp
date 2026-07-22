// Verify the autograd engine: analytic grads vs central finite differences.
// No external libraries — this is the whole point.
#include "autograd.hpp"
#include <random>
#include <cstdio>
#include <string>

static std::mt19937 rng(123);
static void randn(Tensor& t) {
  std::normal_distribution<float> nd(0, 1);
  for (auto& v : t->data) v = nd(rng);
}

// Central-difference gradient check. build() reconstructs the graph from `inputs`' data.
static double gradcheck(const std::string& name,
                        std::function<Tensor()> build, std::vector<Tensor> inputs,
                        float eps = 1e-3f) {
  auto out = build();
  backward(out);
  std::vector<std::vector<float>> analytic;
  for (auto& x : inputs) analytic.push_back(x->grad);

  double maxviol = 0;  // how far max error exceeds the atol+rtol tolerance
  double maxerr = 0;
  for (size_t k = 0; k < inputs.size(); ++k) {
    auto& x = inputs[k];
    for (int64_t i = 0; i < x->numel(); ++i) {
      float o = x->data[i];
      x->data[i] = o + eps; float fp = build()->data[0];
      x->data[i] = o - eps; float fm = build()->data[0];
      x->data[i] = o;
      float num = (fp - fm) / (2 * eps);
      float ana = analytic[k][i];
      double err = std::abs(num - ana);
      double tol = 2e-3 + 1e-2 * std::abs(ana);   // atol + rtol*|ana| (PyTorch-style)
      maxerr = std::max(maxerr, err);
      maxviol = std::max(maxviol, err - tol);
    }
  }
  bool pass = maxviol <= 0;
  printf("[%s] max |num-ana| = %.3e  %s\n", name.c_str(), maxerr, pass ? "OK" : "FAIL");
  return pass ? 0.0 : 1.0;
}

int main() {
  bool ok = true;

  {   // L = sum( silu(a*b) + a )
    auto a = from_data({3, 4}, std::vector<float>(12), true);
    auto b = from_data({3, 4}, std::vector<float>(12), true);
    randn(a); randn(b);
    ok &= gradcheck("silu(a*b)+a -> sum", [&] { return sum(add(silu(mul(a, b)), a)); }, {a, b}) < 1e-2;
  }
  {   // L = mean( sigmoid(a+b) * a )
    auto a = from_data({2, 5}, std::vector<float>(10), true);
    auto b = from_data({2, 5}, std::vector<float>(10), true);
    randn(a); randn(b);
    ok &= gradcheck("sigmoid(a+b)*a -> mean", [&] { return mean(mul(sigmoid(add(a, b)), a)); }, {a, b}) < 1e-2;
  }
  {   // deeper chain: L = sum( silu( silu(a*a) + b ) )
    auto a = from_data({4}, std::vector<float>(4), true);
    auto b = from_data({4}, std::vector<float>(4), true);
    randn(a); randn(b);
    ok &= gradcheck("nested silu -> sum", [&] { return sum(silu(add(silu(mul(a, a)), b))); }, {a, b}) < 1e-2;
  }

  printf("\n%s\n", ok ? "ALL GRADCHECKS PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
