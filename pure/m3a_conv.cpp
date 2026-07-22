// M3a: run one fused Conv (conv+folded-BN then SiLU) in the pure engine and
// compare to yolov8n's real first layer. Self-contained: reads the reference too.
#include "autograd.hpp"
#include <fstream>
#include <cstdio>

static std::vector<float> read_bin(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

int main() {
  const std::string D = "pure/ref/data/";
  int64_t Cin, Cout, H, W, k, stride, pad;
  { std::ifstream f(D + "meta.txt"); f >> Cin >> Cout >> H >> W >> k >> stride >> pad; }

  auto x = from_data({1, Cin, H, W}, read_bin(D + "x.bin"));
  auto w = from_data({Cout, Cin, k, k}, read_bin(D + "wfold.bin"));
  auto b = from_data({Cout}, read_bin(D + "bfold.bin"));

  auto y = silu(conv2d(x, w, b, stride, pad));      // fused Conv forward

  auto ref = read_bin(D + "ref.bin");
  double maxdiff = 0; for (int64_t i = 0; i < y->numel(); ++i)
    maxdiff = std::max(maxdiff, (double)std::abs(y->data[i] - ref[i]));
  printf("out shape [1,%lld,%lld,%lld], max|diff vs yolov8n| = %.3e  %s\n",
         (long long)y->shape[1], (long long)y->shape[2], (long long)y->shape[3],
         maxdiff, maxdiff < 1e-4 ? "OK" : "FAIL");
  return maxdiff < 1e-4 ? 0 : 1;
}
