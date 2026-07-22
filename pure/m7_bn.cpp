// C-1: verify the pure BatchNorm2d op (forward, backward grads, running-stat update)
// against torch.nn.BatchNorm2d, in both train and eval mode.
#include "bn.hpp"
#include <cstdio>
#include <fstream>

static std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("run: python pure/ref/bn_ref.py\n"); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::vector<float> v(n / 4); f.read((char*)v.data(), n); return v;
}
static double md(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::abs(a[i] - b[i])); return m;
}

int main() {
  const std::string D = "pure/ref/data_bn/";
  std::ifstream mf(D + "meta.txt"); if (!mf) { printf("run: python pure/ref/bn_ref.py\n"); return 1; }
  int64_t N, C, H, W; float eps, mom; mf >> N >> C >> H >> W >> eps >> mom;

  auto xd = rd(D + "x.bin"), gd = rd(D + "gamma.bin"), bd = rd(D + "beta.bin");
  auto rm0 = rd(D + "rm.bin"), rv0 = rd(D + "rv.bin"), gy = rd(D + "gy.bin");

  bool ok = true;
  for (int t = 0; t < 2; ++t) {
    bool training = (t == 0); const char* tag = training ? "train" : "eval";
    auto x = from_data({N, C, H, W}, xd, true);
    auto gamma = from_data({C}, gd, true), beta = from_data({C}, bd, true);
    std::vector<float> rm = rm0, rv = rv0;

    auto y = batchnorm2d(x, gamma, beta, rm, rv, eps, training, mom);
    auto gyt = from_data({N, C, H, W}, gy, false);
    auto loss = sum(mul(y, gyt));                          // d loss / d y = gy
    backward(loss);

    double dy   = md(y->data,     rd(D + "y_"      + tag + ".bin"));
    double ddx  = md(x->grad,     rd(D + "dx_"     + tag + ".bin"));
    double ddg  = md(gamma->grad, rd(D + "dgamma_" + tag + ".bin"));
    double ddb  = md(beta->grad,  rd(D + "dbeta_"  + tag + ".bin"));
    double drm  = md(rm,          rd(D + "rm_"     + tag + ".bin"));
    double drv  = md(rv,          rd(D + "rv_"     + tag + ".bin"));
    bool pass = dy < 1e-5 && ddx < 1e-5 && ddg < 1e-5 && ddb < 1e-5 && drm < 1e-6 && drv < 1e-6;
    ok = ok && pass;
    printf("[%-5s] y=%.2e dx=%.2e dgamma=%.2e dbeta=%.2e run_mean=%.2e run_var=%.2e  %s\n",
           tag, dy, ddx, ddg, ddb, drm, drv, pass ? "OK" : "FAIL");
  }
  printf("\n%s\n", ok ? "C-1: PURE BatchNorm2d == torch.nn.BatchNorm2d" : "MISMATCH");
  return ok ? 0 : 1;
}
