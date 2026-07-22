// C-2: verify the pure optimizers against torch.optim on a deterministic quadratic
// f(w) = 0.5*||w-target||^2 (grad = w-target), matching final params after K steps.
#include "optim.hpp"
#include <cstdio>
#include <fstream>
#include <functional>

static std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("run: python pure/ref/optim_ref.py\n"); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::vector<float> v(n / 4); f.read((char*)v.data(), n); return v;
}

int main() {
  const std::string D = "pure/ref/data_optim/";
  std::ifstream mf(D + "meta.txt"); if (!mf) { printf("run: python pure/ref/optim_ref.py\n"); return 1; }
  int64_t N, K; mf >> N >> K;
  auto w0 = rd(D + "w0.bin"), target = rd(D + "target.bin");

  // (name, factory building an optimizer over the single param, a stepper) — the
  // stepper hides SGD vs Adam behind a uniform run() below.
  auto run = [&](const std::string& name, std::function<void(Tensor&, std::function<void()>&)> setup) {
    auto w = from_data({N}, w0, true);
    std::function<void()> step;
    setup(w, step);
    for (int64_t it = 0; it < K; ++it) {
      for (int64_t i = 0; i < N; ++i) w->grad[i] = w->data[i] - target[i];   // quadratic grad
      step();
    }
    auto ref = rd(D + "w_" + name + ".bin");
    double m = 0; for (int64_t i = 0; i < N; ++i) m = std::max(m, (double)std::abs(w->data[i] - ref[i]));
    printf("  %-9s max|diff| = %.3e  %s\n", name.c_str(), m, m < 1e-5 ? "OK" : "FAIL");
    return m < 1e-5;
  };

  bool ok = true;
  ok &= run("sgd",      [](Tensor& w, std::function<void()>& s){ auto o=std::make_shared<SGD>(std::vector<Tensor>{w},0.1f,0.9f,1e-4f,false); s=[o]{o->step();}; });
  ok &= run("sgd_nest", [](Tensor& w, std::function<void()>& s){ auto o=std::make_shared<SGD>(std::vector<Tensor>{w},0.1f,0.9f,1e-4f,true);  s=[o]{o->step();}; });
  ok &= run("adam",     [](Tensor& w, std::function<void()>& s){ auto o=std::make_shared<Adam>(std::vector<Tensor>{w},0.05f,0.9f,0.999f,1e-8f,1e-2f,false); s=[o]{o->step();}; });
  ok &= run("adamw",    [](Tensor& w, std::function<void()>& s){ auto o=std::make_shared<Adam>(std::vector<Tensor>{w},0.05f,0.9f,0.999f,1e-8f,1e-2f,true);  s=[o]{o->step();}; });

  printf("\n%s\n", ok ? "C-2: PURE optimizers == torch.optim" : "MISMATCH");
  return ok ? 0 : 1;
}
