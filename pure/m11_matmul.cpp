// C-3: verify pure matmul + transpose (forward and grads) against PyTorch autograd.
// loss = sum((A@B) * G) + sum(A^T * H) exercises matmul backward (dA, dB) and
// transpose backward (its contribution to dA).
#include "linalg.hpp"
#include "ops2d.hpp"     // sum, mul
#include <cstdio>
#include <fstream>

static std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("run: python pure/ref/matmul_ref.py\n"); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::vector<float> v(n / 4); f.read((char*)v.data(), n); return v;
}
static double md(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::abs(a[i] - b[i])); return m;
}

int main() {
  const std::string D = "pure/ref/data_matmul/";
  std::ifstream mf(D + "meta.txt"); if (!mf) { printf("run: python pure/ref/matmul_ref.py\n"); return 1; }
  int64_t M, K, N; mf >> M >> K >> N;

  auto A = from_data({M, K}, rd(D + "A.bin"), true);
  auto B = from_data({K, N}, rd(D + "B.bin"), true);
  auto G = from_data({M, N}, rd(D + "G.bin"), false);
  auto H = from_data({K, M}, rd(D + "H.bin"), false);

  auto O = matmul(A, B);
  auto loss = add(sum(mul(O, G)), sum(mul(transpose2d(A), H)));
  backward(loss);

  double dO = md(O->data,  rd(D + "O.bin"));
  double dA = md(A->grad,  rd(D + "dA.bin"));
  double dB = md(B->grad,  rd(D + "dB.bin"));
  bool ok = dO < 1e-5 && dA < 1e-5 && dB < 1e-5;
  printf("matmul  fwd=%.3e  dA=%.3e  dB=%.3e  %s\n", dO, dA, dB, ok ? "OK" : "FAIL");
  printf("\n%s\n", ok ? "C-3: PURE matmul/transpose == torch" : "MISMATCH");
  return ok ? 0 : 1;
}
