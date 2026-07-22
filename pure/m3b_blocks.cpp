// M3b: verify pure-engine C2f and SPPF blocks against real yolov8n blocks.
#include "blocks.hpp"
#include <cstdio>

int main() {
  bool ok = true;
  {
    auto blk = load_block("pure/ref/data_c2f/");
    auto y = c2f(blk.x, blk.convs, blk.n_bott, /*shortcut=*/true);
    double d = maxdiff(y, blk.ref);
    printf("[C2f ] out [1,%lld,%lld,%lld] max|diff| = %.3e  %s\n",
           (long long)y->shape[1], (long long)y->shape[2], (long long)y->shape[3],
           d, d < 1e-4 ? "OK" : "FAIL");
    ok &= d < 1e-4;
  }
  {
    auto blk = load_block("pure/ref/data_sppf/");
    auto y = sppf(blk.x, blk.convs);
    double d = maxdiff(y, blk.ref);
    printf("[SPPF] out [1,%lld,%lld,%lld] max|diff| = %.3e  %s\n",
           (long long)y->shape[1], (long long)y->shape[2], (long long)y->shape[3],
           d, d < 1e-4 ? "OK" : "FAIL");
    ok &= d < 1e-4;
  }
  printf("\n%s\n", ok ? "M3b BLOCKS MATCH yolov8n" : "MISMATCH");
  return ok ? 0 : 1;
}
