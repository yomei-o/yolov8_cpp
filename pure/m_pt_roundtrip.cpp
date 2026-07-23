#include "ptio.hpp"
#include <cstdio>
int main(){                             // pure-C++ .pt -> .pt (read then write, no Python)
  auto ts = pt::load_pt("y8_sd.pt");
  printf("read %zu tensors from y8_sd.pt, writing y8_cpp.pt ...\n", ts.size());
  pt::save_pt(ts, "y8_cpp.pt");
  printf("done (pure C++).\n");
  return 0;
}
