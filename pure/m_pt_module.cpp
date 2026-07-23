// Read a RAW Ultralytics checkpoint ({'model': nn.Module}) fully in C++ -> state_dict,
// compare to torch's model.state_dict() (y8_ref.txt), and re-emit it as a state_dict .pt
// (pure C++) that Ultralytics can load. No Python in this program.
#include <algorithm>
#include "ptio.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : "yolov8n.pt";
  auto ts = pt::load_pt_module(path);
  printf("load_pt_module(%s): %zu tensors\n", path.c_str(), ts.size());

  std::ifstream f("y8_ref.txt"); std::string line; size_t idx = 0; double md = 0; int nm_mis = 0, n = 0;
  while (std::getline(f, line)) {
    std::istringstream is(line); std::string name; is >> name;
    std::vector<std::string> tok; std::string t; while (is >> t) tok.push_back(t);
    float rf = std::stof(tok[tok.size() - 2]), rl = std::stof(tok.back());
    if (idx < ts.size()) {
      if (ts[idx].name != name) { if (nm_mis < 3) printf("  name[%zu] %s vs %s\n", idx, ts[idx].name.c_str(), name.c_str()); ++nm_mis; }
      md = std::max({md, (double)std::abs(ts[idx].data.front() - rf), (double)std::abs(ts[idx].data.back() - rl)});
    }
    ++idx; ++n;
  }
  pt::save_pt(ts, "y8_fromraw.pt");
  printf("vs torch state_dict: %d tensors, name-mismatch=%d, first/last max|diff|=%.2e  %s\n", n, nm_mis, md, (nm_mis == 0 && md < 1e-5) ? "OK" : "FAIL");
  printf("wrote y8_fromraw.pt (pure C++, from raw %s)\n", path.c_str());
  return 0;
}
