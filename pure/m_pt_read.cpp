#include <algorithm>
#include "ptio.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
int main(){
  auto ts = pt::load_pt("y8_sd.pt");
  printf("C++ load_pt: %zu tensors\n", ts.size());
  // compare to y8_ref.txt (name shape first last)
  std::ifstream f("y8_ref.txt"); std::string line; size_t idx=0; double md=0; int n=0;
  while(std::getline(f,line)){ std::istringstream is(line); std::string name; is>>name;
    // ref first/last are last two tokens
    std::vector<std::string> tok; std::string t; while(is>>t) tok.push_back(t);
    float rf=std::stof(tok[tok.size()-2]), rl=std::stof(tok.back());
    if(idx<ts.size()){ float cf=ts[idx].data.front(), cl=ts[idx].data.back();
      md=std::max({md,(double)std::abs(cf-rf),(double)std::abs(cl-rl)});
      if(ts[idx].name!=name) printf("  NAME MISMATCH idx %zu: %s vs %s\n",idx,ts[idx].name.c_str(),name.c_str());
    } ++idx; ++n; }
  printf("compared %d tensors, first/last max|diff| = %.2e  %s\n", n, md, md<1e-5?"OK":"FAIL");
  return 0;
}
