#!/bin/bash
# Build-health + numeric self-test (no Python). Compiles every pure/*.cpp with $CXX
# (default g++; set CXX=nvcc etc.) and runs the dependency-free self-tests.
#   usage: ./run_all.sh          (CPU/g++)     CXX="nvcc -x cu -DUSE_CUDA" ./run_all.sh
set -u
CXX=${CXX:-g++}; STD="-std=c++20 -O2 -Ipure/third_party"; ok=0; bad=0; skip=0
# Device-resident files (pure/dtensor.hpp / dnet.hpp / thrust) need CUDA CCCL headers; they
# are skipped unless THRUST is set, e.g.
#   THRUST="-std=c++17 -I/usr/local/cuda/include/cccl -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP" ./run_all.sh   (CPU)
#   CXX="nvcc -x cu -DUSE_CUDA" THRUST="--extended-lambda" ./run_all.sh                                              (GPU)
echo "== compile every pure/*.cpp with: $CXX =="
for f in pure/*.cpp; do
  if grep -qE 'thrust/|dtensor\.hpp|dnet\.hpp' "$f"; then
    if [ -n "${THRUST:-}" ]; then
      if $CXX $STD $THRUST -c "$f" -o /tmp/o.o 2>/tmp/e; then echo "  OK   $(basename "$f") [device]"; ok=$((ok+1));
      else echo "  FAIL $(basename "$f") [device]"; grep -iE "error" /tmp/e | head -3 | sed 's/^/       /'; bad=$((bad+1)); fi
    else echo "  skip $(basename "$f") (device/thrust — set THRUST=... to build)"; skip=$((skip+1)); fi
    continue
  fi
  if $CXX $STD -c "$f" -o /tmp/o.o 2>/tmp/e; then echo "  OK   $(basename "$f")"; ok=$((ok+1));
  else echo "  FAIL $(basename "$f")"; grep -iE "error" /tmp/e | head -3 | sed 's/^/       /'; bad=$((bad+1)); fi
done
[ $skip -gt 0 ] && echo "($skip device files skipped)"
echo "builds: $ok ok, $bad fail"
echo "== dependency-free self-tests =="
for t in gradcheck gradcheck2 m_pt; do
  [ -f "pure/$t.cpp" ] || continue
  if $CXX $STD "pure/$t.cpp" -o "/tmp/$t" 2>/dev/null; then echo "  $t: $("/tmp/$t" 2>&1 | tail -1)"; else echo "  $t: build failed"; fi
done
[ $bad -eq 0 ] && { echo "ALL BUILDS OK"; exit 0; } || { echo "SOME BUILDS FAILED"; exit 1; }
