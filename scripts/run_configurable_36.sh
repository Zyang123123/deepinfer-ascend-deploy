#!/bin/bash
set -eo pipefail
export DEEPINFER_ROOT="${DEEPINFER_ROOT:-/mnt/nfs_share/home/j00930424/DeepInfer}"
cd "$DEEPINFER_ROOT"
source /usr/local/Ascend/cann/set_env.sh
source /opt/openEuler/gcc-toolset-14/enable
export PATH="$PWD/scripts/toolchain:$PATH"
python scripts/toolchain/prepare_headers.py
export DG_BISHENG_GCC_TOOLCHAIN=/opt/openEuler/gcc-toolset-14/root/usr
export DG_BISHENG_COMPAT_INCLUDE="$PWD/build/bisheng-gcc14-compat"
export DEEPINFER_ACCELERATOR=ascend HF_HUB_OFFLINE=1
export DEEPINFER_ASCEND_ENGRAM_CPU_STAGING=1
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_SOCKET_IFNAME=enp34s0f1 GLOO_SOCKET_IFNAME=enp34s0f1
export HCCL_CONNECT_TIMEOUT=1200 HCCL_DETERMINISTIC=true HCCL_IF_IP=192.168.45.36
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0 DEEPINFER_FORCE_LOAD_BALANCE=0
export DEEPINFER_USE_NEXTN_DECODE="${DEEPINFER_USE_NEXTN_DECODE:-5}"
test -s /etc/hccl_rootinfo.json
result_dir="${BENCH_RESULT_DIR:-$PWD/logs/bench-configurable-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$result_dir"
echo "Benchmark artifacts: $result_dir"
python -u scripts/benchmark_configurable.py --result-dir "$result_dir" "$@" 2>&1 | tee "$result_dir/run.log"
