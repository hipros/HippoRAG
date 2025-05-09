#!/usr/bin/env bash
# run_all_experiments.sh

# 결과와 로그를 저장할 디렉터리 생성
mkdir -p results logs

# 5가지 기법 리스트
methods=(montecarlo bidirectional iterative dynamic rp_ep)

# 공통 파라미터
datasets="hotpotqa musique 2wikimultihopqa"
common_args="--datasets $datasets --num_walks 2000 --walk_length 15 --push_eps 1e-4 --hops 3 --iters 20 --top_m 5"

for m in "${methods[@]}"; do
  echo "▶ Running method: $m" | tee logs/"${m}".log
  python new_method_main.py \
    --methods "$m" $common_args \
    > results/"${m}"_results.txt 2>> logs/"${m}".log
  echo "Completed $m – results in results/${m}_results.txt, log in logs/${m}.log"
  echo
done

echo "All experiments finished."