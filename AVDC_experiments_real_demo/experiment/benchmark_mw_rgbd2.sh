# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
# for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
for task in "handle-press-v2-goal-observable"
do 
    CUDA_VISIBLE_DEVICES=$1 accelerate launch benchmark_mw_rgbd.py --env_name $task --n_exps 25 --ckpt_dir "./otf_ft_results" --milestone otf --result_root "./results/results_otf/"
done

python org_results_mw.py --results_root "../results/results_AVDC_mw" 