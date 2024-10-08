# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>

 for task in "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"
# for task in 'assembly-v2-goal-observable'
do 
    CUDA_VISIBLE_DEVICES=$1 accelerate launch generate_data.py --env_name $task --n_exps 5 --ckpt_dir "../ckpts/metaworld" --milestone 24 --result_root "../results/results_AVDC_mw"
done

