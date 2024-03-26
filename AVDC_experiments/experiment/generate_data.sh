# usage: bash benchmark.sh <GPU_ID> <PATH_TO_BENCHMARK.PY>
for task in "door-open-v2-goal-observable"
do 
    CUDA_VISIBLE_DEVICES=$1 accelerate launch generate_data.py --env_name $task --n_exps 25 --ckpt_dir "../ckpts/metaworld" --milestone 24 --result_root "../results/results_AVDC_mw"
done

