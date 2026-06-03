conda activate gflower

for env in maze2d-large-v1; do
    if [ $env == "maze2d-open-dense-v0"]; then
        horizon=48
    elif [ $env == "maze2d-umaze-v1" ]; then
        horizon=128
    elif [ $env == "maze2d-large-v1" ]; then
        horizon=384
    fi

    python run/train_value.py \
        --device cuda:6 \
        --exp_name "$flow_prefix"Friction_True \
        --env $env \
        --inf_horizon \
        --horizon $horizon \
        --n_train_steps 10001 \
        --save_freq 5000 \
        --batch_size 64 \
        --normalizer LimitsNormalizer \
        --learning_rate 2e-4 
done