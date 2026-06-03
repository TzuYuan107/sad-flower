for env in maze2d-open-dense-v0 maze2d-large-v1 maze2d-umaze-v1; do
    if [ $env == "maze2d-open-dense-v0" ]; then
        horizon=48
        batch_size=512
    elif [ $env == "maze2d-umaze-v1" ]; then
        horizon=128
        batch_size=512
    elif [ $env == "maze2d-large-v1" ]; then
        horizon=384
        batch_size=128
    fi

    python run/train_model_dynamics_jac_reg.py \
        --device cuda:7 \
        --log_folder ./logs \
        --exp_name "smooth" \
        --env $env \
        --state_dim 4 \
        --action_dim 2 \
        --normalizer LimitsNormalizer \
        --horizon $horizon \
        --n_train_steps 500001 \
        --save_freq 20000 \
        --val_freq 20000 \
        --lr_schdule_T 1000000 \
        --val_frac 0.05 \
        --batch_size $batch_size
done