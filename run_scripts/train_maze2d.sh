
for flow_matching_type in cfm; do
    for env in  maze2d-large-v1 maze2d-open-dense-v0 maze2d-umaze-v1; do
        if [ $env == "maze2d-open-dense-v0" ]; then
            horizon=48
        elif [ $env == "maze2d-umaze-v1" ]; then
            horizon=128
        elif [ $env == "maze2d-large-v1" ]; then
            horizon=384
        fi
        
        if [ $flow_matching_type == "cfm" ]; then
            flow_prefix=""
        elif [ $flow_matching_type == "ot_cfm" ]; then
            flow_prefix="ot_"
        fi

        python run/train_maze2d.py \
            --device cuda:6 \
            --log_folder ./logs \
            --exp_name "$flow_prefix"Friction_True \
            --env $env \
            --state_dim 4 \
            --action_dim 2 \
            --normalizer LimitsNormalizer \
            --preprocess_fns maze2d_set_terminals\
            --max_path_length 40000\
            --horizon $horizon \
            --n_train_steps 1000001 \
            --save_freq 50000 \
            --lr_schdule_T 1000000 \
            --batch_size 32 \
            --learning_rate 2e-4 \
            --ema_decay 0.995 \
            --flow_matching_type $flow_matching_type
    done
done