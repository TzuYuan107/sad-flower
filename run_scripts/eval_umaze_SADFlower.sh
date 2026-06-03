conda activate gflower

for flow_matching_type in cfm; do
    for env in maze2d-umaze-v1; do
        if [ $env == "maze2d-large-v1" ]; then
            horizon=384
        elif [ $env == "maze2d-umaze-v1" ]; then
            horizon=128
        fi

        if [ $flow_matching_type == "cfm" ]; then
            flow_prefix=""
        elif [ $flow_matching_type == "ot_cfm" ]; then
            flow_prefix="ot_"
        fi

        python run/eval_maze2d.py \
        --horizon $horizon \
        --random_repeat 10 \
        --env $env \
        --state_dim 4 \
        --action_dim 2 \
        --log_folder ./logs \
        --obs_center 2 1.5 2 2 \
        --obs_radius 0.6 1.2 1.52 1.52 \
        --flow_exp_name Friction_True \
        --value_exp_name Friction_True \
        --exp_name Friction_True \
        --normalizer LimitsNormalizer \
        --preprocess_fns maze2d_set_terminals \
        --flow_cp 19 \
        --value_cp 2 \
        --ode_t_span 0 1.0 \
        --guidance_method ss \
        --ss_batch 32 \
        --start_time 0.95 \
        --act_neg_lim "-0.9" \
        --act_pos_lim 0.9 \
        --NN_esemble_idx 0 \
        --constraint_strategy QP_PT_ACSCDC \
        --NN_folder smooth \
        --CBF_c 0.2 0.2 \
        --CLF_c 0.5 \
        --PT_CBF_min 0.005 \
        --PT_CLF_min 0.05 \
        --IsEma \
        --ode_t_steps 250 \
        --ode_solver euler \
        --s_robust 0.0 \
        --a_robust 0.0 \
        --d_robust 0.0 \
        --ACSCDCFlag 1
    done
done