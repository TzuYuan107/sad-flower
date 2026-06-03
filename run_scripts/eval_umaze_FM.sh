#!/usr/bin/env bash

conda activate gflower

python run/eval_maze2d.py \
    --random_repeat 10 \
    --env maze2d-umaze-v1 \
    --log_folder ./logs \
    --horizon 128 \
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
    --act_neg_lim -0.9 \
    --act_pos_lim 0.9 \
    --NN_esemble_idx 0 \
    --constraint_strategy No \
    --NN_folder smooth \
    --IsEma \
    --ode_t_steps 250 \
    --start_time 0.9 \
    --ode_solver euler \