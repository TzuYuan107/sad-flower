#!/usr/bin/env bash

conda activate gflower

python run/eval_maze2d.py \
    --random_repeat 10 \
    --env maze2d-large-v1 \
    --log_folder ./logs \
    --horizon 384 \
    --obs_center 4.5 5.3 1.5 4.8 \
    --obs_radius 1.0 1.0 1.0 1.0 \
    --flow_exp_name Friction_True \
    --value_exp_name Friction_True \
    --exp_name Friction_True \
    --normalizer LimitsNormalizer \
    --preprocess_fns maze2d_set_terminals \
    --flow_cp 21 \
    --value_cp 2 \
    --ode_t_span 0 1.0 \
    --ss_batch 32 \
    --act_neg_lim -0.9 \
    --act_pos_lim 0.9 \
    --NN_esemble_idx 0 \
    --guidance_method reject_ss \
    --constraint_strategy Reject \
    --NN_folder smooth \
    --IsEma \
    --ode_t_steps 1000 \
    --ode_solver euler \