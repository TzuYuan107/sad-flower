#!/bin/bash

# Script to generate main results from evaluation records
# Usage: ./debug_exp_mainresult.sh --env maze2d-large-v1 --strategy ExpJac_ACSCDC_AllInOne_ALim0.9_H384 --exp_name <exp_name> --repeat_time 10

# Default values
ENV="maze2d-umaze-v1"
STRATEGY="Strategy"
EXP_NAME="exp_name"
REPEAT_TIME=10

echo "Running debug_exp_mainresult.py with:"
echo "  env: $ENV"
echo "  strategy: $STRATEGY"
echo "  exp_name: $EXP_NAME"
echo "  repeat_time: $REPEAT_TIME"
echo ""

cd "$(dirname "$0")/.."
python run/debug_exp_mainresult.py \
    --env "$ENV" \
    --strategy "$STRATEGY" \
    --exp_name "$EXP_NAME" \
    --repeat_time "$REPEAT_TIME"
