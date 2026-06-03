"""
Generate main results for evaluation with txt file.

Required input files:
- final_policy_record_{i}.txt for i in range(repeat_time) from evaluation results

This script:
1. Reads final_policy_record_{i}.txt for i in range(repeat_time)
2. Extracts SC, AC, DC values using GetMinSCAC_MaxDC function
3. Computes min, mean, std for SC and AC, and mean/std for DC
4. Saves results using save_main_result function
5. Saves raw SC, AC, DC values as .npy files for further analysis
"""

from gflower.utils.debug import GetMinSCAC_MaxDC, save_main_result
import os
import numpy as np
import argparse


def main(env, strategy, exp_name, repeat_time):
    target_path = os.path.join("logs", env, "eval", strategy, "smooth", exp_name)

    SC_record_ls = []
    AC_record_ls = []
    DC_record_ls = []

    for i in range(repeat_time):
        read_file = os.path.join(target_path, f"final_policy_record_{i}.txt")
        SC_Min, AC_min, DC_mean = GetMinSCAC_MaxDC(read_file)
        SC_record_ls.append(min(0.000, SC_Min))
        AC_record_ls.append(min(0.000, AC_min))
        DC_record_ls.append(DC_mean)
    
    SC_record_npy = np.array(SC_record_ls)
    AC_record_npy = np.array(AC_record_ls)
    DC_record_npy = np.array(DC_record_ls)

    SC_min = np.min(SC_record_npy)
    AC_min = np.min(AC_record_npy)
    SC_mean = np.mean(SC_record_npy)
    AC_mean = np.mean(AC_record_npy)
    SC_std = np.std(SC_record_npy)
    AC_std = np.std(AC_record_npy)
    DC_mean = np.mean(DC_record_npy)
    DC_std = np.std(DC_record_npy)

    print("save analysis")
    save_main_result(SC_min, AC_min, SC_mean, AC_mean, SC_std, AC_std, DC_mean, DC_std, target_path)

    print("save SCACDC source file")
    SC_record_file = os.path.join(target_path, "result_mainSC.npy")
    AC_record_file = os.path.join(target_path, "result_mainAC.npy")
    DC_record_file = os.path.join(target_path, "result_mainDC.npy")
    np.save(SC_record_file, SC_record_npy)
    np.save(AC_record_file, AC_record_npy)
    np.save(DC_record_file, DC_record_npy)

    print("exp_mainresult finish------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate main results from evaluation records')
    parser.add_argument('--env', type=str, default='maze2d-large-v1', help='Environment name')
    parser.add_argument('--strategy', type=str, default='ExpJac_ACSCDC_AllInOne_ALim0.9_H384', help='Strategy name')
    parser.add_argument('--exp_name', type=str, default='0.5_0.5_0.8_CBFt_0.005_CLF_t_0.05_NNema_ode1000_start0.97_euler_dsarobust0.1_10.0_1.5', help='Experiment name')
    parser.add_argument('--repeat_time', type=int, default=10, help='Number of repetitions')
    
    args = parser.parse_args()
    
    main(args.env, args.strategy, args.exp_name, args.repeat_time)







