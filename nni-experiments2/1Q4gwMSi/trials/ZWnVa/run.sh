#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='1Q4gwMsi'
export NNI_SYS_DIR='/home/dsi/haimisakov/nni-experiments/1Q4gwMsi/trials/ZWnVa'
export NNI_TRIAL_JOB_ID='ZWnVa'
export NNI_OUTPUT_DIR='/home/dsi/haimisakov/nni-experiments/1Q4gwMsi/trials/ZWnVa'
export NNI_TRIAL_SEQ_ID='650'
export NNI_CODE_DIR='/home/dsi/haimisakov/NeuralNetworksAfterProposalLab'
export CUDA_VISIBLE_DEVICES='0,1'
cd $NNI_CODE_DIR
eval python main_for_three_models.py --nni 1 --task_number 3 --dataset cirrhosis --device_num 0 2>"/home/dsi/haimisakov/nni-experiments/1Q4gwMsi/trials/ZWnVa/stderr"
echo $? `date +%s%3N` >'/home/dsi/haimisakov/nni-experiments/1Q4gwMsi/trials/ZWnVa/.nni/state'