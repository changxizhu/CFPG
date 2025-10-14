#!/bin/bash
# run mujoco on ICS HPC


# ############################################################################
# #####################   PP domain with 9 predators   #######################
# ############################################################################
# generate a list of random numbers
RANDOM_SEED=(1 5 10 20 50 100 150 200 250 300)
PYTHON_DIR=~/anaconda3/envs/caddpg/bin
JOB_ID="p9"
SCENARIO_NAME="continuous_pred_prey_9a"
AGENT_CONF="None"
COMM_GRAPH="None"
PROGRAM_KEY="pp9a"
recordrm -rf .git_att="False"
T_MAX=4000000

MSG_DIM=2
OUTPUT_DIR=~/CFPG-master/results/pymarl_pp9a_M${MSG_DIM}
mkdir -p $OUTPUT_DIR

MIXER="qmix"
sbatch -J ca${JOB_ID} run_ics.sh $PYTHON_DIR $OUTPUT_DIR ${PROGRAM_KEY}"_fa_"${MIXER} "particle" $SCENARIO_NAME $AGENT_CONF "facmac_pp" $T_MAX $MSG_DIM 0 0 $COMM_GRAPH $MIXER $record_att "${RANDOM_SEED[@]}"

MIXER="vdn"
sbatch -J ca${JOB_ID} run_ics.sh $PYTHON_DIR $OUTPUT_DIR ${PROGRAM_KEY}"_m${MSG_DIM}_ca_"${MIXER} "particle" $SCENARIO_NAME $AGENT_CONF "comm_actor_pp" $T_MAX $MSG_DIM 0 0 $COMM_GRAPH $MIXER $record_att "${RANDOM_SEED[@]}"
sbatch -J cac${JOB_ID} run_ics.sh $PYTHON_DIR $OUTPUT_DIR ${PROGRAM_KEY}"_m${MSG_DIM}_att_ca_"${MIXER} "particle" $SCENARIO_NAME $AGENT_CONF "comm_attention_actor_pp" $T_MAX $MSG_DIM 0 0 $COMM_GRAPH $MIXER $record_att "${RANDOM_SEED[@]}"
MIXER="qmix"
sbatch -J ca${JOB_ID} run_ics.sh $PYTHON_DIR $OUTPUT_DIR ${PROGRAM_KEY}"_m${MSG_DIM}_ca_"${MIXER} "particle" $SCENARIO_NAME $AGENT_CONF "comm_actor_pp" $T_MAX $MSG_DIM 0 0 $COMM_GRAPH $MIXER $record_att "${RANDOM_SEED[@]}"
sbatch -J cac${JOB_ID} run_ics.sh $PYTHON_DIR $OUTPUT_DIR ${PROGRAM_KEY}"_m${MSG_DIM}_att_ca_"${MIXER} "particle" $SCENARIO_NAME $AGENT_CONF "comm_attention_actor_pp" $T_MAX $MSG_DIM 0 0 $COMM_GRAPH $MIXER $record_att "${RANDOM_SEED[@]}"

