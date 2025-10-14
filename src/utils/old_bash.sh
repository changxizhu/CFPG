#############################################################################
######################   Mujoco with 2-Agent Humanoid #######################
#############################################################################

# # run programs with default settings (with batch size 320)
# OUTPUT_DIR=/home/changxiz/data/pymarl2_hv2_caddpg_basic_B100
# mkdir -p OUTPUT_DIR

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c2_Hv2_1 run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.001_m2_a1" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.001 \
#     2 \
#     1
# sleep 3

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c2_Hv2_2 run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.001_m2_a2" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.001 \
#     2 \
#     2
# sleep 3

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c2_Hv2_3 run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.001_m2_a3" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.001 \
#     2 \
#     3
# sleep 3



# # run programs with batch size 320, update rate of target networks is 0.01
# OUTPUT_DIR=/home/changxiz/data/pymarl_hv2_caddpg_tau0.01
# mkdir -p OUTPUT_DIR

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c_Hv2_1T run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.01_m2_a1" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.01 \
#     2 \
#     1
# sleep 3

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c_Hv2_2T run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.01_m2_a2" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.01 \
#     2 \
#     2
# sleep 3

# # run caddpg on mujoco with Humanoid-v2
# sbatch -J c_Hv2_3T run_cpu.sh $PYTHON_DIR $OUTPUT_DIR \
#     1 \
#     "1_Hv2_b32_c0.001_t0.01_m2_a3" \
#     "mujoco_multi" \
#     "Humanoid-v2" \
#     "9|8" \
#     "caddpg_mamujoco" \
#     32 \
#     0.001 \
#     0.01 \
#     2 \
#     3
# sleep 3




#############################################################################
######################   Mujoco with 2-Agent Humanoid      ##################
#############################################################################

# JOB_ID="1_hu"
# SCENARIO_NAME="Humanoid-v2"
# AGENT_CONF="9|8"
# PROGRAM_KEY=${RUN_TIMES}"_humanoid"
# # run programs with default settings (with batch size 100)
# OUTPUT_DIR=/home/changxiz/data/pymarl_humanoid_E1_SPV2_MLP_M${MSG_DIM}_4M
# mkdir -p OUTPUT_DIR

# # run caddpg with type 1
# sbatch -J c${JOB_ID}_1 run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY}"_m${MSG_DIM}_a1" "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "caddpg_mamujoco" $MSG_DIM 1
# # run caddpg with type 2
# sbatch -J c${JOB_ID}_2 run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY}"_m${MSG_DIM}_a2" "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "caddpg_mamujoco" $MSG_DIM 2
# # run caddpg with type 3
# sbatch -J c${JOB_ID}_3 run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY}"_m${MSG_DIM}_a3" "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "caddpg_mamujoco" $MSG_DIM 3
# # run caddpg with type 4
# sbatch -J c${JOB_ID}_4 run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY}"_m${MSG_DIM}_a4" "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "caddpg_mamujoco" $MSG_DIM 4
# # run caddpg with type 5
# sbatch -J c${JOB_ID}_5 run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY}"_m${MSG_DIM}_a5" "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "caddpg_mamujoco" $MSG_DIM 5
# # run facmac
# sbatch -J f${JOB_ID} run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY} "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "facmac_mamujoco" 0 0
# # # # run maddpg
# sbatch -J m${JOB_ID} run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY} "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "maddpg_mamujoco" 0 0
# # # # run g2anet
# sbatch -J g${JOB_ID} run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY} "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "g2anet_mamujoco" 0 0 
# # run central cadddpg
# sbatch -J cc${JOB_ID} run_surf.sh $PYTHON_DIR $OUTPUT_DIR $RUN_TIMES ${PROGRAM_KEY} "mujoco_multi" $SCENARIO_NAME $AGENT_CONF "central_caddpg_mamujoco" $MSG_DIM 0 



#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
