#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --output=results/Logs/R-%x.%j.out
#SBATCH --error=results/Logs/R-%x.%j.err

python_dir=$1
output_dir=$2
exp_key=$3
env_config=$4
scenario_name=$5
agent_conf=$6
config_name=$7
t_max=$8
msg_dim=$9
ala_type=${10}
q_coeff=${11}
comm_graph=${12}
mixer=${13}
record_att=${14}
random_seed=(${@:15:25})

mkdir -p $output_dir
export PYMARL_RESULT_DIR=$output_dir
export OMP_NUM_THREADS=1
source ~/.bashrc
# conda activate caddpg

echo "Running Train_${exp_key}_$(date +%F)"
echo "${random_seed[@]}"
# echo ${random_seed[1]}

for i in `seq 1 $SLURM_NTASKS`; do
  echo ${exp_key}'_s'${random_seed[$(($i-1))]}
  $python_dir/python src/main.py --config=$config_name \
    --env-config=$env_config with env_args.scenario_name=$scenario_name env_args.agent_conf=$agent_conf \
    t_max=$t_max \
    msg_dim=$msg_dim \
    ala_type=$ala_type \
    exp_key=${exp_key}'_s'${random_seed[$(($i-1))]} \
    q_coeff=$q_coeff \
    comm_graph=$comm_graph \
    mixer=$mixer \
    seed=${random_seed[$(($i-1))]} \
    record_att=${record_att} &
done
# wait for programs
wait
