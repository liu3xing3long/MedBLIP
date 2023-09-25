
GPUS_PER_NODE=1
NTASK=1
PARTITION=MIA_LLM
JOB_NAME=MedBLIP

# dp
#srun -p Med -n1 -w $server --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=LLL --kill-on-bad-exit=1 \
srun -n ${NTASK} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${NTASK} \
    --kill-on-bad-exit=1 \
    python run_medblip_pretrain.py