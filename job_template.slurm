#!/bin/bash
#SBATCH --job-name=jhuedo-clip_{DATASET}_lr{LR}_b{BATCH_SIZE}    # Nombre del trabajo
#SBATCH --output=slurm_log/%j.out                                   # Nombre del archivo de salida
#SBATCH --error=slurm_log/%j.err                                    # Nombre del archivo de error
#SBATCH --partition=turing                                          # Cola (partición) a la que enviar el trabajo
#SBATCH --cpus-per-task=1                                           # Número de CPUs por tarea
#SBATCH --mem=8G                                                    # Memoria por nodo
#SBATCH --gres=gpu:1                                                # Número de GPUs
#SBATCH --gres=shard:8                                              # Memoria GPU estimada
#SBATCH --time=3-00:00:00                                             # Tiempo de ejecucion limite (D-HH:MM:SS)

echo "Iniciando trabajo en `hostname` a las `date` $SLURM_IDX"
nvidia-smi

# Ejecutar el contenedor con acceso a la GPU
DOCKER_VOLUME_PATH=`pwd`

echo "Construyendo imagen..."
docker build --build-arg WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2) \
             --build-arg USER_ID=$(id -u) \
             --build-arg GROUP_ID=$(id -g) \
             -t jhuedo-clip-1.0 .

echo "Ejecutando contenedor..."
docker run --name ${SLURM_JOB_ID}_jhuedo-clip-1.0_${BASHPID} --gpus device=$SLURM_IDX --rm \
    -v $DOCKER_VOLUME_PATH:/workspace \
    -e LR={LR} \
    -e K_FOLDS={K_FOLDS} \
    -e BATCH_SIZE={BATCH_SIZE} \
    -e EPOCHS={EPOCHS} \
    -e VOCAB_SIZE={VOCAB_SIZE} \
    -e MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} \
    -e DATASET={DATASET} \
    jhuedo-clip-1.0
