FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ARG USER_ID
ARG GROUP_ID
ARG WANDB_API_KEY

RUN apt-get update && apt-get install -y passwd && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $GROUP_ID usergroup && useradd -m -u $USER_ID -g $GROUP_ID user && mkdir -p /workspace && chown -R user:usergroup /workspace

WORKDIR /workspace

USER user

COPY requirements.txt ./ 
RUN pip install -r requirements.txt

RUN python -m wandb login $WANDB_API_KEY

CMD ["./folds.sh"]
