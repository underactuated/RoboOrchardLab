import config_bridge_robotwin_lmdb_dataset  # noqa: F401
from config_holobrain_gd_common_bridge import (
    build_model,
    build_optimizer,
    build_processors,
    build_training_dataset,
    build_validation_dataset,
)


config = dict(
    hist_steps=1,
    pred_steps=64,
    chunk_size=4,
    embed_dims=256,
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.01,
    max_depth=1.2,
    num_depth=128,
    batch_size=8,
    max_step=9999,#99999,#100,#49999
    step_log_freq=50,#50,
    save_step_freq=1000,#5000,
    num_workers=8,#0,
    lr=1e-4,
    training_datasets=[
        "bridge_robotwin_lmdb_train",
    ],
    validation_datasets=["bridge_robotwin_lmdb_validation"],
    deploy_datasets=[
        "bridge_robotwin_lmdb_train",
    ],
    dst_wh=(320, 256),
    patch_size=64,
    multi_task=True,
    bert_checkpoint="google-bert/bert-base-uncased",
    #checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.0_GD/pretrain/model.safetensors",
    #checkpoint="/home/sergey/my_jobs/exp03/checkpoints/checkpoint_19/model.safetensors",
    checkpoint="/data/Users/sergey.pankov/holobrain/model/model.safetensors",
)
