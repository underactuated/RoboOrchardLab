from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

from config_bridge_dataset import build_transforms

import os

# data folder path, depends on the environment variable DATASET_ROOT,
# if not set, use the default path "./bridge_lmdb_smoke"
def _lmdb_path(split="train") -> str:
    dataset_root = os.getenv("DATASET_ROOT")
    split_path = dict(
        train = "./bridge_lmdb_smoke",
        validation = "./bridge_lmdb_smoke"
    )
    if dataset_root:
        split_path["train"] = "Users/sergey.pankov/holobrain/bridge/train"
        split_path["validation"] = "Users/sergey.pankov/holobrain/bridge/validation"
        return os.path.join(
            dataset_root, split_path[split]
        )
    return split_path[split]

base_dataset_split_config = dict(
    paths=[None],
    scale_shift=[
        [1.12735104, -0.11648428],
        [1.45046443, 1.35436516],
        [1.5324732, 1.45750941],
        [1.80842297, -0.01855904],
        [1.46318083, 0.16631192],
        [2.79637467, 0.24332368],
        [0.5, 0.5],
        [1.12735104, -0.11648428],
        [1.45046443, 1.35436516],
        [1.5324732, 1.45750941],
        [1.80842297, -0.01855904],
        [1.46318083, 0.16631192],
        [2.79637467, 0.24332368],
        [0.5, 0.5],
    ],
    num_joint=14,
    cam_names=["front_camera"],
    kinematics_config=dict(
        urdf="./urdf/arx5/arx5_description_isaac.urdf",
    ),
    T_base2world=[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

(train_ds_conf := base_dataset_split_config.copy())["paths"]=[_lmdb_path("train")]
(validation_ds_conf := base_dataset_split_config.copy())["paths"]=[_lmdb_path("validation")]

dataset_config = dict(
    bridge_robotwin_lmdb_train=train_ds_conf,
    bridge_robotwin_lmdb_validation=validation_ds_conf,
)

# dataset_config = dict(
#     bridge_robotwin_lmdb_train=dict(
#         paths=["./bridge_lmdb_smoke"],
#         scale_shift=[
#             [1.12735104, -0.11648428],
#             [1.45046443, 1.35436516],
#             [1.5324732, 1.45750941],
#             [1.80842297, -0.01855904],
#             [1.46318083, 0.16631192],
#             [2.79637467, 0.24332368],
#             [0.5, 0.5],
#             [1.12735104, -0.11648428],
#             [1.45046443, 1.35436516],
#             [1.5324732, 1.45750941],
#             [1.80842297, -0.01855904],
#             [1.46318083, 0.16631192],
#             [2.79637467, 0.24332368],
#             [0.5, 0.5],
#         ],
#         num_joint=14,
#         cam_names=["front_camera"],
#         kinematics_config=dict(
#             urdf="./urdf/arx5/arx5_description_isaac.urdf",
#         ),
#         T_base2world=[
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#         ],
#     ),
#     bridge_robotwin_lmdb_validation=dict(
#         paths=["./bridge_lmdb_smoke"],
#         scale_shift=[
#             [1.12735104, -0.11648428],
#             [1.45046443, 1.35436516],
#             [1.5324732, 1.45750941],
#             [1.80842297, -0.01855904],
#             [1.46318083, 0.16631192],
#             [2.79637467, 0.24332368],
#             [0.5, 0.5],
#             [1.12735104, -0.11648428],
#             [1.45046443, 1.35436516],
#             [1.5324732, 1.45750941],
#             [1.80842297, -0.01855904],
#             [1.46318083, 0.16631192],
#             [2.79637467, 0.24332368],
#             [0.5, 0.5],
#         ],
#         num_joint=14,
#         cam_names=["front_camera"],
#         kinematics_config=dict(
#             urdf="./urdf/arx5/arx5_description_isaac.urdf",
#         ),
#         T_base2world=[
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#         ],
#     ),
# )


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    datasets = []
    for dataset_name, data_config in dataset_config.items():
        if (
            "bridge_robotwin_lmdb" not in dataset_names
            and dataset_name not in dataset_names
        ):
            continue
        transforms = build_transforms(
            config,
            mode,
            data_config["kinematics_config"],
            data_config["T_base2world"],
            data_config["scale_shift"],
            data_config["num_joint"],
        )
        dataset = RoboTwinLmdbDataset(
            paths=data_config["paths"],
            task_names=config.get("task_names"),
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=dataset_name,
            cam_names=data_config["cam_names"],
            T_base2world=data_config["T_base2world"],
            reset_step=1000,
        )
        datasets.append(dataset)
    return datasets


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    processors = {}
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue
        transforms = build_transforms(
            config,
            "deploy",
            data_config["kinematics_config"],
            data_config["T_base2world"],
            data_config["scale_shift"],
            data_config["num_joint"],
        )
        processor = HoloBrainProcessor(
            HoloBrainProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                transforms=transforms,
                cam_names=data_config["cam_names"],
            )
        )
        processors[dataset_name] = processor
    return processors
