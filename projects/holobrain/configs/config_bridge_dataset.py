# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)


def _bridge_tfds_path() -> str:
    return "/data/jerry/datasets/openx/resize_224"


dataset_config = dict(
    bridge_tfds_train=dict(
        root="./bridge_npz_export_smoke",
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
    ),
    bridge_tfds_validation=dict(
        root="./bridge_npz_export_smoke",
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
    ),
)


def build_transforms(
    config, mode, kinematics_config, t_base2world, scale_shift, num_joint
):
    import numpy as np

    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        DualArmKinematics,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    num_joint_per_arm = num_joint // 2 - 1
    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
        ]
    ).tolist()
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
        ),
    )

    kinematics = dict(type=DualArmKinematics, **kinematics_config)
    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)

    constant_depth = dict(
        type="robo_orchard_lab.dataset.robotwin.transforms:ReplaceDepthWithConstant",
        value=0.6,
    )

    if mode == "training":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            joint_state_noise,
            convert_dtype,
            constant_depth,
            kinematics,
            item_selection,
        ]
    elif mode == "validation":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            convert_dtype,
            constant_depth,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            convert_dtype,
            constant_depth,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return transforms


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    from bridge import BridgeHolobrainDataset, NpzBridgeBackend

    datasets = []
    for dataset_name, data_config in dataset_config.items():
        if (
            "bridge" not in dataset_names
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
        backend = NpzBridgeBackend(
            root=data_config["root"],
            max_episodes=config.get("bridge_max_episodes"),
        )
        dataset = BridgeHolobrainDataset(
            backend=backend,
            constant_depth=config.get("bridge_constant_depth", 0.6),
        )
        dataset.transforms = transforms
        dataset.dataset_name = dataset_name
        dataset.flag = 1000 + len(datasets)
        datasets.append(_TransformWrappedDataset(dataset))
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


class _TransformWrappedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms = getattr(dataset, "transforms", [])
        self.dataset_name = getattr(dataset, "dataset_name", None)
        self.flag = getattr(dataset, "flag", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        for transform in self.transforms:
            if transform is None:
                continue
            from robo_orchard_lab.utils.build import build

            transform = build(transform)
            data = transform(data)
        return data
