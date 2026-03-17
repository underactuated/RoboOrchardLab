import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from script import eval_policy as robotwin_eval


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    video_size = None

    get_model = robotwin_eval.eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = robotwin_eval.get_embodiment_config(
        args["left_robot_file"]
    )
    args["right_embodiment_config"] = robotwin_eval.get_embodiment_config(
        args["right_robot_file"]
    )

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = (
            str(embodiment_type[0]) + "+" + str(embodiment_type[1])
        )

    save_dir = Path(
        f"eval_result/{task_name}/{policy_name}/{task_config}/"
        f"{ckpt_setting}/{current_time}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        camera_config = robotwin_eval.get_camera_config(
            args["camera"]["head_camera_type"]
        )
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = save_dir

    print("============= Config =============\n")
    print(
        "\033[95mMessy Table:\033[0m "
        + str(args["domain_randomization"]["cluttered_table"])
    )
    print(
        "\033[95mRandom Background:\033[0m "
        + str(args["domain_randomization"]["random_background"])
    )
    if args["domain_randomization"]["random_background"]:
        print(
            " - Clean Background Rate: "
            + str(args["domain_randomization"]["clean_background_rate"])
        )
    print(
        "\033[95mRandom Light:\033[0m "
        + str(args["domain_randomization"]["random_light"])
    )
    if args["domain_randomization"]["random_light"]:
        print(
            " - Crazy Random Light Rate: "
            + str(args["domain_randomization"]["crazy_random_light_rate"])
        )
    print(
        "\033[95mRandom Table Height:\033[0m "
        + str(args["domain_randomization"]["random_table_height"])
    )
    print(
        "\033[95mRandom Head Camera Distance:\033[0m "
        + str(args["domain_randomization"]["random_head_camera_dis"])
    )
    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", {args['camera']['collect_head_camera']}"
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", {args['camera']['collect_wrist_camera']}"
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    task_env = robotwin_eval.class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(
        args["left_embodiment_config"]["arm_joints_name"][0]
    )
    usr_args["right_arm_dim"] = len(
        args["right_embodiment_config"]["arm_joints_name"][1]
    )

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    test_num = int(usr_args.get("test_num", 100))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = robotwin_eval.eval_policy(
        task_name,
        task_env,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
    )

    topk_success_rate = sorted([suc_num], reverse=True)[:topk]

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(topk_success_rate) / test_num)))

    print(f"Data has been saved to {file_path}")


if __name__ == "__main__":
    from script.test_render import Sapien_TEST

    Sapien_TEST()
    usr_args = robotwin_eval.parse_args_and_config()
    main(usr_args)
