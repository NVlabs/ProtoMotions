import typer
import os


def main(
    motion_file: str,
    simulator: str,
    robot: str,
    num_envs: int = 1,
    extra_args: str = "",
):
    command = f"python protomotions/eval_agent.py +base=[fabric,structure] +exp=deepmimic_mlp +robot={robot} +simulator={simulator} +checkpoint=null +training_max_steps=1 +motion_file={motion_file} env.config.sync_motion=True ref_respawn_offset=0 +headless=False num_envs={num_envs} {extra_args} +experiment_name=debug"
    os.system(command)


if __name__ == "__main__":
    typer.run(main)
