import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="phys_anim",
        version="1.0",
        description="a framework for physics-based motion controllers",
        url="",
        author="Chen Tessler",
        packages=["phys_anim"],
        install_requires=[
            "lightning==2.3.3",
            "torch==2.2",
            "typer>=0.6.1",
            "wandb>=0.13.4",
            "transformers>=4.20.1",
            "hydra-core>=1.2.0",
            "matplotlib",
            "scikit-image",
            "opencv-python==4.5.5.64",
            "trimesh",
            "rtree==1.2.0",
            "setuptools==69.5.1",
            "rich",
            "smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git@master",
        ],
    )
