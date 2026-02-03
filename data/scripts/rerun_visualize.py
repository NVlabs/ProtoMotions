import argparse
import time
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
from pathlib import Path


class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        data_dir = Path(__file__).resolve().parents[1]
        robot_root = data_dir / "robot_description"
        match robot_type:
            case 'g1':
                urdf_path = robot_root / "g1" / "g1_29dof_rev_1_0.urdf"
                self.robot = pin.RobotWrapper.BuildFromURDF(
                    str(urdf_path),
                    [str(robot_root / "g1")],
                    pin.JointModelFreeFlyer(),
                )
                self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                       -0.15,0,0,0.3,-0.15,0,
                                       -0.15,0,0,0.3,-0.15,0,
                                       0,0,0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
            case _:
                print(robot_type)
                raise ValueError('Invalid robot type')
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Path to motion file (.csv or .npy)", required=True)
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    parser.add_argument('--fps', type=float, default=30.0, help="FPS for CSV files without fps metadata")
    args = parser.parse_args()

    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    input_path = Path(args.input)
    robot_type = args.robot_type

    def _read_csv_numeric(csv_path: Path) -> np.ndarray:
        arr = np.genfromtxt(csv_path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.float32)

    def _strip_frame_index_if_present(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] == 37:
            c0 = arr[:, 0]
            intish = np.all(np.abs(c0 - np.round(c0)) < 1e-3)
            if intish or arr[:, 1:].shape[1] == 36:
                return arr[:, 1:]
        return arr

    if input_path.suffix.lower() == ".npy":
        motion_obj = np.load(input_path, allow_pickle=True)
        if isinstance(motion_obj, np.ndarray) and motion_obj.shape == ():
            motion = motion_obj.item()
        else:
            motion = motion_obj
        root_trans = np.asarray(motion["root_trans"], dtype=np.float32)
        root_ori = np.asarray(motion["root_ori"], dtype=np.float32)
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
        data = np.concatenate([root_trans, root_ori, dof_pos], axis=1)
        fps = float(motion.get("fps", args.fps))
    elif input_path.suffix.lower() == ".csv":
        data = _read_csv_numeric(input_path)
        data = _strip_frame_index_if_present(data)
        fps = float(args.fps)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .csv or .npy")

    rerun_urdf = RerunURDF(robot_type)
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration)
        time.sleep(1.0 / fps)
