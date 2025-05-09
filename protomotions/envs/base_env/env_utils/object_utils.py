import numpy as np
import trimesh


def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def get_object_heightmap(
    mesh, dim_x: int = 10, dim_y: int = 10, dim_multiplier: int = 10
):
    dim_x *= dim_multiplier
    dim_y *= dim_multiplier

    min_x, min_y, min_z = (
        mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].min(),
    )
    max_x, max_y, max_z = (
        mesh.vertices[:, 0].max(),
        mesh.vertices[:, 1].max(),
        mesh.vertices[:, 2].max(),
    )

    mesh.apply_translation([0.0, 0.0, -min_z])  # place the object on the ground

    x = np.linspace(min_x, max_x, dim_x)
    y = np.linspace(min_y, max_y, dim_y)
    X, Y = np.meshgrid(x, y, indexing="ij")  # Ensure consistent indexing
    pos2d = np.stack([X.ravel(), Y.ravel()], axis=-1)

    origins = np.zeros((dim_x * dim_y, 3))
    vectors = np.zeros((dim_x * dim_y, 3))

    origins[:, :2] = pos2d
    origins[:, 2] = source_height = max_z + 1.0

    vectors[:, 2] = -1.0

    # do the actual ray-mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[index_ray], vectors[index_ray])
    # convert depth to height
    height = source_height - depth

    height_map2d = np.zeros((dim_x, dim_y))
    for idx, h in zip(index_ray, height):
        # Calculate the correct row and column indices
        row = idx // dim_y
        col = idx % dim_y

        height_map2d[row, col] = h

    return rebin(height_map2d, (dim_x // dim_multiplier, dim_y // dim_multiplier))


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()
            ]
        )
    else:
        mesh = scene_or_mesh
    return mesh


def compute_bounding_box(mesh):
    min_x, min_y, min_z = (
        mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].min(),
    )
    max_x, max_y, max_z = (
        mesh.vertices[:, 0].max(),
        mesh.vertices[:, 1].max(),
        mesh.vertices[:, 2].max(),
    )

    return max_x - min_x, max_y - min_y, max_z - min_z, min_x, min_y, min_z


if __name__ == "__main__":
    from pathlib import Path

    # iterate over all objects in path
    bad_objects = []
    for obj_path in list(
        Path("../../data/assets/urdf/objects/train/LieDown/LargeSofas").rglob("*.obj")
    ) + list(Path("../../data/assets/urdf/objects/test/LargeSofas").rglob("*.obj")):
        print(obj_path)
        mesh = as_mesh(trimesh.load_mesh(obj_path))

        w_x, w_y, w_z, m_x, m_y, m_z = compute_bounding_box(mesh)

        max_w = max(w_x, w_y)
        heightmap = get_object_heightmap(
            mesh,
            dim_x=int(np.round(w_x / 0.1)),
            dim_y=int(np.round(w_y / 0.1)),
            dim_multiplier=10,
        )

        # heightmap = get_object_heightmap(mesh)
        # print(f"{w_x} {w_y} {w_z} {m_x} {m_y} {m_z}")
        print(heightmap[heightmap.shape[0] // 2, heightmap.shape[1] // 2])
        if heightmap[heightmap.shape[0] // 2, heightmap.shape[1] // 2] > 0.4:
            bad_objects.append(obj_path)
        # print(heightmap)
        # print(heightmap.shape)

        min_x, min_y, min_z = (
            mesh.vertices[:, 0].min(),
            mesh.vertices[:, 1].min(),
            mesh.vertices[:, 2].min(),
        )
        max_x, max_y, max_z = (
            mesh.vertices[:, 0].max(),
            mesh.vertices[:, 1].max(),
            mesh.vertices[:, 2].max(),
        )

    print("Bad objects")
    for obj in bad_objects:
        print(obj)

    exit(0)
    # obj_path = "../../data/assets/urdf/objects/train/Sofas/1bce3a3061de03251009233434be6ec0.obj"
    # obj_path = "../../data/assets/urdf/objects/test/HighStools/758649ba384b28856dc24db120ad1ab9.obj"
    # obj_path = "../../data/assets/urdf/objects/test/Armchairs/11040f463a3895019fb4103277a6b93.obj"
    obj_path = "../../data/assets/urdf/objects/incompatible/train/LargeSofas/1101146651cd32a1bd09c0f277d16187.obj"
    mesh = as_mesh(trimesh.load_mesh(obj_path))

    w_x, w_y, w_z, m_x, m_y, m_z = compute_bounding_box(mesh)

    max_w = max(w_x, w_y)
    heightmap = get_object_heightmap(
        mesh,
        dim_x=int(np.round(w_x / 0.1)),
        dim_y=int(np.round(w_y / 0.1)),
        dim_multiplier=10,
    )

    # heightmap = get_object_heightmap(mesh)
    print(f"{w_x} {w_y} {w_z} {m_x} {m_y} {m_z}")
    # print(heightmap)
    print(heightmap[heightmap.shape[0] // 2, heightmap.shape[1] // 2])
    print(heightmap.shape)

    min_x, min_y, min_z = (
        mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].min(),
    )
    max_x, max_y, max_z = (
        mesh.vertices[:, 0].max(),
        mesh.vertices[:, 1].max(),
        mesh.vertices[:, 2].max(),
    )

    import matplotlib.pyplot as plt

    # set axis based on x,y bounding box
    plt.imshow(heightmap)  # , extent=[min_y, max_y, min_x, max_x])
    plt.colorbar()
    # show grid lines
    plt.grid(True)
    plt.show()
