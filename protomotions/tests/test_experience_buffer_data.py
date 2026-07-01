# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rollout experience buffer utilities."""

import numpy as np
import torch

from protomotions.agents.utils.data import (
    DictDataset,
    ExperienceBuffer,
    swap_and_flatten01,
)


def test_swap_and_flatten01_keeps_trailing_shape_and_env_major_order():
    tensor = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

    flat = swap_and_flatten01(tensor)

    assert flat.shape == (6, 4)
    assert torch.equal(flat[0], tensor[0, 0])
    assert torch.equal(flat[1], tensor[1, 0])
    assert torch.equal(flat[2], tensor[0, 1])


def test_swap_and_flatten01_allows_none_and_batch_update_fills_rollout():
    assert swap_and_flatten01(None) is None

    buffer = ExperienceBuffer(num_envs=2, num_steps=2, device=torch.device("cpu"))
    buffer.register_key("obs", shape=(1,), dtype=torch.float)
    data = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    buffer.batch_update_data("obs", data)
    flattened = buffer.make_dict()["obs"]

    assert torch.equal(flattened, torch.tensor([[1.0], [3.0], [2.0], [4.0]]))
    assert buffer.store_dict["obs"] == 0


def test_experience_buffer_registers_updates_and_flattens_complete_rollout():
    buffer = ExperienceBuffer(num_envs=2, num_steps=3, device=torch.device("cpu"))
    buffer.register_key("obs", shape=(2,), dtype=torch.float)

    for step in range(3):
        buffer.update_data(
            "obs",
            step,
            torch.tensor(
                [
                    [step, step + 0.5],
                    [step + 10, step + 10.5],
                ],
                dtype=torch.float,
            ),
        )

    data = buffer.make_dict()

    assert data["obs"].shape == (6, 2)
    assert torch.equal(data["obs"][0], torch.tensor([0.0, 0.5]))
    assert torch.equal(data["obs"][1], torch.tensor([1.0, 1.5]))
    assert buffer.store_dict["obs"] == 0


def test_experience_buffer_make_dict_rejects_partially_filled_keys():
    buffer = ExperienceBuffer(num_envs=1, num_steps=2, device=torch.device("cpu"))
    buffer.register_key("rewards")
    buffer.update_data("rewards", 0, torch.tensor([1.0]))

    try:
        buffer.make_dict()
    except AssertionError as error:
        assert "Problem with 'rewards'" in str(error)
    else:
        raise AssertionError("Expected partially filled rollout data to fail")


def test_dict_dataset_wraps_shorter_tensors_by_modulo_indexing():
    dataset = DictDataset(
        batch_size=2,
        tensor_dict={
            "rollout": torch.arange(6),
            "expert": torch.tensor([10, 20]),
        },
        shuffle=False,
    )

    batch = dataset[1]

    assert torch.equal(batch["rollout"], torch.tensor([2, 3]))
    assert torch.equal(batch["expert"], torch.tensor([10, 20]))


def test_dict_dataset_shuffle_tracks_original_indices():
    np.random.seed(0)
    dataset = DictDataset(
        batch_size=2,
        tensor_dict={"values": torch.arange(6)},
        shuffle=True,
    )

    batch = dataset[0]

    assert batch["values"].shape == (2,)
    assert set(batch["values"].tolist()).issubset(set(range(6)))
    assert not np.array_equal(dataset.shuffled_to_original, np.arange(6))
