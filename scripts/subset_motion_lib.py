#!/usr/bin/env python3
"""
Script to create a subset of a motion library by sampling every N motions.
Might be useful if you realize your GPU cannot load large motion libraries.
"""

import torch
from pathlib import Path


def subset_motion_lib(input_path: str, output_path: str, sample_every: int = 200):
    """
    Load a motion library and create a subset by sampling every N motions.
    
    Args:
        input_path: Path to input .pt motion library
        output_path: Path to output .pt file
        sample_every: Take every Nth motion (default: 200)
    """
    print(f"Loading motion library from {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Get number of motions
    num_motions = len(data["motion_lengths"])
    print(f"Original motion library has {num_motions} motions")
    
    # Select motion indices (every sample_every motions)
    selected_indices = list(range(0, num_motions, sample_every))
    num_selected = len(selected_indices)
    print(f"Selecting {num_selected} motions (every {sample_every}th)")
    
    # Get the frame ranges for each selected motion
    length_starts = data["length_starts"]
    motion_num_frames = data["motion_num_frames"]
    
    # Collect frame indices for selected motions
    frame_indices = []
    new_motion_num_frames = []
    new_motion_lengths = []
    new_motion_dt = []
    new_motion_weights = []
    new_motion_files = []
    
    for idx in selected_indices:
        start = length_starts[idx].item()
        num_frames = motion_num_frames[idx].item()
        frame_indices.extend(range(start, start + num_frames))
        new_motion_num_frames.append(num_frames)
        new_motion_lengths.append(data["motion_lengths"][idx].item())
        new_motion_dt.append(data["motion_dt"][idx].item())
        new_motion_weights.append(data["motion_weights"][idx].item())
        if "motion_files" in data:
            new_motion_files.append(data["motion_files"][idx])
    
    frame_indices = torch.tensor(frame_indices, dtype=torch.long)
    
    # Create new data dictionary
    new_data = {}
    
    # Tensor fields that are indexed by frame
    frame_indexed_fields = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts"]
    if "lrs" in data and data["lrs"] is not None:
        frame_indexed_fields.append("lrs")
    
    for field in frame_indexed_fields:
        if field in data and data[field] is not None:
            new_data[field] = data[field][frame_indices]
            print(f"  {field}: {data[field].shape} -> {new_data[field].shape}")
    
    # Rebuild length_starts
    new_motion_num_frames_tensor = torch.tensor(new_motion_num_frames, dtype=torch.long)
    lengths_shifted = new_motion_num_frames_tensor.roll(1)
    lengths_shifted[0] = 0
    new_data["length_starts"] = lengths_shifted.cumsum(0)
    
    # Other motion-indexed fields
    new_data["motion_num_frames"] = new_motion_num_frames_tensor
    new_data["motion_lengths"] = torch.tensor(new_motion_lengths, dtype=torch.float32)
    new_data["motion_dt"] = torch.tensor(new_motion_dt, dtype=torch.float32)
    new_data["motion_weights"] = torch.tensor(new_motion_weights, dtype=torch.float32)
    
    if new_motion_files:
        new_data["motion_files"] = tuple(new_motion_files)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_data, output_path)
    
    print(f"\nSaved subset to {output_path}")
    print(f"  Motions: {num_motions} -> {num_selected}")
    print(f"  Total frames: {len(data['gts'])} -> {len(new_data['gts'])}")

