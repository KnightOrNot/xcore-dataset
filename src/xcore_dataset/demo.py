"""Explicitly synthetic data for exercising the full writer without hardware."""

import numpy as np
from PIL import Image
from xcore_adapters import pose10

from .raw import FORMAT, STREAMS, EpisodeWriter


def make_demo(path, frames=12, fps=10):
    if frames < 2 or fps < 1:
        raise ValueError("demo needs at least two frames and positive fps")
    camera = dict(
        serial="SIMULATED",
        depth_scale_m_per_unit=0.001,
        color_intrinsics={
            "width": 64,
            "height": 48,
            "fx": 60,
            "fy": 60,
            "ppx": 32,
            "ppy": 24,
        },
        depth_intrinsics={
            "width": 64,
            "height": 48,
            "fx": 60,
            "fy": 60,
            "ppx": 32,
            "ppy": 24,
        },
        depth_to_color={
            "rotation": np.eye(3).ravel().tolist(),
            "translation_m": [0, 0, 0],
        },
    )
    meta = dict(
        format=FORMAT,
        simulated=True,
        task="合成数据：移动夹爪",
        position_unit="m",
        rotation="columns_0_1",
        base_frame="simulated_base",
        tcp="simulated_tcp",
        action_source="requested_absolute_target",
        cameras={"cam1": camera, "cam2": camera},
    )
    writer = EpisodeWriter(path, meta)
    try:
        for i in range(frames):
            stamp = 1_000_000_000 + round(i * 1e9 / fps)
            t = np.eye(4)
            t[:3, 3] = [i * 0.001, 0.2, 0.3]
            state = pose10(t, 0.25)
            t[0, 3] += 0.002
            joints = [0.1 + i * 0.001, -0.2, 0.3, 0.0, 0.5, 1.0]
            target_joints = [value + 0.002 for value in joints]
            row = dict(
                timestamp_ns=stamp,
                state=state.tolist(),
                action=pose10(t, 0.3).tolist(),
                joints_rad=joints,
                requested_target_rad=target_joints,
                target_rad=target_joints,
                closure=0.25,
                target_closure=0.3,
                wrench=[0.0] * 12,
                wrench_valid=[False, False],
                images={},
            )
            if i == 2:
                row["wrench"] = [float(j) / 10 for j in range(12)]
                row["wrench_valid"] = [True, True]
            for s in STREAMS:
                name = f"{s}/{i:06}.png"
                target = writer.path / name
                target.parent.mkdir(exist_ok=True)
                if s.endswith("depth"):
                    a = np.full(
                        (48, 64),
                        (2000 if s.startswith("cam2") else 1000) + i,
                        dtype=np.uint16,
                    )
                    a[0, 0] = 0
                    a[0, 1] = 65535
                else:
                    a = np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8)
                    a[:, :, 1] = 80
                    if s.startswith("cam2"):
                        a[:, :, 2] = 160
                Image.fromarray(a).save(target)
                row["images"][s] = dict(
                    path=name,
                    timestamp_ns=stamp,
                    clock_domain="remote_monotonic",
                    uncertainty_ns=0,
                    frame_number=i,
                )
            writer.append(row)
        writer.commit()
    finally:
        writer.close()
    return writer.path
