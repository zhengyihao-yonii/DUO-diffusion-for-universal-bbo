# -*- coding: utf-8 -*-
"""RobotPush Oracle（自 mcts-transfer PushReward），无 meta 数据依赖。"""
from __future__ import annotations

import numpy as np

from .push_utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation


class PushOracle:
    dim = 14
    xmin = [-5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0]
    xmax = [5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0]

    def __init__(self):
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

    @property
    def f_max(self):
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) + np.linalg.norm(
            np.array(self.gxy2) - np.array(self.sxy2)
        )

    def __call__(self, x):
        x = np.array(x, dtype=np.float64)
        if x.ndim > 1:
            raise ValueError("PushOracle expects 1d x")
        lb = np.array(self.xmin)
        ub = np.array(self.xmax)
        x = x * (ub - lb) + lb

        rx = float(x[0])
        ry = float(x[1])
        xvel = float(x[2])
        yvel = float(x[3])
        simu_steps = int(float(x[4]) * 10)
        init_angle = float(x[5])
        rx2 = float(x[6])
        ry2 = float(x[7])
        xvel2 = float(x[8])
        yvel2 = float(x[9])
        simu_steps2 = int(float(x[10]) * 10)
        init_angle2 = float(x[11])
        rtor = float(x[12])
        rtor2 = float(x[13])

        initial_dist = self.f_max

        world = b2WorldInterface(False)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = (
            "circle",
            1,
            0.01,
            0.05,
            0.01,
            "rectangle",
            (1, 0.3),
        )

        base = make_base(500, 500, world)
        body = create_body(base, world, "rectangle", (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, "circle", 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2, ry2), base, init_angle2, hand_shape, hand_size)
        ret1, ret2 = run_simulation(
            world, body, body2, robot, robot2, xvel, yvel, xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2
        )

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        result = initial_dist - ret1 - ret2
        return float(-1.0 * result)
