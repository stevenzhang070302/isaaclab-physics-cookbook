# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test: Restitution (Bounciness)

Four rigid spheres with different restitution values (0.0, 0.3, 0.6, 0.95) dropped from the same height.
Observe different bounce heights.

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/tutorials/06_physics_properties/test_restitution.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test: restitution (bounciness) comparison.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

# Restitution values and colors for each sphere
RESTITUTIONS = [0.0, 0.3, 0.6, 0.95]
COLORS = [
    (0.8, 0.0, 0.0),  # red   - no bounce
    (0.8, 0.5, 0.0),  # orange - low bounce
    (0.0, 0.6, 0.0),  # green  - medium bounce
    (0.0, 0.2, 0.8),  # blue   - high bounce
]
DROP_HEIGHT = 2.0


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Ground material with high restitution so bounces are visible
    ground_material = sim_utils.RigidBodyMaterialCfg(
        restitution=1.0,
        restitution_combine_mode="max",
    )
    ground_material.func("/World/defaultGroundPlane/GroundMaterial", ground_material)

    # Create 4 spheres in a row, each with different restitution
    spheres = {}
    origins = []
    for i, (rest, color) in enumerate(zip(RESTITUTIONS, COLORS)):
        x_pos = (i - 1.5) * 0.5  # spread them out
        origins.append([x_pos, 0.0, 0.0])
        sim_utils.create_prim(f"/World/Pos{i}", "Xform", translation=[x_pos, 0.0, 0.0])

        cfg = RigidObjectCfg(
            prim_path=f"/World/Pos{i}/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.08,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=rest),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.3),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DROP_HEIGHT)),
        )
        spheres[f"sphere_{i}"] = RigidObject(cfg=cfg)

    scene_entities = spheres
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    count = 0
    sphere_list = [entities[f"sphere_{i}"] for i in range(4)]

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            for sphere in sphere_list:
                root_state = sphere.data.default_root_state.clone()
                sphere.write_root_pose_to_sim(root_state[:, :7])
                sphere.write_root_velocity_to_sim(root_state[:, 7:])
                sphere.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting spheres...")
            print(f"  Restitution values: {RESTITUTIONS}")
            print(f"  Drop height: {DROP_HEIGHT}m")

        # Step
        for sphere in sphere_list:
            sphere.write_data_to_sim()
        sim.step()
        count += 1
        for sphere in sphere_list:
            sphere.update(sim_dt)

        # Log bounce heights
        if count % 30 == 0:
            heights = [sphere_list[i].data.root_pos_w[:, 2].item() for i in range(4)]
            print(f"[Step {count:3d}] Heights -> "
                  f"r=0.0: {heights[0]:.3f}  r=0.3: {heights[1]:.3f}  "
                  f"r=0.6: {heights[2]:.3f}  r=0.95: {heights[3]:.3f}")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -2.5, 2.0], target=[0.0, 0.0, 0.5])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
