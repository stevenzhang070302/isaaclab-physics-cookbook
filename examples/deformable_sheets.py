# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test: Deformable Body Motion (Cloth-like)

Thin deformable sheets dropped from height with initial horizontal velocity.
Demonstrates deformable body dynamics -- flexing, bending, and settling on impact.
(IsaacLab has no dedicated cloth/particle API, so we approximate with thin deformable meshes.)

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/tutorials/06_physics_properties/test_deformable_motion.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test: deformable body motion (cloth-like).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Two origins
    origins = [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid target platforms (cylinders to land on)
    platform_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Platform",
        spawn=sim_utils.CylinderCfg(
            radius=0.2,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )
    platforms = RigidObject(cfg=platform_cfg)

    # Thin deformable sheets (cloth-like) -- thin cuboid
    sheet_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Sheet",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.3, 0.02),  # thin and flat
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.8)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=1000.0,  # very flexible
                poissons_ratio=0.3,
                elasticity_damping=0.01,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
        debug_vis=True,
    )
    sheets = DeformableObject(cfg=sheet_cfg)

    scene_entities = {"platforms": platforms, "sheets": sheets}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    platforms = entities["platforms"]
    sheets = entities["sheets"]

    sim_dt = sim.get_physics_dt()
    count = 0

    nodal_kinematic_target = sheets.data.nodal_kinematic_target.clone()

    while simulation_app.is_running():
        # Reset every 400 steps
        if count % 400 == 0:
            count = 0

            # Reset sheets with slight random orientation
            nodal_state = sheets.data.default_nodal_state_w.clone()
            # Give initial downward + slight horizontal velocity to simulate "throwing"
            nodal_state[..., 3] = 0.3   # vx - horizontal motion
            nodal_state[..., 5] = -0.5  # vz - downward
            sheets.write_nodal_state_to_sim(nodal_state)

            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0  # free all vertices
            sheets.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            sheets.reset()

            # Reset platforms (kinematic, stays put)
            root_state = platforms.data.default_root_state.clone()
            platforms.write_root_pose_to_sim(root_state[:, :7])
            platforms.write_root_velocity_to_sim(root_state[:, 7:])
            platforms.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting scene... Sheets drop onto platforms")

        # Step
        sheets.write_data_to_sim()
        platforms.write_data_to_sim()
        sim.step()
        count += 1
        sheets.update(sim_dt)
        platforms.update(sim_dt)

        if count % 50 == 0:
            print(f"[Step {count}] Sheet centers z: {sheets.data.root_pos_w[:, 2]}")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, -1.0, 1.5], target=[0.0, 0.0, 0.5])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
