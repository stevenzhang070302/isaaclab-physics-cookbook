# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test: Mass vs Deformable Body Response

A heavy rigid block (50kg) and a light rigid block (1kg) drop onto identical soft deformable cubes.
Observe how mass affects deformation.

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/tutorials/06_physics_properties/test_mass_deformable.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test: mass vs deformable body response.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
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

    # Two origins: left for heavy, right for light
    origins = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Identical soft deformable cubes at both origins (on the ground)
    soft_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/SoftCube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.3, 0.3),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=5000.0,  # very soft
                poissons_ratio=0.45,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
        debug_vis=True,
    )
    soft_cubes = DeformableObject(cfg=soft_cfg)

    # Heavy rigid block (red, 50kg) at origin 0
    heavy_cfg = RigidObjectCfg(
        prim_path="/World/Origin0/HeavyBlock",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    heavy_block = RigidObject(cfg=heavy_cfg)

    # Light rigid block (blue, 1kg) at origin 1
    light_cfg = RigidObjectCfg(
        prim_path="/World/Origin1/LightBlock",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    light_block = RigidObject(cfg=light_cfg)

    scene_entities = {
        "soft_cubes": soft_cubes,
        "heavy_block": heavy_block,
        "light_block": light_block,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    soft_cubes = entities["soft_cubes"]
    heavy_block = entities["heavy_block"]
    light_block = entities["light_block"]

    sim_dt = sim.get_physics_dt()
    count = 0

    # Cache kinematic targets for deformable reset
    nodal_kinematic_target = soft_cubes.data.nodal_kinematic_target.clone()

    while simulation_app.is_running():
        # Reset every 300 steps
        if count % 300 == 0:
            count = 0

            # Reset soft cubes
            nodal_state = soft_cubes.data.default_nodal_state_w.clone()
            soft_cubes.write_nodal_state_to_sim(nodal_state)
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            soft_cubes.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            soft_cubes.reset()

            # Reset heavy block
            root_state = heavy_block.data.default_root_state.clone()
            heavy_block.write_root_pose_to_sim(root_state[:, :7])
            heavy_block.write_root_velocity_to_sim(root_state[:, 7:])
            heavy_block.reset()

            # Reset light block
            root_state = light_block.data.default_root_state.clone()
            light_block.write_root_pose_to_sim(root_state[:, :7])
            light_block.write_root_velocity_to_sim(root_state[:, 7:])
            light_block.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting scene... Heavy=50kg (red/left), Light=1kg (blue/right)")

        # Step simulation
        soft_cubes.write_data_to_sim()
        heavy_block.write_data_to_sim()
        light_block.write_data_to_sim()
        sim.step()
        count += 1
        soft_cubes.update(sim_dt)
        heavy_block.update(sim_dt)
        light_block.update(sim_dt)

        if count % 50 == 0:
            print(f"[Step {count}] Heavy block z: {heavy_block.data.root_pos_w[:, 2].item():.3f} | "
                  f"Light block z: {light_block.data.root_pos_w[:, 2].item():.3f} | "
                  f"Soft cube centers z: {soft_cubes.data.root_pos_w[:, 2]}")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.3])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
