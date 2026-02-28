# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test: Deformable-on-Deformable Interaction

A stiff deformable block drops onto a very soft deformable block.
Demonstrates how different stiffness (Young's modulus) values affect deformable body interactions.

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/tutorials/06_physics_properties/test_deformable_interaction.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test: deformable-on-deformable interaction.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Two origins to replicate the demo
    origins = [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Bottom: soft "couch" cube (very low stiffness, sits on ground)
    couch_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Couch",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.4, 0.4, 0.2),  # wide and flat
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.7)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=2000.0,  # very soft -- will squish under load
                poissons_ratio=0.45,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        debug_vis=True,
    )
    couches = DeformableObject(cfg=couch_cfg)

    # Top: stiffer "pillow" cube (drops onto couch)
    pillow_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Pillow",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.15, 0.15, 0.15),  # smaller
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.3)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=50000.0,  # stiffer -- holds shape better
                poissons_ratio=0.4,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        debug_vis=True,
    )
    pillows = DeformableObject(cfg=pillow_cfg)

    scene_entities = {"couches": couches, "pillows": pillows}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    couches = entities["couches"]
    pillows = entities["pillows"]

    sim_dt = sim.get_physics_dt()
    count = 0

    couch_kinematic_target = couches.data.nodal_kinematic_target.clone()
    pillow_kinematic_target = pillows.data.nodal_kinematic_target.clone()

    while simulation_app.is_running():
        # Reset every 400 steps
        if count % 400 == 0:
            count = 0

            # Reset couches
            nodal_state = couches.data.default_nodal_state_w.clone()
            couches.write_nodal_state_to_sim(nodal_state)
            couch_kinematic_target[..., :3] = nodal_state[..., :3]
            couch_kinematic_target[..., 3] = 1.0
            couches.write_nodal_kinematic_target_to_sim(couch_kinematic_target)
            couches.reset()

            # Reset pillows
            nodal_state = pillows.data.default_nodal_state_w.clone()
            pillows.write_nodal_state_to_sim(nodal_state)
            pillow_kinematic_target[..., :3] = nodal_state[..., :3]
            pillow_kinematic_target[..., 3] = 1.0
            pillows.write_nodal_kinematic_target_to_sim(pillow_kinematic_target)
            pillows.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting scene... Stiff pillow (red) drops onto soft couch (blue)")

        # Step
        couches.write_data_to_sim()
        pillows.write_data_to_sim()
        sim.step()
        count += 1
        couches.update(sim_dt)
        pillows.update(sim_dt)

        if count % 50 == 0:
            print(f"[Step {count}] Couch centers z: {couches.data.root_pos_w[:, 2]} | "
                  f"Pillow centers z: {pillows.data.root_pos_w[:, 2]}")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, -0.5, 1.0], target=[0.0, 0.0, 0.3])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
