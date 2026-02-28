# Isaac Sim Physics Simulation Writer

You are helping write physics simulation scripts for NVIDIA Isaac Sim / IsaacLab.

## Instructions

When the user describes a physics scenario they want to simulate, create a complete runnable IsaacLab script following the patterns below. Ask the user which physics properties they want to test if unclear.

## Supported Physics Types (proven working)

### 1. Rigid Bodies
- Use `RigidObjectCfg` with shape spawners (`ConeCfg`, `SphereCfg`, `CuboidCfg`, `CylinderCfg`)
- Properties: `RigidBodyPropertiesCfg`, `MassPropertiesCfg`, `CollisionPropertiesCfg`
- Materials: `RigidBodyMaterialCfg(static_friction, dynamic_friction, restitution)`
- Good for: mass, restitution/bounciness, friction, collisions

### 2. Deformable Bodies
- Use `DeformableObjectCfg` with MESH spawners only (`MeshCuboidCfg`, `MeshSphereCfg`, `MeshCylinderCfg`)
- Shape spawners (CuboidCfg, SphereCfg) do NOT support deformable — must use Mesh variants
- Properties: `DeformableBodyPropertiesCfg(youngs_modulus, poissons_ratio, damping_scale)`
- Materials: `DeformableBodyMaterialCfg(youngs_modulus, poissons_ratio, damping_scale)`
- Good for: soft bodies, cushions, elastic objects, cloth-like behavior

### 3. Particles / Fluids
- NOT currently working in IsaacLab standalone scripts
- IsaacLab's SimulationContext prevents particle simulation from running
- Particles work ONLY through Isaac Sim's built-in demo browser (Window > Physics > Demo Scenes)
- If the user asks for particles/fluids, warn them and suggest deformable alternatives

## Script Template

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
<TITLE>

<DESCRIPTION>

.. code-block:: bash

    isaaclab -p <SCRIPT_PATH>

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="<DESCRIPTION>")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # --- Create objects here ---
    # scene_entities = {...}
    # return scene_entities


def run_simulator(sim, entities):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            # Reset objects to initial state
            for obj in entities.values():
                if isinstance(obj, RigidObject):
                    root_state = obj.data.default_root_state.clone()
                    obj.write_root_pose_to_sim(root_state[:, :7])
                    obj.write_root_velocity_to_sim(root_state[:, 7:])
                    obj.reset()
                elif isinstance(obj, DeformableObject):
                    nodal_state = obj.data.default_nodal_state_w.clone()
                    obj.write_nodal_state_to_sim(nodal_state)
                    nodal_kinematic_target = obj.data.nodal_kinematic_target.clone()
                    nodal_kinematic_target[..., :3] = nodal_state[..., :3]
                    nodal_kinematic_target[..., 3] = 1.0
                    obj.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
                    obj.reset()
            print("[INFO]: Resetting object states...")

        # Write data, step, update
        for obj in entities.values():
            obj.write_data_to_sim()
        sim.step()
        count += 1
        for obj in entities.values():
            obj.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.5])

    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

## Rigid Body Example (Restitution)

```python
cfg = RigidObjectCfg(
    prim_path="/World/Origin/Sphere",
    spawn=sim_utils.SphereCfg(
        radius=0.1,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(R, G, B)),
        physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.9),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(X, Y, Z)),
)
sphere = RigidObject(cfg=cfg)
```

## Deformable Body Example

```python
cfg = DeformableObjectCfg(
    prim_path="/World/Origin/SoftBody",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.5, 0.5, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=5000.0,    # stiffness (low=soft, high=rigid)
            poissons_ratio=0.4,       # volume preservation
            damping_scale=0.1,        # energy dissipation
        ),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
soft = DeformableObject(cfg=cfg)
```

## Key Rules

1. Multiple origins: Use `origins = [[x, 0, 0] for x in positions]` for side-by-side comparison
2. Deformable sim loop uses `write_nodal_state_to_sim()` + `write_nodal_kinematic_target_to_sim()` instead of `write_root_pose_to_sim()`
3. Regex in prim_path: Use `/World/Origin.*/ObjectName` to match multiple origins
4. Always use `MeshCuboidCfg`/`MeshSphereCfg` for deformables (NOT `CuboidCfg`/`SphereCfg`)
5. Run with: `isaaclab -p path/to/script.py` (or `isaaclab.bat` on Windows)

## User's argument: $ARGUMENTS
