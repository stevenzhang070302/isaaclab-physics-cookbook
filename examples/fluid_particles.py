# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test: Fluid Particle Dynamics (Standalone — bypasses SimulationContext)

Uses Isaac Sim's native PhysX APIs directly, matching the pattern from
omni.physx.demos that is proven to work.

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/tutorials/06_physics_properties/test_fluid_particles.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test: fluid particle dynamics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
import omni.timeline
import omni.physx

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt
from omni.physx.scripts import physicsUtils, particleUtils


def create_scene():
    """Creates the full scene using raw USD + PhysX APIs (no SimulationContext)."""
    stage = omni.usd.get_context().get_stage()

    # Stage settings
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Default prim
    default_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(default_prim.GetPrim())

    # --- Physics scene (matching demo pattern) ---
    physics_scene_path = Sdf.Path("/World/physicsScene")
    scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateSolverTypeAttr().Set("TGS")
    print(f"[INFO]: Physics scene created at {physics_scene_path} (GPU dynamics + TGS solver)")

    # --- Ground plane (collision-enabled cube) ---
    ground = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    ground.CreateSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.5))
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(100.0, 100.0, 1.0))
    ground.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)]))

    # --- Light ---
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr().Set(2000.0)

    # --- Particle system ---
    particle_system_path = Sdf.Path("/World/ParticleSystem")
    material_path = Sdf.Path("/World/ParticleMaterial")
    particle_set_path = Sdf.Path("/World/Particles")

    particle_contact_offset = 0.03  # 3cm

    particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
    particle_system.CreateSimulationOwnerRel().SetTargets([physics_scene_path])
    particle_system.CreateParticleContactOffsetAttr().Set(particle_contact_offset)
    particle_system.CreateMaxVelocityAttr().Set(100.0)

    # --- PBD material ---
    particleUtils.add_pbd_particle_material(
        stage,
        material_path,
        cohesion=0.01,
        viscosity=0.005,
        surface_tension=0.074,
        friction=0.1,
    )
    physicsUtils.add_physics_material_to_prim(
        stage, particle_system.GetPrim(), material_path
    )

    # --- Particle grid ---
    fluid_rest_offset = 0.99 * 0.6 * particle_contact_offset
    particle_spacing = 2.0 * fluid_rest_offset
    dim = 10

    positions, velocities = particleUtils.create_particles_grid(
        lower=Gf.Vec3f(-dim * particle_spacing * 0.5),
        particle_spacing=particle_spacing,
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
    )

    # --- PointInstancer (default prototype) ---
    particleUtils.add_physx_particleset_pointinstancer(
        stage,
        particle_set_path,
        Vt.Vec3fArray(positions),
        Vt.Vec3fArray(velocities),
        particle_system_path,
        self_collision=True,
        fluid=True,
        particle_group=0,
        particle_mass=0.001,
        density=0.0,
    )

    # Position 1m above ground
    point_instancer = UsdGeom.PointInstancer.Get(stage, particle_set_path)
    physicsUtils.set_or_add_translate_op(point_instancer, translate=Gf.Vec3f(0.0, 0.0, 1.0))

    # Customize prototype sphere
    prototype = UsdGeom.Sphere.Get(
        stage, particle_set_path.AppendChild("particlePrototype0")
    )
    prototype.CreateRadiusAttr().Set(fluid_rest_offset)
    prototype.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.2, 0.5, 0.9)]))

    print(f"[INFO]: Created {len(positions)} particles")
    print(f"  particle_contact_offset = {particle_contact_offset}")
    print(f"  fluid_rest_offset = {fluid_rest_offset:.4f}")
    print(f"  particle_spacing = {particle_spacing:.4f}")
    print(f"  grid = {dim}x{dim}x{dim}")

    return stage


def main():
    """Main function — uses timeline + app.update() like the demos."""
    stage = create_scene()

    # Set viewport camera
    import omni.kit.viewport.utility as vp_utils
    vp = vp_utils.get_active_viewport()
    if vp:
        from omni.kit.viewport.utility import frame_viewport_selection
        vp.set_camera_position("/OmniverseKit_Persp", 3.0, 3.0, 3.0)
        vp.set_camera_target("/OmniverseKit_Persp", 0.0, 0.0, 0.5)

    # Start simulation using timeline (like the demos do, NOT SimulationContext)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[INFO]: Timeline playing — particles should be dropping")

    point_instancer = UsdGeom.PointInstancer.Get(stage, Sdf.Path("/World/Particles"))

    count = 0
    while simulation_app.is_running():
        simulation_app.update()
        count += 1
        if count % 60 == 0:
            positions = point_instancer.GetPositionsAttr().Get()
            if positions and len(positions) > 0:
                p = positions[0]
                print(f"[Step {count}] Particle 0 pos: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            else:
                print(f"[Step {count}] No particle positions found!")


if __name__ == "__main__":
    main()
    simulation_app.close()
