# IsaacLab Physics Cookbook

Practical reference and working examples for physics simulation in [IsaacLab](https://github.com/isaac-sim/IsaacLab) (v2.3.2) with NVIDIA Isaac Sim 5.1.

Covers rigid bodies, deformable bodies, and documents the current state of particle/fluid support.

## Setup

- **Isaac Sim 5.1** (pip install)
- **IsaacLab v2.3.2** (from source)
- Python 3.11, CUDA-capable GPU

## Running Examples

```bash
# From your IsaacLab root directory:
isaaclab -p examples/restitution.py
isaaclab -p examples/deformable_interaction.py
isaaclab -p examples/mass_vs_deformable.py
isaaclab -p examples/deformable_sheets.py

# On Windows:
isaaclab.bat -p examples/restitution.py
```

## Examples

| Script | What it shows |
|--------|--------------|
| `restitution.py` | 4 spheres with different bounce values (0.0, 0.3, 0.6, 0.95) |
| `deformable_interaction.py` | Stiff deformable block dropped onto soft deformable block |
| `mass_vs_deformable.py` | Heavy (50kg) vs light (1kg) rigid blocks on identical soft cubes |
| `deformable_sheets.py` | Thin flexible sheets dropping onto platforms |
| `fluid_particles.py` | Particle fluid attempt (see Known Issues) |

## Quick Reference

### Rigid Bodies

Use shape spawners — no mesh needed:

```python
from isaaclab.assets import RigidObject, RigidObjectCfg

cfg = RigidObjectCfg(
    prim_path="/World/Origin/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.1,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8,  # 0=no bounce, 1=perfect bounce
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)
ball = RigidObject(cfg=cfg)
```

Available shape spawners: `SphereCfg`, `CuboidCfg`, `ConeCfg`, `CylinderCfg`

**Sim loop (rigid):**
```python
obj.write_root_pose_to_sim(root_state[:, :7])
obj.write_root_velocity_to_sim(root_state[:, 7:])
obj.reset()
obj.write_data_to_sim()
sim.step()
obj.update(sim_dt)
```

### Deformable Bodies

**Must** use `Mesh` spawners — shape spawners don't support deformable:

```python
from isaaclab.assets import DeformableObject, DeformableObjectCfg

cfg = DeformableObjectCfg(
    prim_path="/World/Origin/SoftBody",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.5, 0.5, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=5000.0,    # low=soft/squishy, high=stiff
            poissons_ratio=0.4,       # volume preservation (0-0.5)
            damping_scale=0.1,        # energy dissipation
        ),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
soft = DeformableObject(cfg=cfg)
```

Available mesh spawners: `MeshCuboidCfg`, `MeshSphereCfg`, `MeshCylinderCfg`

**Sim loop (deformable):**
```python
nodal_state = obj.data.default_nodal_state_w.clone()
obj.write_nodal_state_to_sim(nodal_state)
obj.write_nodal_kinematic_target_to_sim(...)
obj.reset()
obj.write_data_to_sim()
sim.step()
obj.update(sim_dt)
```

### Multiple Origins Pattern

Replicate scenes side-by-side using regex prim paths:

```python
origins = [[i * 1.5, 0.0, 0.0] for i in range(N)]
for i, origin in enumerate(origins):
    sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

# Then use regex in prim_path:
cfg = RigidObjectCfg(prim_path="/World/Origin.*/Object", ...)
```

### SimulationCfg Defaults

| Parameter | Default |
|-----------|---------|
| `physics_prim_path` | `/physicsScene` |
| `device` | `cuda:0` |
| `use_fabric` | `True` |
| `dt` | `1/60` |
| `solver_type` | `1` (TGS) |

## Known Issues

### Fluid/Particle Simulation Does NOT Work in IsaacLab Standalone Scripts

Particles are visible but **never move**. We tried everything:

1. `particleUtils.add_physx_particle_system()` with all manual offsets
2. Direct `PhysxSchema.PhysxParticleSystem.Define()`
3. `use_fabric=False` on SimulationCfg
4. Manual GPU dynamics enable before/after reset
5. Bypassing SimulationContext entirely (`timeline.play()` + `app.update()`)

**Why:** IsaacLab's `SimulationContext` does something that prevents PhysX particle simulation from running. All of Isaac Sim's own working particle demos explicitly disable Fabric and run outside SimulationContext. Even matching their exact pattern in a standalone script doesn't work.

**Workaround:** Particles only work through Isaac Sim's built-in demo browser (Window > Physics > Demo Scenes).

### Key Particle APIs (for future reference)

```python
PhysxSchema.PhysxParticleSystem.Define()        # create particle system
particleUtils.add_pbd_particle_material()        # PBD material
particleUtils.add_physx_particleset_pointinstancer()  # visible particles
particleUtils.create_particles_grid()            # generate grid positions
physicsUtils.add_physics_material_to_prim()      # bind material

# Offset formula:
fluid_rest_offset = 0.99 * 0.6 * particle_contact_offset
```

## License

BSD-3-Clause (same as IsaacLab)
