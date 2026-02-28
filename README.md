# IsaacLab Physics — Claude Code Skill

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) custom slash command that generates complete, runnable physics simulation scripts for [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Describe a physics scenario in plain English, and `/isaac-sim-physics` produces a working IsaacLab script with the right spawners, materials, sim loop, and reset logic.

## Install

Clone this repo (or copy `.claude/commands/isaac-sim-physics.md` into your project):

```bash
git clone https://github.com/stevenzhang070302/isaaclab-physics-cookbook.git
cd isaaclab-physics-cookbook
```

The skill is automatically available when you run Claude Code from this directory:

```
claude
> /isaac-sim-physics bouncing balls with different restitution values
```

Or copy it into any project:

```bash
# Project-level (available in that project)
mkdir -p .claude/commands
cp isaaclab-physics-cookbook/.claude/commands/isaac-sim-physics.md .claude/commands/

# User-level (available everywhere)
cp isaaclab-physics-cookbook/.claude/commands/isaac-sim-physics.md ~/.claude/commands/
```

## Usage

```
/isaac-sim-physics <describe your scene>
```

### Examples

```
/isaac-sim-physics 4 spheres with different bounciness dropped from the same height
/isaac-sim-physics heavy block vs light block dropped onto soft deformable cubes
/isaac-sim-physics stiff deformable object landing on a very soft deformable surface
/isaac-sim-physics thin flexible sheets falling onto cylindrical platforms
```

## What It Knows

| Physics Type | Status | Spawners |
|---|---|---|
| Rigid bodies | Working | `SphereCfg`, `CuboidCfg`, `CylinderCfg`, `ConeCfg` |
| Deformable bodies | Working | `MeshSphereCfg`, `MeshCuboidCfg`, `MeshCylinderCfg` |
| Particles / fluids | Broken in IsaacLab | See [Known Issues](#known-issues) |

### Rigid body properties
- Mass (`MassPropertiesCfg`)
- Restitution / bounciness (`RigidBodyMaterialCfg`)
- Static & dynamic friction (`RigidBodyMaterialCfg`)
- Collisions (`CollisionPropertiesCfg`)

### Deformable body properties
- Stiffness via Young's modulus (`DeformableBodyMaterialCfg`)
- Volume preservation via Poisson's ratio
- Energy dissipation via damping

## Example Scripts

The `examples/` directory contains working scripts generated with this skill:

| Script | Description |
|--------|-------------|
| `restitution.py` | 4 spheres with bounce values 0.0, 0.3, 0.6, 0.95 |
| `deformable_interaction.py` | Stiff deformable dropped onto soft deformable |
| `mass_vs_deformable.py` | 50kg vs 1kg rigid on identical soft cubes |
| `deformable_sheets.py` | Thin flexible sheets onto platforms |
| `fluid_particles.py` | Particle fluid attempt (documented non-working) |

Run them with:

```bash
isaaclab -p examples/restitution.py
# Windows:
isaaclab.bat -p examples/restitution.py
```

## Requirements

- NVIDIA Isaac Sim 5.1 (pip)
- IsaacLab v2.3.2
- Python 3.11
- CUDA-capable GPU

## Known Issues

**Particles/fluids do not work in IsaacLab standalone scripts.** Particles appear but never simulate. IsaacLab's `SimulationContext` prevents PhysX particle simulation from running. This has been tested extensively — see `examples/fluid_particles.py` and the skill prompt for details. Particles only work through Isaac Sim's built-in demo browser.

## License

BSD-3-Clause
