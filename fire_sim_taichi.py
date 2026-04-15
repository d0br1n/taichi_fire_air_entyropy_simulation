import sys
import math
import time
from typing import Tuple
import taichi as ti

def parse_taichi_version(ver) -> Tuple[int, int, int]:
    """
    Parse taichi.__version__ into a (major, minor, patch) tuple of ints.
    Accepts string like "1.6.0", tuple/list like (1, 6, 0), or other similar forms.
    Non-numeric suffixes are ignored.
    """
    nums = []

    # Normalize to list of string parts
    if isinstance(ver, (tuple, list)):
        parts = [str(x) for x in ver]
    elif isinstance(ver, str):
        parts = ver.split(".")
    else:
        # Fallback: unknown type
        parts = [str(ver)]

    for p in parts[:3]:
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        nums.append(int(n) if n else 0)

    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]


# Prefer modern UI (ti.ui). Avoid removed ti.GUI in newer versions.
HAS_UI = hasattr(ti, "ui") and hasattr(ti.ui, "Scene") and hasattr(ti.ui, "Camera")


def safe_ti_init():
    # gpu only
    try:
        ti.init(arch=ti.gpu)
    except RuntimeError:
        print("GPU not available, falling back to CPU.")
        ti.init(arch=ti.cpu)

# Simulation parameters (tweakable)
MAX_FIRE = 60_000
MAX_AIR = 10_000
EMIT_RATE = 6_000  # particles per second (approx)
PARTICLE_LIFETIME = 1.7  # seconds
EMITTER_RADIUS = 0.20
EMITTER_HEIGHT = 0.0
BASE_UPWARD_SPEED = 1.5  # m/s equivalent units
BASE_CENTER_SPEED = 0.1
NOISE_STRENGTH = 1.2
DRAG = 0.6
GRAVITY = 0.0  # fire buoyancy dominates; keep gravity 0 for simplicity
PARTICLE_RADIUS = 0.010
GRID_CELL_SIZE = 0.1
GRID_SIZE_X = 48
GRID_SIZE_Y = 48
GRID_SIZE_Z = 48
GRID_ORIGIN = ti.Vector([-2.4, 0.0, -2.4])
PUSH_STRENGTH = 2.5
PUSH_MAX = 10.0
PUSH_FADE = 0.018
CENTER_ATTRACT_STRENGTH = 0.6
CENTER_ATTRACT_MAX = 3.0

# Entropy visualization parameters
ENTROPY_GRID_DIV = 5  # Divide each dimension into 5 parts (5x5x5 = 125 cubes)
BOLTZMANN_K = 1.0  # Use dimensionless scaling so entropy magnitude is comparable for coloring

# Initialize Taichi BEFORE creating any fields
_ARCH = safe_ti_init()

# Fields
pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FIRE)
vel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FIRE)
life = ti.field(dtype=ti.f32, shape=MAX_FIRE)
color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FIRE)

pos1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_AIR)
vel1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_AIR)
color1 = ti.Vector.field(3, dtype=ti.f32, shape=MAX_AIR)

alive_fire_count = ti.field(dtype=ti.i32, shape=())
alive_air_count = ti.field(dtype=ti.i32, shape=())
accum_emit = ti.field(dtype=ti.f32, shape=())

grid_density = ti.field(dtype=ti.f32, shape=(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z))

# Entropy grid fields
entropy_grid_count = ti.field(dtype=ti.i32, shape=(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV))
entropy_grid_entropy = ti.field(dtype=ti.f32, shape=(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV))
entropy_grid_color = ti.Vector.field(3, dtype=ti.f32, shape=(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV))  # Changed to vec3 (RGB only)

# Entropy cube visualization (8 vertices per cube, 12 edges per cube, 2 points per edge)
ENTROPY_CUBES_COUNT = ENTROPY_GRID_DIV ** 3
entropy_cube_edges = ti.Vector.field(3, dtype=ti.f32, shape=(ENTROPY_CUBES_COUNT * 24))  # 12 edges * 2 points

# For mesh: positions and per-vertex colors (RGB only - no alpha for per_vertex_color)
cube_vertices = ti.Vector.field(3, ti.f32, ENTROPY_CUBES_COUNT * 8)      # 8 verts per cube
cube_colors   = ti.Vector.field(4, ti.f32, ENTROPY_CUBES_COUNT * 8)      # RGB vec3 (changed from vec4)
cube_indices  = ti.field(ti.i32, ENTROPY_CUBES_COUNT * 36)               # 36 indices (12 triangles × 3)

@ti.func
def random_float():
    return ti.random(ti.f32)


@ti.func
def random_sign() -> ti.i32:
    out = 1
    if random_float() < 0.5:
        out = -1
    return out


@ti.func
def world_to_cell(p: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.i32):
    g = (p - GRID_ORIGIN) / GRID_CELL_SIZE
    return ti.Vector([ti.i32(ti.floor(g.x)), ti.i32(ti.floor(g.y)), ti.i32(ti.floor(g.z))])


@ti.func
def world_to_entropy_cell(p: ti.types.vector(3, ti.f32), cube_half: ti.f32) -> ti.types.vector(3, ti.i32):
    """Convert world position to entropy grid cell index.
    X/Z axes span [-cube_half, cube_half]; Y spans [0, cube_half].
    Use separate cell sizes so all Y layers are reachable.
    """
    # Cell sizes per axis
    cell_size_xz = (cube_half * 2.0) / ENTROPY_GRID_DIV
    cell_size_y = cube_half / ENTROPY_GRID_DIV

    # Offset X/Z into [0, 2*cube_half]; Y already in [0, cube_half]
    ox = p.x + cube_half
    oy = p.y
    oz = p.z + cube_half

    # Calculate cell indices
    ix = ti.i32(ti.floor(ox / cell_size_xz))
    iy = ti.i32(ti.floor(oy / cell_size_y))
    iz = ti.i32(ti.floor(oz / cell_size_xz))

    # Clamp to valid range
    ix = ti.max(0, ti.min(ENTROPY_GRID_DIV - 1, ix))
    iy = ti.max(0, ti.min(ENTROPY_GRID_DIV - 1, iy))
    iz = ti.max(0, ti.min(ENTROPY_GRID_DIV - 1, iz))

    return ti.Vector([ix, iy, iz])


@ti.kernel
def clear_grid():
    for i, j, k in grid_density:
        grid_density[i, j, k] = 0.0


@ti.kernel
def clear_entropy_grid():
    """Clear entropy grid counts"""
    for i, j, k in entropy_grid_count:
        entropy_grid_count[i, j, k] = 0


@ti.kernel
def accumulate_to_grid():
    for idx in range(MAX_FIRE):
        if life[idx] > 0:
            p = pos[idx]
            ijk = world_to_cell(p)
            if 0 <= ijk.x < GRID_SIZE_X and 0 <= ijk.y < GRID_SIZE_Y and 0 <= ijk.z < GRID_SIZE_Z:
                ti.atomic_add(grid_density[ijk.x, ijk.y, ijk.z], 1.0)


@ti.kernel
def accumulate_air_to_entropy_grid(cube_half: ti.f32):
    """Count air particles in each entropy grid cell"""
    for idx in range(alive_air_count[None]):
        p = pos1[idx]
        # Only count particles within the main cube
        if (-cube_half <= p.x <= cube_half and
                0.0 <= p.y <= cube_half and
                -cube_half <= p.z <= cube_half):
            ijk = world_to_entropy_cell(p, cube_half)
            ti.atomic_add(entropy_grid_count[ijk.x, ijk.y, ijk.z], 1)

max_entropy_val = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def calculate_entropy():
    max_entropy_val[None] = 0.0  # Reset to capture true max each frame

    for i, j, k in entropy_grid_count:
        n = ti.f32(entropy_grid_count[i, j, k])
        s_val = 0.0

        if n > 1.0:
            # Boltzmann Formula: S = k_B * ln(n!)
            # Using Stirling's Approximation: ln(n!) ≈ n*ln(n) - n
            s_val = BOLTZMANN_K * (n * ti.log(n) - n)
        entropy_grid_entropy[i, j, k] = s_val
        ti.atomic_max(max_entropy_val[None], s_val)


@ti.kernel
def update_entropy_cube_edges(cube_half: ti.f32):
    """Update the edge positions of entropy cubes for rendering"""
    cube_size_xz = cube_half * 2.0  # X and Z go from -cube_half to +cube_half
    cube_size_y = cube_half          # Y goes from 0 to cube_half
    cell_size_xz = cube_size_xz / ENTROPY_GRID_DIV
    cell_size_y = cube_size_y / ENTROPY_GRID_DIV

    for i, j, k in ti.ndrange(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV):
        cube_idx = i * (ENTROPY_GRID_DIV * ENTROPY_GRID_DIV) + j * ENTROPY_GRID_DIV + k

        # Calculate cube corners in world space
        x0 = -cube_half + i * cell_size_xz
        y0 = 0.0 + j * cell_size_y
        z0 = -cube_half + k * cell_size_xz

        x1 = x0 + cell_size_xz
        y1 = y0 + cell_size_y
        z1 = z0 + cell_size_xz

        # Define 8 corners
        v000 = ti.Vector([x0, y0, z0])
        v001 = ti.Vector([x0, y0, z1])
        v010 = ti.Vector([x0, y1, z0])
        v011 = ti.Vector([x0, y1, z1])
        v100 = ti.Vector([x1, y0, z0])
        v101 = ti.Vector([x1, y0, z1])
        v110 = ti.Vector([x1, y1, z0])
        v111 = ti.Vector([x1, y1, z1])

        # Define 12 edges (24 points)
        base = cube_idx * 24

        # Bottom face
        entropy_cube_edges[base + 0] = v000
        entropy_cube_edges[base + 1] = v100
        entropy_cube_edges[base + 2] = v100
        entropy_cube_edges[base + 3] = v101
        entropy_cube_edges[base + 4] = v101
        entropy_cube_edges[base + 5] = v001
        entropy_cube_edges[base + 6] = v001
        entropy_cube_edges[base + 7] = v000

        # Top face
        entropy_cube_edges[base + 8] = v010
        entropy_cube_edges[base + 9] = v110
        entropy_cube_edges[base + 10] = v110
        entropy_cube_edges[base + 11] = v111
        entropy_cube_edges[base + 12] = v111
        entropy_cube_edges[base + 13] = v011
        entropy_cube_edges[base + 14] = v011
        entropy_cube_edges[base + 15] = v010

        # Vertical edges
        entropy_cube_edges[base + 16] = v000
        entropy_cube_edges[base + 17] = v010
        entropy_cube_edges[base + 18] = v100
        entropy_cube_edges[base + 19] = v110
        entropy_cube_edges[base + 20] = v101
        entropy_cube_edges[base + 21] = v111
        entropy_cube_edges[base + 22] = v001
        entropy_cube_edges[base + 23] = v011

@ti.func
def sample_density(i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:
    val = ti.f32(0.0)
    if 0 <= i < GRID_SIZE_X and 0 <= j < GRID_SIZE_Y and 0 <= k < GRID_SIZE_Z:
        val = grid_density[i, j, k]
    return val

@ti.func
def density_gradient_world (p: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    #estimate gradient
    gpos = (p - GRID_ORIGIN) / GRID_CELL_SIZE
    i = ti.i32(ti.floor(gpos.x))
    j = ti.i32(ti.floor(gpos.y))
    k = ti.i32(ti.floor(gpos.z))
    #neighbor samples
    dx = (sample_density(i + 1, j, k) - sample_density(i - 1, j, k)) / 2.0
    dy = (sample_density(i, j + 1, k) - sample_density(i, j - 1, k)) / 2.0
    dz = (sample_density(i, j, k + 1) - sample_density(i, j, k - 1)) / 2.0

    grad_grid = ti.Vector([dx, dy, dz])
    grad_world = grad_grid / GRID_CELL_SIZE

    mag = grad_world.norm() + 1e-6
    if mag > PUSH_MAX:
        grad_world = grad_world  * (PUSH_MAX / mag)
    return grad_world


# Random helper (ti.random is available in current Taichi)

@ti.func
def sample_disc(radius: ti.f32) -> ti.types.vector(2, ti.f32):
    # Uniform disc sampling via concentric mapping approximation
    r = ti.sqrt(random_float()) * radius
    theta = 2.0 * math.pi * random_float()
    return ti.Vector([r * ti.cos(theta), r * ti.sin(theta)])


@ti.kernel
def reset_all():
    # RESET FIRE PARTICLES (MAX_FIRE)
    for i in range(MAX_FIRE):
        life[i] = 0.0
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        pos[i] = ti.Vector([0.0, -10.0, 0.0])
        color[i] = ti.Vector([0.0, 0.0, 0.0])

    # RESET AIR PARTICLES (MAX_AIR)
    for i in range(MAX_AIR):
        vel1[i] = ti.Vector([0.0, 0.0, 0.0])
        pos1[i] = ti.Vector([0.0, -10.0, 0.0])
        color1[i] = ti.Vector([0.0, 0.0, 0.0])

    # RESET COUNTS ȘI ACCUMULATORI
    alive_fire_count[None] = 0
    alive_air_count[None] = 0
    accum_emit[None] = 0.0


@ti.kernel
def spawn_fire(n_to_spawn: int, t: ti.f32, energy: ti.f32):
    # Spawn up to n_to_spawn particles by reusing dead slots
    spawned = 0
    for i in range(MAX_FIRE):
        if spawned >= n_to_spawn:
            continue
        if life[i] <= 0.0:
            # Sample emitter disc at base
            d = sample_disc(EMITTER_RADIUS)
            pos[i] = ti.Vector([d.x, EMITTER_HEIGHT, d.y])
            # Upward initial velocity + a bit of radial spread
            dir = ti.Vector([d.x, 0.0, d.y])
            spread = 1.5 * (dir.norm() + 0.05)
            v = ti.Vector([
                (random_float() - 0.5) * spread,
                BASE_UPWARD_SPEED * (0.85 + 0.3 * random_float()) * energy,
                (random_float() - 0.5) * spread,
            ])
            vel[i] = v
            # Fresh life
            life[i] = ti.max(1.0, PARTICLE_LIFETIME * (0.4 + 1.5 * random_float()))
            # Initial hot color (near white/yellow)
            color[i] = ti.Vector([1.0, 0.92, 0.15])
            spawned += 1
    alive_fire_count[None] += spawned


@ti.kernel
def spawn_air(to_spawn: ti.i32):
    # Legacy: spawn from index 0; kept for compatibility but not used in run loop
    spawned = 0
    for i in range(MAX_AIR):
        if spawned >= to_spawn:
            continue
        pos1[i] = ti.Vector([
            random_float() * 2.5 * random_sign(),
            random_float() * 2.5,
            random_float() * 2.5 * random_sign()
        ])
        vel1[i] = ti.Vector([
            random_float() * 0.3 * random_sign(),
            random_float() * 0.3 * random_sign(),
            random_float() * 0.3 * random_sign()
        ])
        color1[i] = ti.Vector([1.0, 1.0, 1.0])
        spawned += 1
    alive_air_count[None] += spawned

@ti.kernel
def spawn_air_from(start: ti.i32, count: ti.i32):
    for c in range(count):
        idx = start + c
        if idx < MAX_AIR:
            pos1[idx] = ti.Vector([
                random_float() * 2.5 * random_sign(),
                random_float() * 2.5,
                random_float() * 2.5 * random_sign()
            ])
            vel1[idx] = ti.Vector([
                random_float() * 0.3 * random_sign(),
                random_float() * 0.3 * random_sign(),
                random_float() * 0.3 * random_sign()
            ])
            color1[idx] = ti.Vector([1.0, 1.0, 1.0])

@ti.kernel
def deactivate_air_range(start: ti.i32, end_exclusive: ti.i32):
    for idx in range(start, end_exclusive):
        if 0 <= idx < MAX_AIR:
            vel1[idx] = ti.Vector([0.0, 0.0, 0.0])
            pos1[idx] = ti.Vector([0.0, -10.0, 0.0])
            color1[idx] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_fire(dt: ti.f32, t: ti.f32, energy: ti.f32):
    # Simple particle dynamics with upward drift, noise, and fade
    for i in range(MAX_FIRE):
        if life[i] > 0.0:
            # Procedural jitter/noise to create turbulent look
            n = ti.Vector([
                (random_float() - 0.5),
                (random_float() - 0.5),
                (random_float() - 0.5),
            ]) * NOISE_STRENGTH

            # Upward buoyancy is baked into BASE_UPWARD_SPEED; add mild vortex effect
            v = vel[i]
            v += ti.Vector([0.0, GRAVITY, 0.0]) * dt
            v += n * dt
            v *= ti.exp(-DRAG * dt)

            to_center = ti.Vector([-pos[i].x, 0.0, -pos[i].z])
            dist = ti.sqrt(to_center.x * to_center.x + to_center.z * to_center.z) + 1e-6
            dir_center = to_center / dist
            acc_center = dir_center * CENTER_ATTRACT_STRENGTH
            acc_max = ti.min(CENTER_ATTRACT_MAX, acc_center.norm() + 1e-6)
            acc_center = dir_center * acc_max
            v += acc_center * dt


            grad = density_gradient_world(pos[i])
            push_cancel = -PUSH_STRENGTH * grad * PUSH_FADE * ((energy * 4.5) ** -1)
            # Never allow downward vertical push
            if push_cancel.y < 0:
                push_cancel = ti.Vector([push_cancel.x, 0.0, push_cancel.z])
            v += push_cancel * dt

            # Final safety clamp: never go downward
            if v.y < 0.0:
                v.y = 0.0

            p = pos[i]
            p += v * dt

            # Lifetime and color ramp: hot (white/yellow) -> orange -> red -> dark
            lf = life[i] - max(dt, dt * (energy * 3.0))
            life[i] = lf

            # Height can modulate cooling a bit
            h = ti.max(0.0, p.y)
            cool = ti.min(1.2, 0.7 * ((PARTICLE_LIFETIME - ti.max(0.0, lf)) / PARTICLE_LIFETIME + 0.2 * h) - energy/2)

            # Color ramp via simple piecewise smixes
            vhot = ti.Vector([0.1, 0.3, 1.0])
            white = ti.Vector([1.0, 1.0, 1.0])
            hot = ti.Vector([1.0, 0.82, 0.10])
            orange = ti.Vector([1.0, 0.6, 0.1])
            red = ti.Vector([0.9, 0.1, 0.05])
            dark_red = ti.Vector([0.40, 0.2, 0.2])
            c = ti.Vector([0.0, 0.0, 0.0])
            # cool este in [0.0, 1.0]

            c_low_hot = hot  # galben-portocaliu
            c_low_mid = orange  # portocaliu
            c_low_cool = dark_red  # rosu inchis (cool ~ 1.0)

            c_high_hot = vhot  # Albastru
            c_high_mid = white  # Alb
            c_high_cool = hot * 0.5 + red * 0.5  # galben deschis/desaturat

            c_low = ti.Vector([0.0, 0.0, 0.0])
            if cool < 0.3:
                k_mid = cool / 0.3
                c_low = c_low_hot * (1.0 - k_mid) + c_low_mid * k_mid
            else:
                k_mid = (cool - 0.3) / 0.3
                c_low = c_low_mid * (1.0 - k_mid) + c_low_cool * k_mid

            c_high = ti.Vector([0.0, 0.0, 0.0])
            if cool < 0.3:
                k_mid = cool / 0.3
                c_high = c_high_hot * (1.0 - k_mid) + c_high_mid * k_mid
            else:
                k_mid = (cool - 0.3) / 0.3
                c_high = c_high_mid * (1.0 - k_mid) + c_high_cool * k_mid

            # c = LERP(c_low, c_high, energy)
            c = c_low * (1.0 - energy) + c_high * energy

            min_col = ti.Vector([0.18, 0.18, 0.18])
            color[i] = ti.max(min_col, c)
            pos[i] = p
            vel[i] = v
        else:
            # Make dead particles invisible
            pos[i] = ti.Vector([0.0, -1e6, 0.0])
            color[i] = ti.Vector([0.0, 0.0, 0.0])

    # Update alive count and mark dead
    dead_local = 0
    for i in range(MAX_FIRE):
        if life[i] <= 0.0:
            dead_local += 1
    alive_fire_count[None] = MAX_FIRE - dead_local


@ti.kernel
def step_air(dt: ti.f32, energy: ti.f32):
    for i in range(MAX_AIR):
        # Only update active air particles
        if i >= alive_air_count[None]:
            continue

        v = vel1[i]
        p = pos1[i]

        grad = density_gradient_world(p)
        push_force = PUSH_STRENGTH * grad * energy  #push from grid gradient
        if push_force.y < 0:
            push_force = ti.Vector([push_force.x, 0.0, push_force.z])
        v += push_force * dt

        p += v * dt

        for j in range(i + 1, alive_air_count[None]):
            p1 = pos1[j]
            r = p - p1
            dist = r.norm()

            if dist < PARTICLE_RADIUS * 2:
                # swap real velocity vectors
                vel1[i], vel1[j] = -vel1[j], -vel1[i]

        # collision handling
        if p.x > cube_half:
            p.x = cube_half
            v.x *= -1
        if p.y > cube_half:
            p.y = cube_half
            v.y *= -1
        if p.z > cube_half:
            p.z = cube_half
            v.z *= -1
        if p.x < -cube_half:
            p.x = -cube_half
            v.x *= -1
        if p.y < 0:
            p.y = 0
            v.y *= -1
        if p.z < -cube_half:
            p.z = -cube_half
            v.z *= -1

        pos1[i] = p
        vel1[i] = v

N = 24
cube_half = 2.5
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

@ti.kernel
def build_entropy_cubes_mesh(cube_half: ti.f32):
    cube_size_xz = cube_half * 2.0  # X and Z go from -cube_half to +cube_half
    cube_size_y = cube_half          # Y goes from 0 to cube_half
    cell_size_xz = cube_size_xz / ENTROPY_GRID_DIV
    cell_size_y = cube_size_y / ENTROPY_GRID_DIV

    for i, j, k in ti.ndrange(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV):
        cube_idx = i * (ENTROPY_GRID_DIV**2) + j * ENTROPY_GRID_DIV + k
        base_v = cube_idx * 8
        base_i = cube_idx * 36

        # Calculate cube corners in world space (matching update_entropy_cube_edges)
        x0 = -cube_half + i * cell_size_xz
        x1 = x0 + cell_size_xz
        y0 = 0.0 + j * cell_size_y
        y1 = y0 + cell_size_y
        z0 = -cube_half + k * cell_size_xz
        z1 = z0 + cell_size_xz

        # 8 vertices
        v = [
            ti.Vector([x0,y0,z0]), ti.Vector([x1,y0,z0]), ti.Vector([x1,y0,z1]), ti.Vector([x0,y0,z1]),
            ti.Vector([x0,y1,z0]), ti.Vector([x1,y1,z0]), ti.Vector([x1,y1,z1]), ti.Vector([x0,y1,z1]),
        ]

        for lv in ti.static(range(8)):
            cube_vertices[base_v + lv] = v[lv]  # Positions only (colors set separately in update_cube_colors)

        # Indices for 12 triangles
        faces = [
            [0,1,2], [0,2,3],     # bottom
            [4,5,6], [4,6,7],     # top
            [0,1,5], [0,5,4],     # front
            [2,3,7], [2,7,6],     # back
            [0,3,7], [0,7,4],     # left
            [1,2,6], [1,6,5],     # right
        ]
        idx = 0
        for f in ti.static(faces):
            for vv in ti.static(range(3)):
                cube_indices[base_i + idx] = base_v + f[vv]
                idx += 1


@ti.kernel
def update_mesh_colors():
    # Use the max value from the previous calculation
    m_s = max_entropy_val[None]

    for i, j, k in ti.ndrange(ENTROPY_GRID_DIV, ENTROPY_GRID_DIV, ENTROPY_GRID_DIV):
        count = entropy_grid_count[i, j, k]
        s_val = entropy_grid_entropy[i, j, k]

        # Normalize 0.0 to 1.0 (robust against tiny m_s)
        normalized = 0.0
        if m_s > 1e-8:
            normalized = s_val / m_s
            normalized = ti.min(ti.max(normalized, 0.0), 1.0)
            # Slight gamma to emphasize low values
            normalized = ti.pow(normalized, 0.35)

        # Map low entropy (blue) -> high entropy (red)
        r = normalized
        g = 0.0
        b = 1.0 - normalized
        alpha = 0.28  # ensure all meshes are visible

        final_col = ti.Vector([r, g, b, alpha])

        # Apply to all 8 vertices of this specific cube
        cube_idx = i * (ENTROPY_GRID_DIV ** 2) + j * ENTROPY_GRID_DIV + k
        base_v = cube_idx * 8
        for v_idx in range(8):
            cube_colors[base_v + v_idx] = final_col


@ti.kernel
def init_points_pos(points : ti.template()):
    points[0] = ti.Vector([cube_half, 0.0, cube_half])
    points[1] = ti.Vector([-cube_half, 0.0, cube_half])
    points[2] = ti.Vector([-cube_half, 0.0, cube_half])
    points[3] = ti.Vector([-cube_half, 0.0, -cube_half])
    points[4] = ti.Vector([-cube_half, 0.0, -cube_half])
    points[5] = ti.Vector([cube_half, 0.0, -cube_half])
    points[6] = ti.Vector([cube_half, 0.0, -cube_half])
    points[7] = ti.Vector([cube_half, 0.0, cube_half])
    points[8] = ti.Vector([cube_half, 0.0, cube_half])
    points[9] = ti.Vector([cube_half, cube_half, cube_half])
    points[10] = ti.Vector([-cube_half, 0, cube_half])
    points[11] = ti.Vector([-cube_half, cube_half, cube_half])
    points[12] = ti.Vector([-cube_half, 0, -cube_half])
    points[13] = ti.Vector([-cube_half, cube_half, -cube_half])
    points[14] = ti.Vector([cube_half, 0, -cube_half])
    points[15] = ti.Vector([cube_half, cube_half, -cube_half])
    points[16] = ti.Vector([cube_half,cube_half, cube_half])
    points[17] = ti.Vector([-cube_half,cube_half, cube_half])
    points[18] = ti.Vector([-cube_half, cube_half, cube_half])
    points[19] = ti.Vector([-cube_half,cube_half, -cube_half])
    points[20] = ti.Vector([-cube_half,cube_half, -cube_half])
    points[21] = ti.Vector([cube_half,cube_half, -cube_half])
    points[22] = ti.Vector([cube_half,cube_half, -cube_half])
    points[23] = ti.Vector([cube_half,cube_half, cube_half])

init_points_pos(points_pos)


def run():
    ver_raw = getattr(ti, "__version__", "0.0.0")
    ver_tuple = parse_taichi_version(ver_raw)
    ver_str = ".".join(map(str, ver_raw)) if isinstance(ver_raw, (tuple, list)) else str(ver_raw)
    print(f"Taichi version: {ver_str} (parsed: {ver_tuple}), backend: {_ARCH}")

    if not HAS_UI:
        # Avoid using removed/deprecated GUI APIs on newer Taichi; ask user to upgrade if too old.
        print(
            "This demo requires taichi.ui (Scene/Camera), which your Taichi build lacks.\n"
            "Please install a recent Taichi (e.g., pip install -U taichi).",
            file=sys.stderr,
        )
        return None

    reset_all()

    # One-time mesh setup (positions and indices are static)
    build_entropy_cubes_mesh(cube_half)

    # Setup window and 3D scene
    window = ti.ui.Window("Simple 3D Fire (Taichi)", res=(900, 600), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()

    # Camera defaults
    cam_dist = 2.4
    cam_angle = 0.0
    cam_height = 1.0  # vertical camera position

    energy = 0.5
    emit_rate_variable = EMIT_RATE
    max_active_fire_particles = MAX_FIRE
    max_active_air_particles = MAX_AIR

    # Air spawning is managed every frame to match the slider target

    last_time = time.perf_counter()

    # Main loop
    while window.running:
        # input

        if window.is_pressed("r"):
            reset_all()
        if window.is_pressed("a"):
            cam_angle -= 0.05
        if window.is_pressed("d"):
            cam_angle += 0.05
        if window.is_pressed("w"):
            cam_dist -= 0.05
        if window.is_pressed("s"):
            cam_dist += 0.05
        if window.is_pressed("q"):
            cam_height = min(5.0, cam_height + 0.05)
        if window.is_pressed("e"):
            cam_height = max(-1.0, cam_height - 0.05)

        now = time.perf_counter()
        dt = max(1e-3, min(1/30.0, now - last_time))  # clamp dt for stability
        last_time = now

        # Emit particles at target rate using accumulator
        accum_emit_np = accum_emit.to_numpy()
        accum_emit_np[...] += emit_rate_variable * dt
        n_to_spawn = int(float(accum_emit_np[()]))
        accum_emit_np[...] -= n_to_spawn
        accum_emit.from_numpy(accum_emit_np)

        if alive_fire_count[None] < max_active_fire_particles and n_to_spawn > 0:
            spawn_fire(n_to_spawn, now, energy)

        # Reconcile air particles with slider target every frame
        current_air = int(alive_air_count[None])
        target_air = int(max_active_air_particles)
        if target_air > MAX_AIR:
            target_air = MAX_AIR
        if target_air < 0:
            target_air = 0

        if current_air < target_air:
            to_add = target_air - current_air
            spawn_air_from(current_air, to_add)
            alive_air_count[None] = target_air
        elif current_air > target_air:
            # Deactivate surplus particles and update count
            deactivate_air_range(target_air, current_air)
            alive_air_count[None] = target_air

        clear_grid()
        accumulate_to_grid()

        step_fire(dt, now, energy)
        step_air(dt, energy)

        # Entropy update chain
        clear_entropy_grid()
        accumulate_air_to_entropy_grid(cube_half)

        # 1. Calculate entropy and find the frame's max value
        calculate_entropy()

        # 2. Use that max value to color each cube independently
        update_mesh_colors()

        # 3. Update wireframes (if desired)
        update_entropy_cube_edges(cube_half)

        eye_x = cam_dist * math.sin(cam_angle)
        eye_z = cam_dist * math.cos(cam_angle)
        camera.position(eye_x, cam_height, eye_z)
        camera.lookat(0.0, 0.6, 0.0)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(55)

        scene.set_camera(camera)
        scene.ambient_light((0.05, 0.05, 0.06))
        scene.point_light(pos=(0.6, 3.2, 2.0), color=(1.0, 0.9, 0.8))
        scene.particles(pos, per_vertex_color=color, radius=PARTICLE_RADIUS)
        scene.particles(pos1, per_vertex_color=color1, radius=PARTICLE_RADIUS)
        scene.lines(points_pos, color = (1.0, 1.0, 1.0), width = 1.0)

        # Draw the colored cubes (each with its own entropy-based color)
        scene.mesh(cube_vertices,
                   indices=cube_indices,
                   per_vertex_color=cube_colors,  # This gets updated every frame
                   two_sided=True)  # important for seeing inside/backfaces

        # Keep your edges if you want wireframe on top (optional)
        scene.lines(entropy_cube_edges, color=(0.9, 0.9, 1.0), width=1.2)

        # Render particles
        scene.particles(pos, per_vertex_color=color, radius=PARTICLE_RADIUS)
        scene.particles(pos1, per_vertex_color=color1, radius=PARTICLE_RADIUS)
        #GUI
        with gui.sub_window("Controls", 0.01, 0.01, 0.25, 0.25):
            gui.text("Sim Controls")
            energy = gui.slider_float("energy", energy, 0.185, 1.0)
            emit_rate_variable = gui.slider_float("Emit Rate", emit_rate_variable, 1000.0, 10000.0)
            max_active_fire_particles = gui.slider_float("Max Fire Particles", max_active_fire_particles, 10000.0, float(MAX_FIRE))
            max_active_air_particles = gui.slider_int("Max Air Particles", max_active_air_particles, 10, int(MAX_AIR))
            gui.text(f"Alive: {alive_fire_count[None]} / {int(max_active_fire_particles)}")

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    # Provide a quick CLI message and run
    try:
        run()
    except KeyboardInterrupt:
        pass