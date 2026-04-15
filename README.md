# Fire Simulation with Taichi

A real-time 3D fire particle simulation using Taichi, featuring particle dynamics, buoyancy effects, and entropy-based visualization of air particle distribution using colored cubes.

## Features

- **Interactive Fire Particle System**: Physics-based particle dynamics with procedural turbulence and color gradients
- **Air Particle Simulation**: Grid-based density tracking with particle-particle collision handling
- **Entropy Visualization**: Real-time entropy calculation and visualization using a 5×5×5 cube grid with color mapping
- **Real-time 3D Rendering**: Interactive visualization using Taichi's modern UI (Scene/Camera)
- **Adjustable Parameters**: Live GUI sliders for energy, emission rate, particle counts, and more
- **GPU-Accelerated**: Optimized for GPU computation with automatic CPU fallback

## Requirements

- Python 3.8+
- Taichi 1.6.0+
- NumPy 1.21.0+

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/d0br1n/taichi_fire_air_entyropy_simulation.git
   cd taichi_fire_air_entyropy_simulation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main simulation:
```bash
python fire_sim_taichi.py
```

### Controls

- **R** - Reset simulation
- **A/D** - Rotate camera left/right
- **W/S** - Zoom in/out
- **Q/E** - Move camera up/down
- Use GUI sliders to adjust:
  - **Energy** (0.185-1.0) - Affects particle behavior and color intensity
  - **Emit Rate** (1,000-10,000 particles/sec) - Fire emission rate
  - **Max Fire Particles** (10,000-60,000) - Maximum active fire particles
  - **Max Air Particles** (10-10,000) - Maximum active air particles

## Project Structure

```
.
├── fire_sim_taichi.py           # Main simulation and rendering
├── fire_sim_taichi_modified.py  # Alternative/experimental version
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── doc/                         # Documentation folder
└── imgui.ini                    # ImGui configuration (auto-generated)
```

## Technical Details

### Particle System
- **Fire Particles**: Up to 60,000 particles with lifetime-based color transitions
- **Air Particles**: Up to 10,000 particles for density tracking and buoyancy simulation
- Color gradients: Hot (white/yellow) → Orange → Red → Dark Red based on particle lifetime and energy

### Entropy Visualization
- Grid divided into 5×5×5 = 125 cells for entropy tracking
- Entropy calculated using Boltzmann formula: S = k_B × ln(n!)
- Stirling's approximation for computational efficiency
- Color mapping: Low entropy (blue) → High entropy (red)

### Simulation Parameters
- **Base Upward Speed**: 1.5 m/s
- **Gravity**: 0.0 (buoyancy dominates)
- **Drag Coefficient**: 0.6
- **Center Attraction**: Particles drift toward emitter center
- **Density Gradient Push**: Air particles respond to fire density gradients

## Performance Notes

- GPU backend is strongly recommended for smooth simulation
- Optimal particle counts: 30,000-60,000 fire + 1,000-5,000 air particles
- Adjust `MAX_FIRE` and `MAX_AIR` constants in code for different hardware
- Frame rate: Target 60 FPS with V-Sync enabled

## Troubleshooting

**Issue**: "GPU not available, falling back to CPU"
- Solution: Ensure your GPU drivers are up-to-date; CPU mode will work but be slower

**Issue**: Low frame rate
- Solution: Reduce particle counts via the GUI sliders or adjust `MAX_FIRE`/`MAX_AIR` constants

**Issue**: Missing taichi.ui module
- Solution: Update Taichi: `pip install -U taichi`

## License

MIT License

## Author

d0br1n - [GitHub](alexandrudobrin5@gmail.com)

## Acknowledgments

Built with [Taichi](https://taichi-lang.org/) - a high-performance parallel programming language

