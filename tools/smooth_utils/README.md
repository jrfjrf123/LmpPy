# Distribution Smoothing Module

Distribution smoothing module for IBM (Inverse Boltzmann Method) potential calculation. This module preprocesses bond, angle, dihedral, and RDF distributions before Boltzmann inversion to produce smooth, physically meaningful potentials.

## Features

- **Multiple Distribution Types**: Supports bond, angle, dihedral, and RDF distributions
- **Adaptive Smoothing**: Different window sizes for peak vs valley regions
- **Harmonic Boundary Potential**: Eliminates Gibbs ringing at boundaries for bond/angle distributions
- **Periodic Gaussian Smoothing**: Special handling for dihedral distributions (periodic boundary)
- **RDF Boundary Handling**: Smooth transition to g(r) → 1-ε at large r
- **Quality Metrics**: Peak fidelity, force smoothness, consistency checks
- **Report Generation**: PNG comparison plots and TXT quality reports

---

## Module Structure

The module is organized into the following files:

```
dist_smooth/
├── smooth_constants.py         # Constants and default parameters
├── smooth_io.py                # Data loading functions
├── smooth_preprocess.py        # Preprocessing functions
├── smooth_peaks.py             # Peak detection functions
├── smooth_zones.py             # Zone definition functions
├── smooth_core.py              # Core smoothing functions (bond, rdf)
├── smooth_angle_dihedral.py    # Angle and dihedral smoothing [NEW]
├── smooth_quality.py           # Quality assessment functions
├── smooth_optimize.py          # Optimization functions
├── smooth_report.py            # Report generation functions
├── smooth_utils_init.py        # Unified export interface
├── smooth_all_bonded_distribution.py  # Main script
└── smooth_distribution_backup.py      # Original backup
```

---

## Usage

### Command Line Interface

```bash
# Process a single file
python smooth_all_bonded_distribution.py bond_type1_dist.txt -o smoothed_output/

# Process all files in a directory
python smooth_all_bonded_distribution.py -d distributions/ -o smoothed_output/

# Process only angle and dihedral files
python smooth_all_bonded_distribution.py -d distributions/ -o smoothed_output/ --types angle dihedral

# Specify temperature
python smooth_all_bonded_distribution.py bond_type1_dist.txt -T 300 -o smoothed_output/

# Quiet mode (suppress progress output)
python smooth_all_bonded_distribution.py -q bond_type1_dist.txt
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-d, --directory` | Process all *_dist.txt files in directory |
| `-o, --output` | Output directory (default: smoothed_output) |
| `--types` | Distribution types to process (bond/angle/dihedral/rdf/all) |
| `-T, --temperature` | Temperature in Kelvin (default: 400) |
| `-q, --quiet` | Suppress progress output |

---

## Input/Output

### Input Files

Two-column text files (coordinate, probability):

| File Pattern | Type | Unit | Range |
|--------------|------|------|-------|
| `bond_type{N}_dist.txt` | Bond length distribution | Å | [0, ∞) |
| `angle_type{N}_dist.txt` | Angle distribution | degrees | [0°, 180°] |
| `dihedral_type{N}_dist.txt` | Dihedral distribution | degrees | [-180°, +180°] |
| `rdf_type{X}_{Y}.txt` | Radial distribution function | Å | [0, ∞) |

The distribution type is auto-detected from filename.

### Output Files

For each input file `{name}.txt`:

| Output File | Content |
|-------------|---------|
| `{name}_smoothed.txt` | Smoothed distribution data |
| `{name}_comparison.png` | 3-panel comparison plot |
| `{name}_report.txt` | Quality metrics report |

---

## Smoothing Methods

### 1. Bond Distributions

- **Method**: Harmonic boundary smoothing with S-G core
- **Boundary**: Harmonic potential `U(r) = 0.5*k*(r₀-r)² + F₀*(r₀-r) + U₀`
- **Advantage**: Produces linear force (no oscillations)

### 2. Angle Distributions [NEW]

- **Method**: Same as bond (harmonic boundary)
- **Unit**: Degrees instead of Ångströms
- **Boundary**: θ → 0° and θ → 180° both have harmonic potential

### 3. Dihedral Distributions [NEW]

- **Method**: Periodic Gaussian smoothing
- **Key Feature**: Handles periodic boundary at φ = ±180°
- **Process**: Periodic extension → Gaussian smooth → Extract center → Force boundary continuity

### 4. RDF Distributions

- **Method**: Gaussian smoothing with boundary protection
- **Left boundary**: Harmonic potential (eliminates oscillations at r→0)
- **Right boundary**: Exponential decay to g(r) → 1-ε

---

## Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `load_distribution(filepath)` | smooth_io | Load distribution, auto-detect type |
| `detect_peaks(x, P, ...)` | smooth_peaks | Detect peaks using derivative sign change |
| `define_zones(x, P, ...)` | smooth_zones | Define processing zones |
| `smooth_bond_with_harmonic_boundary(...)` | smooth_core | Bond smoothing |
| `smooth_angle_with_harmonic_boundary(...)` | smooth_angle_dihedral | Angle smoothing [NEW] |
| `smooth_dihedral_periodic(...)` | smooth_angle_dihedral | Dihedral smoothing [NEW] |
| `smooth_rdf_with_harmonic_left_boundary(...)` | smooth_core | RDF smoothing |
| `calculate_potential_and_force(x, P, T)` | smooth_quality | Boltzmann inversion |
| `generate_report(...)` | smooth_report | Generate PNG and TXT reports |

---

## Quality Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| Peak Position Fidelity | 0.50 | How well peak positions are preserved |
| Force Curve Smoothness | 0.35 | Reduction in force roughness (d²F/dx²) |
| Peak Height Fidelity | 0.15 | How well peak heights are preserved |

**Total Score** = 0.50 × Peak Pos + 0.35 × Force Smooth + 0.15 × Peak Height

---

## Theory

### Harmonic Boundary Potential (Bond/Angle)

Boundary regions use harmonic potential:

```
U(x) = 0.5*k*(x₀ - x)² + F₀*(x₀ - x) + U₀
```

This produces linear force: `F(x) = k*(x₀ - x) + F₀`

### Periodic Gaussian Smoothing (Dihedral)

For dihedral distributions with periodic boundary:

```
1. Extend: P_ext = [P[-n_pad:], P, P[:n_pad]]
2. Smooth: Gaussian filter on extended data
3. Extract: Center portion
4. Enforce: P(-180°) = P(+180°)
```

### Boltzmann Inversion

```
U(x) = -kB*T*ln(P(x))
```

---

## Dependencies

**Required:**
- numpy
- scipy
- matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

---

## Example Workflow

1. Calculate distributions using `calculate_all_bonded_dist.py`
2. Run smoothing:
   ```bash
   python smooth_all_bonded_distribution.py -d distributions_output/ -o smoothed_output/
   ```
3. Check output in `smoothed_output/` directory
4. Use smoothed distributions for IBM potential calculation

---

## Version

smooth_distribution 2.0.0 (modular structure with angle/dihedral support)