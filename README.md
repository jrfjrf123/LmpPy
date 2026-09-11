# LmpPy - LAMMPS Bond/React Post-Processing Framework v2.7

> English | [中文](README.zh-CN.md)

> Version: 2.7
> Date: 2026-09-11
> Language: English

LmpPy is a post-processing framework for LAMMPS `bond/react` and `bond/create` simulations of polymer reaction systems. It converts all-atom (AA) simulations into coarse-grained (CG) trajectories in real time, detects and records bond formation and breaking events, automatically updates the CG mapping and topology after reactions, and provides a rich set of post-processing analysis tools.

The framework targets multiscale modelling of polymer materials: radical copolymerization, epoxy ring-opening, polyurethane step-growth, and crosslinked networks. Beyond trajectory conversion, it supplies the CG distributions needed for iterative Boltzmann inversion (IBI) force-field development, and the per-event data used to fit machine-learning models that replace `bond/react` at the CG level.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dependencies and Installation](#dependencies-and-installation)
- [Quick Start](#quick-start)
- [Running the Simulation](#running-the-simulation)
- [Bond-Creation Modes](#bond-creation-modes)
- [Main Loop Execution Order](#main-loop-execution-order)
- [Configuration File Quick Reference](#configuration-file-quick-reference)
- [CG Mapping Configuration](#cg-mapping-configuration)
- [Restart (Continuation)](#restart-continuation)
- [Output Files Overview](#output-files-overview)
- [Module API Reference](#module-api-reference)
- [Tool Modules](#tool-modules)
- [Post-processing Analysis](#post-processing-analysis)
- [CLI Scripts](#cli-scripts)
- [Smoke Test](#smoke-test)
- [FAQ](#faq)
- [Module Dependencies and Data Flow](#module-dependencies-and-data-flow)
- [Changelog](#changelog)

---

## Project Overview

### Core Features

1. **CG trajectory generation** — converts all-atom trajectories of reacting polymer systems into coarse-grained trajectories in real time
2. **Reaction monitoring** — detects bond formation and breaking events and counts them per cycle (supports both bond/react and bond/create modes), providing the raw data for monomer conversion and reactivity-ratio statistics
3. **CG mapping update** — automatically updates the atom-to-bead mapping after reactions
4. **CG topology derivation** — derives coarse-grained bonds, angles, and dihedrals from bond connectivity
5. **Restart / continuation** — supports continuing a calculation from the final state of a previous run

### Highlights

| Feature | Description |
|---------|-------------|
| **Multi-system support** | Define different polymer reaction systems via configuration files (radical copolymerization, epoxy ring-opening, polyurethane step-growth, crosslinked networks, etc.) |
| **Dual bond-creation modes** | bond/react template matching + bond/create distance criteria, covering different scenarios |
| **Force-field development** | IBI (iterative Boltzmann inversion) pipeline from CG distributions to tabulated potentials, including cross-mixing for unlike bead pairs |
| **GROMACS interoperability** | Convert GROMACS `top` + `gro` (GAFF) systems into LAMMPS `data` files, so existing AA force fields can be reused directly |
| **Configuration driven** | All parameters are defined in YAML configuration files; no code changes required |
| **High performance** | Vectorized/Numba-optimized key computations; some operations accelerated 30–300x |
| **MPI parallelism** | Supports multi-process parallel execution |
| **Modular design** | Clear module separation, easy to maintain and extend |
| **Comprehensive documentation** | Detailed technical documentation (`docs/` directory) covering configuration, workflow, API, output files, and more |

### Performance Optimizations

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| Vectorized image flag decoding | **362x** | 8.2ms → 0.023ms |
| Numba find_molecules | **30–80x** | BFS molecule finding |
| Numba unwrap_coords | **4–9x** | Coordinate unwrapping |
| Lazy index caching | O(n^2) → O(n) | CG coordinate conversion |

### Data Flow (Three-Repository Pipeline)

LmpPy is the first stage of a larger multiscale pipeline, working together with the following repositories:

```
LmpPy (CG post-processing) → SOAP_calc_and_Feature_select (SOAP calculation + feature selection) → mlcgsim (XGBoost driven)
```

The AA stage produces reaction events and CG mappings; SOAP descriptors are computed around the reacting bonds and screened down to a compact feature set; an XGBoost classifier trained on those features then drives CG simulations without invoking `bond/react` at runtime.

See [docs/workflow.md](docs/workflow.md) for the full data flow description.

---

## Project Structure

```
LmpPy/
├── __init__.py                       # Package entry point
├── run_refactored.py                 # Main entry script (LAMMPSReactionRunner)
├── pyproject.toml                    # Package metadata and dependency declaration
├── test_integration.py               # Integration tests
├── test_gmx2lmp_data.py              # gmx2lmp_data test suite
├── test_reactivity_ratio.py          # Reactivity-ratio statistics test suite
│
├── core/                             # Core functionality modules
│   ├── __init__.py                   # Exports all core classes and functions
│   ├── config_loader.py              # YAML configuration loader (ConfigLoader, SystemConfig, LAMMPSParams)
│   ├── mapping_generator.py          # CG mapping generator (CGCompareList, MappingGenerator)
│   ├── template_parser.py            # LAMMPS template parser (TemplateParser, ReactionTemplate)
│   ├── lammps_data_extractor.py      # LAMMPS data extractor (LAMMPSDataExtractor, AtomData, BondData)
│   ├── bond_detector.py              # Bond change detector (BondDetector, BondChanges)
│   ├── reaction_locator.py           # Reaction site locator (ReactionLocator, ReactionMatch)
│   ├── cg_reaction_identifier.py     # CG-level reaction identification (v2.6+, template signature matching / cross-validation)
│   ├── reaction_commands.py          # Reaction command generation (v2.6+, bond/create + bond/react commands)
│   ├── cg_mapper.py                  # CG mapping updater (CGMapper, CGMapping)
│   ├── cg_converter.py               # CG coordinate converter (CGConverter)
│   ├── cg_bond_mapper.py             # Coarse-grained bond mapper (CGBondMapper)
│   ├── cg_topology.py                # CG topology deriver (CGTopology)
│   ├── bonds_recorder.py             # Bond table recorder (BondsRecorder, backward compatible)
│   ├── cg_initializer.py             # CG system initializer (CGInitializer, CGSystem)
│   ├── smoke_validator.py            # Smoke test validator (SmokeValidator)
│   └── smoke_test_harness.py         # Smoke test orchestrator (SmokeTestHarness)
│
├── output_analysis/                  # Post-processing analysis subpackage (v2.6+)
│   ├── __init__.py
│   ├── __main__.py                   # `python -m LmpPy.output_analysis` entry point
│   ├── cli.py                        # Unified CLI entry point
│   ├── chain_length.py               # Chain length distribution analysis
│   ├── distance.py                   # Reaction distance distribution analysis
│   ├── reaction_stats.py             # Reaction statistics
│   ├── reactivity_ratio.py           # Reactivity ratio (r1/r2) estimation, CG and AA paths
│   ├── js_divergence.py              # JS divergence calculation
│   ├── loader.py                     # Data loading utilities
│   ├── theory.py                     # Theoretical distribution models (Schulz-Zimm, Poisson, Log-normal)
│   └── plot.py                       # Unified plotting configuration
│
├── utils/                            # Utility functions
│   ├── __init__.py
│   ├── coordinate_utils.py           # Coordinate handling (wrap, PBC distance, unwrap)
│   ├── file_utils.py                 # File I/O (dump read/write)
│   ├── graph_utils.py                # Graph algorithms (find_molecules, Numba optimized)
│   ├── topology.py                   # Topology file read/write
│   └── units.py                      # Unit conversion
│
├── tools/                            # Standalone tool modules
│   ├── __init__.py
│   ├── aa2cg/                        # All-atom to coarse-grained conversion
│   │   ├── data_converter.py         # LAMMPS data conversion
│   │   ├── trj_converter.py          # Trajectory conversion
│   │   └── mapping_utils.py          # Mapping utility functions
│   ├── ibm_potential/                # IBM / IBI potential calculation
│   │   ├── config.py                 # Configuration loading
│   │   ├── distribution.py           # Distribution calculation
│   │   ├── boltzmann.py              # Boltzmann inversion
│   │   ├── lammps_table.py           # LAMMPS table generation
│   │   ├── tabulated_potential.py    # Tabulated potential parsing / fitting / plotting
│   │   ├── gromacs_loader.py         # GROMACS trajectory loading
│   │   ├── pickle_loader.py          # Pickled CG trajectory loading
│   │   ├── dist_config.py            # Distribution plot configuration
│   │   └── dist_plot.py              # Distribution plotting
│   └── smooth_utils/                 # Distribution smoothing toolkit
│       ├── core.py / cli.py / constants.py / io.py
│       ├── preprocess.py / peaks.py / zones.py / quality.py
│       ├── optimize.py / angle_dihedral.py
│       └── report.py
│
├── scripts/                          # CLI scripts
│   ├── build_cg_config.py            # Generate CG configuration
│   ├── build_cg_system.py            # Build CG system LAMMPS data file
│   ├── gmx2lmp_data.py               # GROMACS top+gro → LAMMPS data conversion
│   ├── convert_aa2cg.py              # AA→CG conversion CLI
│   ├── generate_initial_mapping.py   # Generate initial CG mapping
│   ├── yaml2csv_mapping.py           # YAML→CSV mapping conversion
│   ├── data2gro.py                   # LAMMPS data to GRO conversion
│   ├── smooth_distribution.py        # Distribution smoothing CLI
│   ├── calc_dist.py                  # Distribution calculation CLI
│   ├── plot_dist.py                  # Distribution plotting CLI
│   ├── calc_ibm_potential.py         # IBM potential calculation CLI
│   ├── calc_ibm_potential_from_dist.py # IBM potential from precomputed distributions
│   ├── fit_tabulated.py              # Fit tabulated potentials to analytic forms
│   ├── mix_cross_tabulated.py        # Cross-mix tabulated potentials for unlike pairs
│   ├── plot_tabulated.py             # Plot tabulated potentials
│   ├── extract_reaction_frame.py     # Extract a single reaction frame for visualization
│   ├── validate_config.py            # Configuration validation
│   └── test_smoke_run.py             # Standalone smoke test runner script
│
├── config/                           # Configuration file templates
│   ├── system.yaml                   # System configuration template
│   ├── lammps_params.yaml            # LAMMPS run parameter template
│   ├── mass_list.yaml                # Atom mass list
│   └── mapping/                      # CG mapping configuration templates
│
├── docs/                             # Documentation and examples
│   ├── bond-modes.md                 # Bond-creation modes (bond/react vs bond/create)
│   ├── configuration.md              # Complete configuration parameter reference
│   ├── cg-mapping.md                 # CG mapping configuration details
│   ├── workflow.md                   # Run workflow and data flow
│   ├── output-files.md               # Output file format description
│   ├── modules.md                    # Module API reference
│   ├── cli-scripts.md                # CLI scripts usage guide
│   ├── gro_format.md                 # GRO format notes
│   ├── lammps_data_format.md         # LAMMPS data format notes
│   ├── reaction_locator_fix.md       # ReactionLocator fix notes
│   └── examples/                     # Example configuration files
│       ├── bond_react/               # bond/react mode configuration example
│       ├── bond_create/              # bond/create mode configuration example
│       └── reactions/                # Reaction template examples
│
├── README.md                         # English README (this document)
└── README.zh-CN.md                   # Chinese README
```

---

## Dependencies and Installation

### 1. Create a Virtual Environment (conda)

```bash
conda create -n lmp_py_react python=3.10 -y
conda activate lmp_py_react
```

`pyproject.toml` requires Python >= 3.9; 3.10 is recommended. Without conda, the equivalent `venv` setup is:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

Every `pip` command below must be run inside that environment (the shell prompt shows the environment name after `conda activate`).

### 2. Install Dependencies

The required dependencies are imported unconditionally by `core/` and `utils/`:

```bash
pip install numpy pandas pyyaml
```

Numba is strongly recommended (JIT acceleration for `find_molecules` and `unwrap_coords`); without it LmpPy falls back to pure Python:

```bash
pip install numba
```

The remaining functionality is grouped into extras declared in `pyproject.toml`. **`cd` into the repository root containing `pyproject.toml` first**, then pick what you need:

```bash
pip install ".[mpi]"        # mpi4py — MPI parallel execution
pip install ".[analysis]"   # scipy / matplotlib / seaborn / tqdm — post-processing analysis and plotting
pip install ".[tools]"      # MDAnalysis / scikit-optimize — GROMACS conversion and IBI potential fitting
pip install ".[dev]"        # pytest / black — development and testing
```

To install everything except LAMMPS in one go:

```bash
pip install ".[mpi,analysis,tools,dev]"
```

`numba`, `mpi4py` and the LAMMPS Python interface are all imported inside `try`/`except` blocks: if they are missing, LmpPy falls back to pure-Python implementations or prints a warning and runs in test mode, so the package remains importable in a minimal environment.

### 3. Compiling LAMMPS

LAMMPS must be compiled with Python interface support:

```bash
cd lammps/src
make yes-python yes-mpi
make mpi
```

---

## Quick Start

### 1. Prepare Configuration Files

Create a configuration directory containing the following files (see `docs/examples/` for examples):

```
my_system/
├── system.yaml                    # System configuration
├── lammps_params.yaml             # LAMMPS run parameters
├── molecule1_mapping.yaml         # Molecule CG mapping configuration
├── system.data                    # LAMMPS initial data file
└── reactions/                     # Reaction configuration directory (required for bond/react mode)
    └── rxn1/
        ├── rxn1_pre.lammpstemplate
        ├── rxn1_post.lammpstemplate
        ├── rxn1.map
        ├── rxn1_pre_mapping.yaml
        └── rxn1_post_mapping.yaml
```

**Documentation navigation**:
- `docs/examples/` — Runnable mini_test system (config + mapping + data + reaction templates)
- [docs/examples/bond_react/](docs/examples/bond_react/) — bond/react field reference (YAML skeleton only; supply your own mapping/data/reaction files)
- [docs/examples/bond_create/](docs/examples/bond_create/) — bond/create field reference (YAML skeleton only; supply your own mapping/data files)
- [docs/configuration.md](docs/configuration.md) — Detailed configuration parameter fields
- [docs/cg-mapping.md](docs/cg-mapping.md) — CG mapping format description

> The detailed documents under `docs/` are written in Chinese, which is the framework's working language.

If the all-atom system already exists in GROMACS form, `system.data` can be generated from a GAFF `top` + `gro` pair instead of being written by hand:

```bash
python -m LmpPy.scripts.gmx2lmp_data --top system.top --gro system.gro -o my_system/system.data
```

### 2. Run the Simulation

```bash
# Test mode (load configuration only, do not run LAMMPS)
python -m LmpPy.run_refactored my_system/ --test

# Single-process run
python -m LmpPy.run_refactored my_system/

# MPI parallel run (4 processes)
mpirun -np 4 python -m LmpPy.run_refactored my_system/ --loop-num 100

# Smoke test (5 loops, automatic output validation)
python -m LmpPy.run_refactored my_system/ --smoke-test --loop-num 5
```

---

## Running the Simulation

### Command-Line Arguments

```bash
python -m LmpPy.run_refactored <config_dir> [options]

Arguments:
  config_dir            Path to the configuration directory (positional, required)

Options:
  --test                Test mode: load configuration only, do not run LAMMPS
  --loop-num N          Override the loop count in the configuration file
  --smoke-test          Smoke test mode (temporary directory + automatic output validation)
```

### MPI Parallel Execution

```bash
# 4-process parallelism
mpirun -np 4 python -m LmpPy.run_refactored config/

# 8-process parallelism
mpirun -np 8 python -m LmpPy.run_refactored config/ --loop-num 100
```

### Environment Variables

```bash
# Set the number of OpenMP threads (default is 1)
export OMP_NUM_THREADS=1
```

---

## Bond-Creation Modes

LmpPy supports two LAMMPS bond-creation modes, selected mutually exclusively via `lammps_params.yaml`:

| Mode | Bonding Strategy | Configuration Complexity | Best For |
|------|------------------|--------------------------|----------|
| **bond/react** | Template chemistry matching (pre/post .lammpstemplate) | High | Well-defined chemical reactions (epoxy ring opening, polyurethane formation, etc.) |
| **bond/create** | Distance + atom type criteria (no templates) | Low | Crosslinking, percolating networks, non-specific bond formation |

Core difference: bond/react relies on molecule template matching, and the product structure is strictly defined by the templates; bond/create decides whether to form a bond based only on atom types and spatial distance, with no template files required.

**Selection guide**:

- Well-defined reaction chemistry, precise control over product structure needed → **bond/react**
- Random crosslinking, rapid prototyping, probabilistic bonding → **bond/create**
- Building a new system from scratch → **bond/create** (faster)

**Detailed documentation**: [docs/bond-modes.md](docs/bond-modes.md)

---

## Main Loop Execution Order

Each cycle of the LmpPy simulation main loop executes in the following order:

| Step | Description | Modules Involved |
|------|-------------|------------------|
| **Step 1** | Run bond-creation commands (bond/react or bond/create) | LAMMPS fix |
| **Step 2** | Detect reactions: extract data, detect bond changes, locate reactions, update CG mapping, convert CG coordinates, write trajectory | `LAMMPSDataExtractor` → `BondDetector` → `ReactionLocator` → `CGMapper` → `CGConverter` |
| **Step 3** | Relaxation: NVE/limit + NVT (reacted atoms) → NPT (all atoms) | LAMMPS command sequence |
| **Step 4** | Update: invalidate the data extractor cache, prepare for the next round | `LAMMPSDataExtractor.invalidate_cache()` |

**Step 2 detailed flow**:

```
1. LAMMPSDataExtractor: extract atom and bond data
2. BondDetector: detect bond changes (created_bonds, deleted_bonds)
3. ReactionLocator: reaction template matching (dual validation)
4. CGMapper: update CG mapping (based on ReactionMatch)
5. CGConverter: convert CG coordinates (lazy index caching optimization)
6. Write CG trajectory frame (post-reaction, pre-relaxation)
7. Cache reaction frame data to reaction_frames.npz
```

**Detailed documentation**: [docs/workflow.md](docs/workflow.md)

---

## Configuration File Quick Reference

LmpPy is driven by YAML configuration files. There are two main configuration files: `system.yaml` and `lammps_params.yaml`.

### system.yaml Field Summary

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `system.name` | string | No | System name, default `"unnamed"` |
| `system.bead_type_names` | dict | **Yes** | Mapping of bead type names → LAMMPS type numbers |
| `system.mapping_files[].path` | string | **Yes** | Path to the molecule mapping YAML file |
| `system.mapping_files[].copies` | int | **Yes** | Number of molecules of this type in the initial data file |
| `system.work_dir` | string | No | Working directory, default `"work"` |
| `system.output_dir` | string | No | Output directory, default `"output"` |

### lammps_params.yaml Field Summary

| Section | Key Fields | Description |
|---------|-----------|-------------|
| `simulation` | loop_num, dt, temperature, pressure, ensemble | Basic simulation parameters |
| `steps` | bond_react_check, run_per_loop, nve_limit | Step allocation |
| `npt` | tcouple, pcouple | Temperature/pressure coupling parameters |
| `bond_react` | stabilization, reactions[] | bond/react mode configuration |
| `bond_create` | enabled, pairs[], cg_update | bond/create mode configuration |
| `molecules` | {name: template_path} | Molecule template mapping |
| `files` | data_file, initial_cg_mapping, output_* | Input/output file paths |

**Detailed documentation**: [docs/configuration.md](docs/configuration.md)

---

## CG Mapping Configuration

The CG mapping defines which atoms in the all-atom simulation make up a coarse-grained bead, along with the bead's mass weights.

### YAML Mapping Format

```yaml
# 1. site-types: bead type definitions
site-types:
  Bead1:
    index: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]   # relative atom indices (0-based)
    x-weight: [12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1] # mass weights

# 2. config: mapping configuration
config:
  - anchor: 0                   # fixed to 0 (the program accumulates offsets automatically)
    repeat: 1                   # number of molecules
    offset: 23                  # number of atoms per molecule
    sites:
      - [Bead1, 0]              # [bead type name, starting position]
      - [Bead2, 4]
      - [Bead3, 17]
```

### Core Rules

- `anchor` must be 0; the program automatically handles cumulative offsets for multiple molecules and multiple files
- Atom ID formula: `atom_id = anchor + site_start_offset + index_value + 1`
- `x-weight` is used for center-of-mass calculation: `R_bead = sum(m_i * r_i) / sum(m_i)`
- Reactions require separate pre/post mapping files (`pre_mapping` / `post_mapping`) reflecting bead_type changes

**Detailed documentation**: [docs/cg-mapping.md](docs/cg-mapping.md)

---

## Restart (Continuation)

The program supports continuing a calculation from the final state of a previous run, without re-simulating completed steps.

### How to Use

1. Copy `final_frame.data` and `final_cg_compare_list.csv` from the previous run into the new configuration directory
2. Update the file paths in `lammps_params.yaml`:

```yaml
files:
  data_file: "final_frame.data"                # use the final data file from the previous run
  initial_cg_mapping: "final_cg_compare_list.csv"  # use the final CG mapping from the previous run
```

3. Set `loop_num` to the remaining number of loops needed, then run:

```bash
mpirun -np 4 python -m LmpPy.run_refactored continue_config/ --loop-num 50
```

### How It Works

- The program checks whether the file specified by `initial_cg_mapping` exists
- If it exists, the existing CG mapping is loaded directly (the generation step is skipped)
- If it does not exist, the mapping is generated from `mapping_files` in `system.yaml`

---

## Output Files Overview

Core files produced by the main simulation pipeline:

| File | Format | Description |
|------|--------|-------------|
| `cg_trajectory.lammpstrj` | LAMMPS dump | CG trajectory file, visualizable in VMD |
| `reaction_num.txt` | Text | Per-step reaction count statistics |
| `final_frame.data` | LAMMPS data | Final frame atom data (for restart) |
| `final_cg_compare_list.csv` | CSV (5 columns) | Final CG mapping |
| `reaction_frames.npz` | NPZ (7 arrays) | Detailed reaction frame data (input for SOAP calculation) |
| `cg_bonds.txt` / `cg_angles.txt` / `cg_dihedrals.txt` | Text | CG topology files |

**Detailed documentation**: [docs/output-files.md](docs/output-files.md)

### reaction_frames.npz Core Contents

This is the **core input data source for the SOAP calculation module** and contains 7 arrays:

| Array Name | Shape | Description |
|------------|-------|-------------|
| `aa_coords_before` | (n_reactions, n_atoms, 3) | AA coordinates before reaction |
| `aa_coords_after` | (n_reactions, n_atoms, 3) | AA coordinates after reaction |
| `aa_bonds_before` | (n_reactions,) object | Atom bonds before reaction |
| `aa_bonds_after` | (n_reactions,) object | Atom bonds after reaction |
| `cg_mapping_before` | (n_reactions, n_atoms, 2) | CG mapping before reaction [bead_id, bead_type] |
| `cg_mapping_after` | (n_reactions, n_atoms, 2) | CG mapping after reaction |
| `timestep` | (n_reactions,) | Reaction timestep |

---

## Module API Reference

### Importing Core Modules

```python
from LmpPy.core import (
    # Configuration
    ConfigLoader, SystemConfig, LAMMPSParams,
    # Mapping
    MappingGenerator, CGCompareList, generate_cg_compare_list,
    # Templates
    TemplateParser, ReactionTemplate, load_all_reaction_templates,
    # Data extraction
    LAMMPSDataExtractor, AtomData, BondData,
    # Reaction processing
    BondDetector, BondChanges,
    ReactionLocator, ReactionMatch,
    CGMapper, CGMapping,
    CGConverter, lammpstrj2cg,
    CGBondMapper, atom_bonds_to_cg_bonds,
    BondsRecorder,
    # CG system
    CGInitializer, CGSystem, initialize_cg_system,
    # CG topology
    CGTopology, derive_cg_topology_from_bonds,
    # New in v2.6+
    BondCreateConfig,
    get_reaction_mode, generate_fix_bond_create, generate_fix_bond_react,
    update_cg_mapping_create,
)

# The symbols below are not re-exported by core/__init__.py
from LmpPy.core.reaction_commands import BondCreatePair
from LmpPy.core.cg_reaction_identifier import (
    CGReactionSignature, load_template_signatures, identify_reaction,
)
from LmpPy.core.smoke_test_harness import SmokeTestHarness
from LmpPy.core.smoke_validator import SmokeValidator, SmokeTestReport
```

### Usage Examples

```python
# Configuration loading
loader = ConfigLoader("config/")
system_config = loader.load_system_config()
lammps_params = loader.load_lammps_params()

# Generate CG mapping
cg_list = MappingGenerator().generate_from_system(system_config, Path("config/"))
print(f"Beads: {cg_list.n_beads}, Molecules: {cg_list.n_molecules}")

# Initialize the full CG system
cg_system = initialize_cg_system("config/")
print(f"CG bonds: {len(cg_system.bonds)}")

# Load reaction templates
templates = load_all_reaction_templates(Path("config/reactions/"))

# Bond change detection
changes = BondDetector(n_atoms=10000).detect(bonds_before, bonds_after)
print(f"Created bonds: {len(changes.created_bonds)}, Deleted bonds: {len(changes.deleted_bonds)}")
```

**Detailed API documentation**: [docs/modules.md](docs/modules.md)

---

## Tool Modules

LmpPy provides three standalone tool subpackages under the `LmpPy/tools/` directory:

### aa2cg — All-Atom to Coarse-Grained Conversion

Supports conversion from GROMACS TRR trajectories, LAMMPS data files, and GRO+TPR structure files to CG format.

```python
from LmpPy.tools.aa2cg import (
    load_aa_to_cg_mapping,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
)
```

### ibm_potential — IBM / IBI Potential Calculation

Full pipeline from CG trajectories through distribution calculation and Boltzmann inversion to LAMMPS table file generation, plus parsing, fitting and plotting of the resulting tabulated potentials.

```python
from LmpPy.tools.ibm_potential import (
    # Distribution → potential → table
    load_ibm_config,
    calculate_bond_distribution,
    calculate_bond_potential,
    create_lammps_table_files,
    # Tabulated potential parsing / fitting / plotting
    read_tabulated_table,
    fit_table,
    fit_lj,
    fit_bond_harmonic,
    plot_single_table,
    plot_all_tables,
)
```

Both GROMACS trajectories (`gromacs_loader.py`) and pickled CG trajectories (`pickle_loader.py`) are supported as input.

### smooth_utils — Distribution Smoothing Toolkit

Supports adaptive Savitzky-Golay smoothing, Gaussian smoothing, and boundary-constrained smoothing.

```python
from LmpPy.tools.smooth_utils import (
    smooth_distribution,
    smooth_bond_with_harmonic_boundary,
)
```

**Detailed documentation**: [docs/modules.md#2-工具模块-tools](docs/modules.md#2-工具模块-tools)

---

## Post-processing Analysis

The `output_analysis/` subpackage (v2.6+) provides post-processing statistical analysis of simulation results: chain length distribution, reaction distance distribution, reaction statistics, JS divergence, and copolymer reactivity ratio.

### CLI Entry Point

```bash
# Chain length distribution analysis
python -m LmpPy.output_analysis chain-length <input_dir> [--min-length N] [--theory-type TYPE]

# Reaction distance distribution analysis
python -m LmpPy.output_analysis distance <input_dir> [--bins N]

# Reaction statistics
python -m LmpPy.output_analysis reaction-stats <input_dir>

# Reactivity ratio r1/r2 from CG simulation output
python -m LmpPy.output_analysis reactivity-ratio <input_dir>

# Reactivity ratio r1/r2 from AA bond/react output
# (--bond-react-check-step must match steps.bond_react_check in lammps_params.yaml)
python -m LmpPy.output_analysis reactivity-ratio-aa <aa_dir> [--bond-react-check-step N]

# Run all analyses
python -m LmpPy.output_analysis all <input_dir>

# Rebuild reaction details from npz
python -m LmpPy.output_analysis rebuild <npz_path> -o <csv_path>
```

The two reactivity-ratio commands estimate the monomer reactivity ratios r1/r2 from per-cycle reaction counts, using block bootstrap for confidence intervals and reporting conversion-resolved evolution. They differ only in the data source: `reactivity-ratio` reads CG simulation output, while `reactivity-ratio-aa` reconstructs the counts from AA `reaction_frames.npz` plus the accompanying trajectory.

### Python API

```python
from LmpPy.output_analysis.chain_length import analyze_chain_length
from LmpPy.output_analysis.distance import analyze_distance
from LmpPy.output_analysis.reaction_stats import analyze_reaction_stats
from LmpPy.output_analysis.js_divergence import js_divergence

analyze_chain_length("output/", theory_type="schulz-zimm", Mn=5000, PDI=2.0)
analyze_distance("output/", bins=50)
stats = analyze_reaction_stats("output/")
```

**Detailed documentation**: [docs/modules.md#4-后处理分析-output_analysis](docs/modules.md#4-后处理分析-output_analysis)

---

## CLI Scripts

LmpPy provides a series of CLI scripts that can be run as `python -m LmpPy.scripts.<name>`:

**System preparation**

| Script | Purpose | Detailed Documentation |
|--------|---------|------------------------|
| `gmx2lmp_data.py` | GROMACS `top`+`gro` (GAFF) → LAMMPS data file | [docs/cli-scripts.md#15-gmx2lmp_datapy](docs/cli-scripts.md#15-gmx2lmp_datapy) |
| `build_cg_config.py` | Generate CG configuration | [docs/cli-scripts.md#2-build_cg_configpy](docs/cli-scripts.md#2-build_cg_configpy) |
| `build_cg_system.py` | Build CG system LAMMPS data file | [docs/cli-scripts.md#3-build_cg_systempy](docs/cli-scripts.md#3-build_cg_systempy) |
| `convert_aa2cg.py` | AA→CG conversion (data file or trajectory) | [docs/cli-scripts.md#7-convert_aa2cgpy](docs/cli-scripts.md#7-convert_aa2cgpy) |
| `generate_initial_mapping.py` | Generate initial CG mapping | [docs/cli-scripts.md#9-generate_initial_mappingpy](docs/cli-scripts.md#9-generate_initial_mappingpy) |
| `yaml2csv_mapping.py` | YAML→CSV mapping conversion | [docs/cli-scripts.md#14-yaml2csv_mappingpy](docs/cli-scripts.md#14-yaml2csv_mappingpy) |
| `data2gro.py` | LAMMPS data to GRO conversion | [docs/cli-scripts.md#8-data2gropy](docs/cli-scripts.md#8-data2gropy) |

**Distributions and potentials**

| Script | Purpose | Detailed Documentation |
|--------|---------|------------------------|
| `calc_dist.py` | Distribution calculation (VOTCA format, `--skip-existing` for incremental reruns) | [docs/cli-scripts.md#4-calc_distpy](docs/cli-scripts.md#4-calc_distpy) |
| `smooth_distribution.py` | Distribution smoothing | [docs/cli-scripts.md#11-smooth_distributionpy](docs/cli-scripts.md#11-smooth_distributionpy) |
| `plot_dist.py` | Distribution plotting | [docs/cli-scripts.md#10-plot_distpy](docs/cli-scripts.md#10-plot_distpy) |
| `calc_ibm_potential.py` | IBM potential calculation full pipeline | [docs/cli-scripts.md#5-calc_ibm_potentialpy](docs/cli-scripts.md#5-calc_ibm_potentialpy) |
| `calc_ibm_potential_from_dist.py` | IBM potential from precomputed distributions | [docs/cli-scripts.md#6-calc_ibm_potential_from_distpy](docs/cli-scripts.md#6-calc_ibm_potential_from_distpy) |
| `fit_tabulated.py` | Fit tabulated potentials to analytic forms (LJ well/direct, harmonic, cosine) | — |
| `mix_cross_tabulated.py` | Cross-mix two tabulated potentials into an unlike-pair table | — |
| `plot_tabulated.py` | Plot tabulated potentials and fits | — |

**Utilities**

| Script | Purpose | Detailed Documentation |
|--------|---------|------------------------|
| `extract_reaction_frame.py` | Extract a single reaction frame for visualization | — |
| `validate_config.py` | Configuration validation | [docs/cli-scripts.md#13-validate_configpy](docs/cli-scripts.md#13-validate_configpy) |
| `test_smoke_run.py` | Standalone smoke test runner | [docs/cli-scripts.md#12-test_smoke_runpy](docs/cli-scripts.md#12-test_smoke_runpy) |

**Detailed documentation**: [docs/cli-scripts.md](docs/cli-scripts.md)

---

## Smoke Test

The smoke test is a fast correctness verification tool introduced in v2.5. It runs the complete simulation pipeline for a small number of loops and automatically validates the integrity and consistency of the output files.

### Design Goals

- **Fast verification**: after configuration changes, confirm the pipeline runs correctly with a small number of loops (default 5)
- **Correctness guarantee**: automatically check output file integrity and data consistency
- **CI/CD friendly**: returns standard exit codes (0 = passed, 1 = failed)

### Four Checks

| Check | Name | Description |
|-------|------|-------------|
| A | File output integrity | Check that all expected output files exist and have reasonable content |
| B | CG mapping consistency | Verify bead_type uniqueness, no duplicate or missing AA_ids |
| C | Post-reaction mapping cross-validation | Reconstruct the expected mapping with an independent CG-level signature algorithm and compare |
| D | Reaction count sanity | Check that counts are non-negative and the line count matches loop_num |

### How to Use

```bash
# Method 1: via run_refactored.py (recommended)
python -m LmpPy.run_refactored my_system/ --smoke-test
python -m LmpPy.run_refactored my_system/ --smoke-test --loop-num 3

# Method 2: via the standalone script
python LmpPy/scripts/test_smoke_run.py my_system/ --loop-num 3 --keep-output
```

### Example Output

The report itself is printed in Chinese (the framework's working language); the block below is verbatim console output:

```
============================================================
Smoke Test 报告
============================================================
  配置目录: /path/to/my_system
  循环次数: 5
    [✅] A. 文件输出完整性
    [✅] B. CG mapping 一致性
    [✅] C. 反应后 mapping 更新正确
    [✅] D. 反应计数合理性
  结果: 通过
============================================================
```

### Programming Interface

```python
from LmpPy.core.smoke_test_harness import SmokeTestHarness

harness = SmokeTestHarness("config/", loop_num=5)
report = harness.run()
if report is not None:
    report.print()
    sys.exit(0 if report.passed else 1)
```

---

## FAQ

### Q1: IndexError: index X is out of bounds

**Cause**: CG mapping configuration error; atom ID out of range.

**Solution**: Check that `offset` and `repeat` in `mapping_files` are correct, and ensure all `anchor` values are 0.

### Q2: Bond atoms missing on proc

**Cause**: Atoms moved outside the box boundaries, or bond definition problems.

**Solution**: Check the bond definitions in the initial data file, reduce the timestep, increase the box size.

### Q3: Reactions do not occur

**Possible causes**:
- Reaction cutoff radius `cutoff` too small
- Molecules initially too far apart
- Reaction template mismatch (bond/react mode)

**Solution**: Increase the `cutoff` value, check template matching against the actual molecules, extend the simulation time.

### Q4: Tab character YAML parse error

**Cause**: YAML file contains tab characters.

**Solution**:
```bash
sed -i 's/\t/  /g' *.yaml
```

### Q5: MPI process communication error

**Cause**: MPI environment configuration problem.

**Solution**: Ensure all processes can access the same file system; check the MPI installation and environment variables.

### Q6: Bonds not created in bond_create mode

**Cause**: `Rmin` in the bond/create configuration is too small, or the `maxbond` limit has been reached.

**Solution**: Increase the `Rmin` value, check whether the `iparam.maxbond` / `jparam.maxbond` settings are reasonable, confirm `enabled: true` is set and `bond_react.reactions` is not configured.

### Q7: Error when bond_create and bond_react are both enabled

**Cause**: The two modes are mutually exclusive and cannot be enabled at the same time.

**Solution**: Ensure that when `bond_create.enabled=true`, `bond_react.reactions` does not exist or is an empty list; and vice versa.

### Q8: reactivity-ratio-aa reports a bond_react_check_step mismatch

**Cause**: The AA path locates the pre-reaction frame as `timestep - bond_react_check_step`. Its default is 1, so any configuration with a larger `steps.bond_react_check` will miss the frame.

**Solution**: Pass `--bond-react-check-step` with the same value used in `lammps_params.yaml`. The command raises an error rather than silently reading the wrong frame.

---

## Module Dependencies and Data Flow

### Module Call Graph

```
run_refactored.py
    ├── core/config_loader.py
    │   └── ConfigLoader → SystemConfig, LAMMPSParams, ReactionInfo
    ├── core/mapping_generator.py
    │   └── MappingGenerator → CGCompareList
    ├── core/template_parser.py
    │   └── TemplateParser → ReactionTemplate (bond/react mode)
    ├── core/reaction_commands.py
    │   └── load_bond_create_config, generate_fix_bond_* (v2.6+)
    ├── core/cg_reaction_identifier.py
    │   └── load_template_signatures, identify_reaction (v2.6+)
    ├── core/lammps_data_extractor.py
    │   └── LAMMPSDataExtractor → AtomData, BondData
    │   └── decode_image_flags_vectorized() (362x speedup)
    ├── core/bond_detector.py
    │   └── BondDetector → BondChanges
    ├── core/reaction_locator.py
    │   └── ReactionLocator → ReactionMatch
    ├── core/cg_mapper.py
    │   └── CGMapper → CGMapping
    ├── core/cg_converter.py
    │   └── CGConverter (lazy index caching)
    ├── core/cg_bond_mapper.py
    │   └── atom_bonds_to_cg_bonds()
    ├── core/cg_topology.py
    │   └── CGTopology, derive_cg_topology_from_bonds()
    ├── core/bonds_recorder.py
    │   └── BondsRecorder (backward compatible)
    ├── core/cg_initializer.py
    │   └── CGInitializer → CGSystem
    ├── utils/graph_utils.py
    │   └── find_molecules() (Numba 30-80x)
    ├── utils/coordinate_utils.py
    │   ├── wrap_coordinates(), pbc_distance()
    │   └── unwrap_coords_python()
    └── utils/file_utils.py
        ├── write_lammps_dump_file()
        └── write_cg_trajectory()
```

### Data Flow

```
1. Configuration loading
   ConfigLoader.load_system_config() → SystemConfig
   ConfigLoader.load_lammps_params() → LAMMPSParams

2. Initialization
   MappingGenerator.generate() → CGCompareList (initial CG mapping)
   TemplateParser / ReactionInfo → reaction configuration
   CGInitializer.initialize() → CGSystem (initial CG topology)

3. Main loop
   ┌─────────────────────────────────────────────────────────────┐
   │ Step 1: LAMMPS fix bond creation                            │
   │   bond/react or bond/create                                 │
   │                                                             │
   │ Step 2: Detect reactions                                    │
   │   LAMMPSDataExtractor → BondDetector → ReactionLocator      │
   │   → CGMapper → CGConverter → write CG trajectory            │
   │   Cache reaction frames → reaction_frames.npz               │
   │                                                             │
   │ Step 3: Relaxation (NVE/limit → NVT → NPT)                  │
   │                                                             │
   │ Step 4: Update (invalidate_cache)                           │
   └─────────────────────────────────────────────────────────────┘

4. Output
   write_cg_trajectory() → cg_trajectory.lammpstrj
   CGCompareList.to_csv() → final_cg_compare_list.csv
   save_reaction_frames() → reaction_frames.npz
```

---

## Changelog

### v2.7 (2026-09-11)

- **New GROMACS → LAMMPS conversion path**
  - `scripts/gmx2lmp_data.py`: convert a GAFF `top` + `gro` pair into a LAMMPS `data` file (real units, `atom_style full`)
  - Multi-molecule expansion with a unified offset basis, bonded coefficient sections, and atomtypes duplication checks
  - `test_gmx2lmp_data.py`: end-to-end cross-validation suite
  - Documentation: `docs/cli-scripts.md#15`

- **New tabulated potential toolchain**
  - `tools/ibm_potential/tabulated_potential.py`: VOTCA/LAMMPS table parsing, type detection, analytic fitting (LJ 12-6, harmonic bond, cosine/harmonic angle) and plotting
  - `scripts/fit_tabulated.py`: fitting CLI, with `well` and `direct` parameterisation strategies for LJ
  - `scripts/mix_cross_tabulated.py`: cross-mix two tables into an unlike-pair potential
  - `scripts/plot_tabulated.py`: table plotting CLI

- **New reactivity-ratio analysis for AA data**
  - `output_analysis/reactivity_ratio.py`: `collect_per_cycle_counts_aa()` reconstructs per-cycle counts from `reaction_frames.npz` plus trajectory; the pre-reaction frame is traced back by `bond_react_check_step`
  - `analyze_reactivity_ratio_aa()` reuses the existing estimator and adds conversion-evolution output and a detailed report
  - New CLI subcommands: `reactivity-ratio`, `reactivity-ratio-aa`
  - `test_reactivity_ratio.py`: test suite covering both CG and AA paths

- **Incremental distribution calculation**
  - `scripts/calc_dist.py`: `--skip-existing` now also supported in `run_pipeline`, matching the two pickle pipelines

- **Correctness fixes**
  - `tools/ibm_potential/{gromacs_loader,pickle_loader}.py`: the dihedral distribution divided `y` by `|bc|` while dividing `x` by `|n1||n2|`, inflating `tan(phi)` by `|n1||n2|` (up to hundreds at CG bond lengths) and collapsing the distribution onto ±90°
  - `scripts/convert_aa2cg.py`: CG `data` files were written with AA atomic masses instead of CG bead masses; harmless for single-bead mappings but ejected light beads and destabilised IBI for multi-bead ones
  - `tools/aa2cg/data_converter.py`: the header declared the *count* of bead types rather than the *maximum type number*, which broke systems with gaps in global type numbering
  - `core/cg_reaction_identifier.py`: template signatures now expand a list-valued interior `bead_type` (e.g. `[1, 2]`) into one signature per candidate

- **Packaging**
  - `pyproject.toml`: added the missing `pandas` requirement, fixed the `readme` and package-discovery paths, added `analysis` / `tools` extras, and corrected `package-data` for `config/mapping/`

- **Documentation**
  - `docs/configuration.md`: recommended production configuration for AA-AM radical copolymerization, validated on a 10480-atom GAFF system

### v2.6 (2026-06-22)

- **New bond/create bond-creation mode**
  - `core/reaction_commands.py`: brand-new module supporting bond/create configuration loading, LAMMPS command generation, and CG mapping updates
  - `BondCreatePair` / `BondCreateConfig` dataclasses encapsulating bond/create type-pair parameters
  - `generate_fix_bond_create()` / `generate_fix_bond_react()` pure-function command generation
  - `update_cg_mapping_create()` CG mapping update based on YAML `type_map`
  - Mutual-exclusion validation of configurations: the two modes cannot be enabled at the same time

- **New CG-level reaction identification module**
  - `core/cg_reaction_identifier.py`: reaction type identification independent of LAMMPS
  - `CGReactionSignature` dataclass encapsulating 3-bead type signatures
  - Two signature loading methods: `load_template_signatures()` / `load_reaction_signatures()`
  - Two-level matching strategy: 3-bead coarse screening + chain fine screening
  - Dual validation: production-level and cross-validation independent verification pipelines
  - Documentation: `docs/bond-modes.md`

- **New post-processing analysis subpackage**
  - `output_analysis/`: chain length distribution, reaction distance, reaction statistics, JS divergence analysis
  - Unified CLI entry point `python -m LmpPy.output_analysis <command>`
  - YAML-driven `PlotConfig` plotting configuration system
  - Schulz-Zimm/Poisson/Log-normal theoretical distribution models

- **Major documentation system refactor**
  - README streamlined into a quick-start navigation hub; detailed content moved to `docs/`
  - New: `docs/bond-modes.md`, `docs/configuration.md` (rewritten)
  - Rewritten: `docs/cg-mapping.md`, `docs/output-files.md`, `docs/cli-scripts.md`
  - New: `docs/modules.md` (complete API reference)
  - Rewritten: `docs/workflow.md` (complete flowcharts + MPI model + restart)
  - New example configurations: `docs/examples/bond_react/`, `docs/examples/bond_create/`
  - README documentation navigation; each topic links to the corresponding `docs/` file

- **Enhanced configuration validation**
  - `scripts/validate_config.py`: brand-new configuration validation script
  - `ConfigValidator` class: file checks, cross-configuration consistency, semantic validation
  - Supports both JSON and text output formats

### v2.5 (2026-06-09)

- **New smoke test system**
  - `core/smoke_validator.py`: pure-function validator with four checks (A/B/C/D)
  - `core/smoke_test_harness.py`: MPI-aware test orchestrator
  - `scripts/test_smoke_run.py`: standalone CLI script
  - New `--smoke-test` argument in `run_refactored.py`

### v2.4 (2026-06-03)

- **Output file streamlining refactor**
  - `reaction_frames.npz` reduced from 11 arrays to 7 arrays
  - Removed `bonds_records/` directory output
  - Removed `changed_bead_id_list.pkl` output
  - Backward compatible with the old npz format

### v2.3 (2026-05-19)

- **Documentation improvements**
  - Added module dependency graph and data flow description
  - Added documentation for all CLI scripts
  - Added CGTopology and ReactionLocator API references

### v2.2 (2026-04-29)

- **New CLI scripts**: `calc_dist.py`, `plot_dist.py`, `yaml2csv_mapping.py`
- **New**: `CGInitializer` / `CGSystem` / `initialize_cg_system` modules
- **smooth_utils expansion**: 11 new submodules

### v2.1 (2026-04-02)

- **New tool modules**: `aa2cg`, `ibm_potential`, `smooth_utils`
- **New CLI scripts**: CLI entry points corresponding to the tool modules
- **New utility functions**: `topology.py`, `units.py`

### v2.0 (2026-03-28)

- **Complete refactor**: multi-system support, configuration-driven design, vectorized/Numba optimizations
- **New**: restart (continuation) feature, `CGCompareList.from_csv()`
- **Modular architecture**: clear core/utils/tools/scripts separation

---

## License

MIT License
