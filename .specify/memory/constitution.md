<!--
SYNC IMPACT REPORT
==================
Version Change: N/A → 1.0.0
Rationale: Initial constitution creation for DPCT library

Modified Principles:
- All principles newly created from PROJECT_REQUIREMENTS.md

Added Sections:
- Core Principles (5 principles)
- Development Framework Requirements
- Quality Assurance Requirements
- Governance

Templates Requiring Updates:
✅ plan-template.md - Constitution Check section exists (compatible)
✅ spec-template.md - User stories and requirements sections exist (compatible)
✅ tasks-template.md - Phase organization exists (compatible)
⚠ No commands directory found - no command file updates needed

Follow-up TODOs:
- None - All placeholders filled with concrete values
-->

# DPCT Library Constitution

## Core Principles

### I. Nbdev-First Development (NON-NEGOTIABLE)

All code MUST be developed using the nbdev framework (https://nbdev.fast.ai). Development happens exclusively in Jupyter notebooks located in the `nbs/` directory, with code automatically exported to Python modules via nbdev directives.

**Rationale**: nbdev enables literate programming, combining documentation, implementation, and testing in a single maintainable format, ensuring documentation never drifts from implementation.

**Requirements**:
- All development notebooks MUST be in `nbs/` directory
- Notebooks MUST follow naming pattern: `NN_packagename.ipynb` (two-digit prefix)
- Each package MUST have three notebooks:
  1. `NN_packagename.ipynb` - Main implementation with `#export` directives
  2. `NN_packagename_usage.ipynb` - Usage examples with `#gui` directive in each code cell
  3. `NN_packagename_unittests.ipynb` - Comprehensive unit tests
- `nbdev_prepare` MUST be run after each modification and all errors fixed before proceeding
- No direct editing of generated Python modules - all changes via notebooks

### II. Test-Driven Development (NON-NEGOTIABLE)

Tests MUST be written before implementation. The Red-Green-Refactor cycle is strictly enforced: write tests → tests fail → implement → tests pass → refactor.

**Rationale**: TDD ensures code correctness, prevents regressions, and serves as executable documentation of expected behavior.

**Requirements**:
- Unit tests in `*_unittests.ipynb` notebooks MUST exist before implementation
- Tests MUST be comprehensive and cover edge cases
- Tests MUST use deterministic behavior with fixed random seeds where applicable
- All tests MUST pass before code is considered complete
- Integration tests MUST verify:
  - Environment integration (e.g., CartPole, Lunar Lander)
  - Model creation and execution
  - Configuration save/load round-trips
  - Evolutionary operators (mate, mutate, evaluate)

### III. Configuration-Driven Architecture

All hierarchical control systems MUST be fully representable as configurations that can be saved, loaded, serialized, and version-controlled independently of code.

**Rationale**: Configuration-driven design enables reproducibility, experimentation tracking, sharing of evolved individuals, deployment flexibility, and interoperability with legacy systems.

**Requirements**:
- Every `DHPCTIndividual` MUST implement:
  - `config()` method returning complete configuration dictionary
  - `save_config(filepath)` method persisting to JSON
  - `from_config(config_dict)` class method for instantiation
  - `to_legacy_config()` and `from_legacy_config()` methods for legacy format conversion
- Configuration dictionaries MUST be pickleable for serialization beyond JSON
- Configuration MUST include:
  - Environment specification (`env_name`, `properties`)
  - Hierarchy structure (`levels`, `activation_funcs`, `weight_types`)
  - All weights and biases
  - Input references
- Configurations MUST be human-readable and editable
- Loading a saved configuration MUST produce identical behavior

### IV. Hierarchical Control Purity

PCT (Perceptual Control Theory) hierarchy structure MUST be preserved with clear separation of perception, reference, comparator, and output layers at each level.

**Rationale**: Maintaining PCT architectural principles ensures the library remains true to control theory foundations and enables meaningful hierarchical learning.

**Requirements**:
- Each control unit MUST have: reference function, perception function, comparator (reference - perception), output function
- Lowest level perceptions MUST connect to environment observations, unless another level is explicitly specified for connection to observations
- Highest level references MUST connect to external reference inputs
- Comparator values (errors) across all levels MUST be exposed as model outputs
- Actions from lowest level MUST be applied to environment
- Model structure MUST follow naming convention:
  - Perception layers: `PL##` (e.g., `PL00`, `PL01`)
  - Reference layers: `RL##`
  - Comparator layers: `CL##`
  - Output layers: `OL##`
  - Input layer: `Observations`
  - Action outputs: `Actions`
  - Error outputs: `Errors`

### V. Evolution & Optimization Transparency

Evolutionary and optimization processes MUST provide comprehensive visibility through logging, statistics, visualization, and configuration tracking.

**Rationale**: Transparency enables debugging, reproducibility, hyperparameter tuning, understanding of evolutionary dynamics, and publication-quality analysis.

**Requirements**:
- Generation statistics MUST include: min/mean/max fitness, mutation percentages, elapsed time
- Option to save best individual configuration each generation
- Option to save all individual configurations for full population tracking
- Early termination based on stagnation detection (unchanged generations)
- Optional comet_ml integration for experiment logging and tracking
- Evolution MUST support parallelization of fitness evaluations
- Evolution MUST support:
  - Random structure initialization (random level counts and units per level within limits)
  - Initialization with existing or pre-trained individuals
  - Marking specific weights, nodes, or levels as FIXED to prevent modification
- Execution history MUST be optionally recordable:
  - All observations, layer values, and actions over time
  - Graphical visualization of execution history
- Network visualization MUST support three views:
  - Layer view: all layers, nodes, and connections
  - PCT unit view: control units as single nodes
  - Weighted view: connections showing weight values
- Optuna optimization MUST support:
  - Fixed vs. variable parameter specification
  - Parameter importance visualization
  - Optimization history tracking
  - Study persistence to storage
- All random processes MUST support seeding for reproducibility

## Development Framework Requirements

### Mandatory Dependencies

Projects MUST include:
- `nbdev` - Literate programming framework (MANDATORY)

### Suggested Dependencies

Projects SHOULD include these dependencies for full functionality:
- `tensorflow` - Keras model implementation
- `gymnasium` - Environment interface (successor to OpenAI Gym)
- `deap` - Evolutionary algorithms
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `optuna` - Hyperparameter optimization
- `networkx` - Network visualization
- `comet_ml` - Optional experiment logging
- `pickle` - Configuration serialization (Python standard library)

**Note**: The library MUST support environments from Gymnasium or any environments following the same interface pattern.

### Development Workflow

The following workflow MUST be followed:

1. Create notebook files following naming convention
2. Develop code using nbdev directives (`#export`, `#hide`, etc.)
3. Run `nbdev_prepare` to validate and export code
4. Fix any errors reported by nbdev_prepare
5. Use `nbdev_build_lib` to build the library
6. Use `nbdev_build_docs` to generate documentation
7. All steps MUST complete without errors before proceeding

### Validation Requirements

- Run `nbdev_prepare` after creating or modifying each notebook
- Fix any reported errors before proceeding
- Ensure all exported functions have complete docstrings
- Verify documentation builds successfully

## Quality Assurance Requirements

### Documentation Standards

- All exported functions, classes, and methods MUST have docstrings
- Usage examples in `*_usage.ipynb` MUST be clear and executable
- Visualizations of hierarchy structure MUST be provided
- Evolution process demonstrations MUST use simple, understandable environments

### Code Quality

- Default activation function: linear (unless specified)
- Default weight type: float (support boolean and ternary)
- Keras Functional API MUST be used for model creation
- Model layers MUST follow strict naming conventions (see Principle IV)
- No breaking changes without major version bump

### Performance Validation

- Individuals MUST successfully run standard environments (CartPole, LunarLanderContinuous)
- Evaluation MUST support multiple runs with configurable aggregation (average/maximum)
- Early termination refers to the environment returning Done=True (not a fitness target)
- Online learning MUST be optional and configurable

## Governance

This constitution supersedes all other development practices for the DPCT library. Any code, architecture, or process decisions MUST align with these principles.

### Amendment Process

1. Proposed changes MUST be documented with rationale
2. Impact on existing code and templates MUST be assessed
3. Version number MUST be updated following semantic versioning:
   - **MAJOR**: Backward-incompatible governance/principle removals or redefinitions
   - **MINOR**: New principle/section added or materially expanded guidance
   - **PATCH**: Clarifications, wording, typo fixes, non-semantic refinements
4. All dependent artifacts MUST be updated for consistency

### Compliance

- All development work MUST verify alignment with this constitution
- Violations MUST be justified and documented in project artifacts
- Constitution checks in plan templates MUST be completed before implementation phases

**Version**: 1.0.0 | **Ratified**: 2026-01-15 | **Last Amended**: 2026-01-15
