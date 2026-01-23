# Implementation Plan: DPCT Core Library

**Branch**: `001-dpct-core-library` | **Date**: 2026-01-20 | **Spec**: [spec.md](../001-dpct-core-library/spec.md)
**Input**: Feature specification from `/specs/001-dpct-core-library/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement Deep Perceptual Control Theory (DPCT) library enabling researchers to create hierarchical control systems, evolve them through evolutionary algorithms, and optimize hyperparameters. The library provides three main components: `DHPCTIndividual` for creating and running PCT hierarchies using Keras models, `DHPCTEvolver` for evolutionary optimization using DEAP, and `DHPCTOptimizer` for hyperparameter tuning using Optuna. All development follows nbdev literate programming workflow with Jupyter notebooks as the primary development medium.

## Technical Context

**Language/Version**: Python 3.8+  
**Primary Dependencies**: nbdev (MANDATORY), tensorflow (Keras), gymnasium, deap, numpy, optuna, matplotlib, networkx, comet_ml (optional), pickle  
**Storage**: JSON for configurations, pickle for serialization, optional comet_ml for experiment tracking  
**Testing**: pytest (via nbdev test framework), unit tests in `*_unittests.ipynb` notebooks, >90% code coverage target  
**Target Platform**: Cross-platform (Windows, Linux, macOS) - CPU sufficient, GPU optional for acceleration  
**Project Type**: Single Python library developed with nbdev literate programming  
**Performance Goals**: Evolution of 50 individuals over 100 generations in <10 minutes (CartPole), parallelization support for fitness evaluations  
**Constraints**: <200KB configuration files for typical hierarchies (3-5 levels, 2-10 units/level), deterministic behavior via random seeds, environment execution <30 seconds setup time  
**Scale/Scope**: Support 2-10 hierarchy levels, 2-20 units per level, population sizes 20-100, compatible with Gymnasium and similar interfaces

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Nbdev-First Development (NON-NEGOTIABLE)
**Status**: ✅ COMPLIANT  
**Verification**: All code will be developed in Jupyter notebooks in `nbs/` directory. Three-notebook pattern for each component: implementation (`NN_packagename.ipynb`), usage examples (`NN_packagename_usage.ipynb`), unit tests (`NN_packagename_unittests.ipynb`). `nbdev_prepare` will be run after each modification.

### II. Test-Driven Development (NON-NEGOTIABLE)
**Status**: ✅ COMPLIANT  
**Verification**: Unit tests will be written in `*_unittests.ipynb` notebooks before implementation. Tests will cover all functional requirements (FR-001 through FR-060) with deterministic behavior using fixed random seeds. Integration tests will verify environment compatibility, model creation, config save/load, and evolutionary operators.

### III. Configuration-Driven Architecture
**Status**: ✅ COMPLIANT  
**Verification**: DHPCTIndividual implements `config()`, `save_config()`, `from_config()`, `to_legacy_config()`, and `from_legacy_config()` per FR-005, FR-006, FR-007, FR-011. Configurations are pickleable and include all environment, hierarchy, and weight data. Loading saved configs produces identical behavior (SC-002).

### IV. Hierarchical Control Purity
**Status**: ✅ COMPLIANT  
**Verification**: Each level has perception, reference, comparator (reference - perception), and output layers per FR-014 through FR-023. Flexible perception-observation connections per FR-015. Layer naming follows strict convention: PL##, RL##, CL##, OL##, Observations, Actions, Errors. Comparator values exposed as model outputs.

### V. Evolution & Optimization Transparency
**Status**: ✅ COMPLIANT  
**Verification**: Generation statistics track min/mean/max fitness, mutation rates, elapsed time per FR-035. Optional comet_ml logging (FR-042). Execution history recording and visualization (FR-012, FR-013). Three network visualization views (FR-024/25/26). Parallelization support (FR-037). Random structure initialization, pre-trained individuals, and fixed weights support (FR-043/44/45). Optuna integration with parameter importance and history visualization (FR-050, FR-051).

**Overall Assessment**: All constitution principles are satisfied by the feature specification. No violations require justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-dpct-core-library/
├── spec.md              # Feature specification (existing)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── dhpct_individual.md
│   ├── dhpct_evolver.md
│   └── dhpct_optimizer.md
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
nbs/                     # Jupyter notebooks (primary development)
├── _quarto.yml          # Documentation configuration
├── index.ipynb          # Library overview and getting started
├── 00_individual.ipynb        # DHPCTIndividual implementation
├── 00_individual_usage.ipynb  # DHPCTIndividual usage examples
├── 00_individual_unittests.ipynb  # DHPCTIndividual unit tests
├── 01_evolver.ipynb     # DHPCTEvolver implementation
├── 01_evolver_usage.ipynb   # DHPCTEvolver usage examples
├── 01_evolver_unittests.ipynb  # DHPCTEvolver unit tests
├── 02_optimizer.ipynb   # DHPCTOptimizer implementation
├── 02_optimizer_usage.ipynb    # DHPCTOptimizer usage examples
├── 02_optimizer_unittests.ipynb  # DHPCTOptimizer unit tests
├── 03_visualization.ipynb  # Visualization functions
├── 03_visualization_usage.ipynb
├── 03_visualization_unittests.ipynb
└── styles.css

dpct/                    # Auto-generated Python modules (via nbdev)
├── __init__.py
├── individual.py        # Exported from 00_individual.ipynb
├── evolver.py           # Exported from 01_evolver.ipynb
├── optimizer.py         # Exported from 02_optimizer.ipynb
└── visualization.py     # Exported from 03_visualization.ipynb

_proc/                   # nbdev processing directory
└── _docs/               # Generated documentation

tests/                   # Auto-generated from *_unittests.ipynb notebooks
```

**Structure Decision**: Selected single project structure (Option 1) as this is a Python library with no frontend/backend separation or mobile components. Development follows nbdev's literate programming model where all code originates in Jupyter notebooks in `nbs/` directory and is exported to Python modules in `dpct/`. Each major component (Individual, Evolver, Optimizer, Visualization) has three notebooks: implementation, usage examples, and unit tests. Tests are automatically extracted from `*_unittests.ipynb` notebooks by nbdev.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations identified. All requirements align with constitutional principles.

## Phase 0: Outline & Research

**Status**: ✅ COMPLETED

**Output**: [research.md](research.md)

**Key Decisions**:
1. **Keras Functional API** for hierarchical models - supports multi-input/output and explicit layer connections
2. **DEAP** for evolutionary algorithms - mature, flexible, supports parallelization
3. **Optuna** for hyperparameter optimization - modern API, built-in pruning and visualization
4. **Gymnasium** environment interface - actively maintained successor to OpenAI Gym
5. **multiprocessing.Pool** for parallelization - simple, effective for CPU-bound fitness evaluations
6. **NetworkX + Matplotlib** for visualization - graph structures for networks, time-series for history
7. **JSON primary, pickle secondary** for configuration - human-readable vs Python-native
8. **Adapter pattern** for legacy config format conversion
9. **NumPy with constraints** for different weight types (float, boolean, ternary)
10. **Boolean masks/sets** for fixed weights/nodes/levels

All technical unknowns from specification have been resolved with concrete implementation approaches.

## Phase 1: Design & Contracts

**Status**: ✅ COMPLETED

**Outputs**:
- [data-model.md](data-model.md) - Entity definitions and relationships
- [contracts/dhpct_individual.md](contracts/dhpct_individual.md) - DHPCTIndividual API
- [contracts/dhpct_evolver.md](contracts/dhpct_evolver.md) - DHPCTEvolver API
- [contracts/dhpct_optimizer.md](contracts/dhpct_optimizer.md) - DHPCTOptimizer API
- [contracts/visualization.md](contracts/visualization.md) - Visualization functions
- [quickstart.md](quickstart.md) - Usage examples and patterns

**Key Entities Defined**:
1. **DHPCTIndividual** - Hierarchical control system with Keras model
2. **DHPCTEvolver** - Population-based evolutionary optimizer
3. **DHPCTOptimizer** - Hyperparameter search using Optuna
4. **HierarchyConfiguration** - Serializable configuration dictionary
5. **GenerationStatistics** - Evolution metrics per generation
6. **PCTControlUnit** - Conceptual PCT control loop (implemented as layer groups)
7. **ExecutionHistory** - Recorded state transitions during execution

**API Contracts** cover all functional requirements (FR-001 through FR-060) with:
- Constructor signatures and parameters
- Method contracts with inputs, outputs, side effects, and exceptions
- Property definitions
- Usage examples
- Validation rules

**Agent Context Updated**: GitHub Copilot instructions file created with project technologies

## Constitution Check (Re-evaluation)

*Post-design validation of constitution compliance*

### I. Nbdev-First Development (NON-NEGOTIABLE)
**Status**: ✅ COMPLIANT  
**Verification**: Project structure defines nbs/ directory with three-notebook pattern per component (implementation, usage, unittests). All development happens in Jupyter notebooks with nbdev export directives. No direct Python module editing.

### II. Test-Driven Development (NON-NEGOTIABLE)
**Status**: ✅ COMPLIANT  
**Verification**: Unit test notebooks (*_unittests.ipynb) specified for each component. Contracts define testable behaviors for all methods. Integration tests planned for environment compatibility, config round-trips, and evolutionary operators.

### III. Configuration-Driven Architecture
**Status**: ✅ COMPLIANT  
**Verification**: HierarchyConfiguration entity fully specified in data-model.md. DHPCTIndividual API contract includes config(), save_config(), from_config(), to_legacy_config(), from_legacy_config() methods. Configurations are pickleable and JSON-serializable. Round-trip save/load preserves behavior.

### IV. Hierarchical Control Purity
**Status**: ✅ COMPLIANT  
**Verification**: PCTControlUnit conceptual entity defined with perception, reference, comparator (reference - perception), output components. Data model specifies flexible perception-observation connections per FR-015. Layer naming convention strictly defined (PL##, RL##, CL##, OL##). Comparator values exposed as model outputs per FR-022.

### V. Evolution & Optimization Transparency
**Status**: ✅ COMPLIANT  
**Verification**: GenerationStatistics entity captures comprehensive metrics. DHPCTEvolver API includes comet_ml integration, execution history recording, three network visualization modes (layer, PCT unit, weighted), parallelization support, random structure initialization, pre-trained individual support, and fixed weights/levels. Optuna integration provides parameter importance and optimization history visualization.

**Overall Re-assessment**: All constitution principles remain satisfied after detailed design phase. No violations introduced. Implementation can proceed.

## Phase 2: Implementation Planning

**Status**: ⏸️ DEFERRED (per workflow - use /speckit.tasks command)

Phase 2 (implementation task breakdown) is handled by the `/speckit.tasks` command, not `/speckit.plan`. This plan provides complete research and design foundation for task generation.

## Summary

**Branch**: `001-dpct-core-library`  
**Plan Location**: `C:\Users\ruper\Versioning\python\nbdev\dpct\specs\main\plan.md`

**Deliverables**:
- ✅ Technical Context filled
- ✅ Constitution Check passed (pre and post-design)
- ✅ Project Structure defined
- ✅ Phase 0: research.md created (12 technology decisions)
- ✅ Phase 1: data-model.md created (7 entities)
- ✅ Phase 1: contracts/ created (4 API contracts)
- ✅ Phase 1: quickstart.md created (10 examples)
- ✅ Phase 1: Agent context updated (GitHub Copilot instructions)

**Next Steps**:
1. Run `/speckit.tasks` command to generate Phase 2 implementation tasks
2. Begin nbdev development with 00_individual.ipynb for DHPCTIndividual
3. Follow TDD workflow: write tests in 00_individual_unittests.ipynb first
4. Implement remaining components following three-notebook pattern
5. Run nbdev_prepare after each notebook modification

**Key Technologies**:
- Python 3.8+, nbdev, TensorFlow/Keras, Gymnasium, DEAP, Optuna, NumPy, Matplotlib, NetworkX
- Optional: comet_ml for experiment tracking

**Performance Targets**:
- Evolution: 50 individuals × 100 generations < 10 minutes (CartPole)
- Configurations: < 200KB JSON files
- Coverage: > 90% code coverage across all classes
