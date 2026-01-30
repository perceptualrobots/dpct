# Tasks: DPCT Core Library

**Feature Branch**: `001-dpct-core-library`  
**Input**: Design documents from `specs/main/` and `specs/001-dpct-core-library/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Tests include unit tests in `*_unittests.ipynb` notebooks following nbdev framework. Individual layers are tested against perceptual functions from the PCT library (https://pypi.org/project/pct/) to verify behavioral equivalence (e.g., linear activation layers vs pct.functions.WeightedSum).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. Each component follows nbdev's three-notebook pattern: implementation, usage examples, and unit tests.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5, US6)
- All tasks include exact file paths

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and nbdev structure

- [X] T001 Verify nbdev installation and project structure per plan.md
- [X] T002 Configure nbdev settings in settings.ini with library metadata
- [X] T003 [P] Setup _quarto.yml for documentation generation
- [X] T004 [P] Create index.ipynb with library overview and getting started guide

**Checkpoint**: nbdev project structure ready for development

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story implementation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Install and verify core dependencies: tensorflow, gymnasium, deap, numpy, optuna, matplotlib, networkx, pct
- [X] T006 [P] Create base configuration utilities for JSON/pickle serialization
- [X] T007 [P] Create base environment wrapper utilities for Gymnasium interface
- [X] T008 [P] Setup DEAP toolbox helper functions for evolutionary operators
- [X] T009 Verify Gymnasium CartPole-v1 environment can be created and run

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Create and Run PCT Hierarchies (Priority: P1) 🎯 MVP

**Goal**: Enable researchers to create hierarchical control systems, compile them into Keras models, and execute them in environments

**Independent Test**: Create DHPCTIndividual with levels [4, 3, 2], compile it, run in CartPole-v1 for 500 steps, verify it produces fitness score and controls environment state. Additionally verify individual layer outputs match corresponding PCT library function outputs for equivalent inputs.

### Implementation for User Story 1

- [X] T010 [P] [US1] Create 00_individual.ipynb with DHPCTIndividual class skeleton in nbs/
- [X] T011 [P] [US1] Create 00_individual_unittests.ipynb in nbs/ with test structure
- [X] T012 [US1] Implement DHPCTIndividual.__init__() with env_name, levels, activation_funcs, weight_types in nbs/00_individual.ipynb
- [X] T013 [US1] Implement DHPCTIndividual.compile() to build Keras Functional API model in nbs/00_individual.ipynb
- [X] T014 [US1] Create perception layers (PL##) with correct inputs per level in nbs/00_individual.ipynb
- [X] T015 [US1] Create reference layers (RL##) with correct inputs per level in nbs/00_individual.ipynb
- [ ] T016 [US1] Create comparator layers (CL##) computing reference minus perception in nbs/00_individual.ipynb
- [ ] T017 [US1] Create output layers (OL##) computing weighted comparator values in nbs/00_individual.ipynb
- [ ] T018 [US1] Connect Level 0 to Observations input and Actions output in nbs/00_individual.ipynb
- [ ] T019 [US1] Connect all comparators to Errors output in nbs/00_individual.ipynb
- [ ] T020 [US1] Implement weight initialization for float, boolean, and ternary types in nbs/00_individual.ipynb
- [ ] T021 [US1] Implement DHPCTIndividual.run() with environment interaction loop in nbs/00_individual.ipynb
- [ ] T022 [US1] Add early_termination support to run() method in nbs/00_individual.ipynb
- [ ] T023 [US1] Add fitness calculation and return in run() method in nbs/00_individual.ipynb
- [ ] T023a [US1] Add support for different fitness calculation methods including pct.errors functions (e.g., RMS, MAE) in nbs/00_individual.ipynb
- [ ] T024 [US1] Add support for obs_connection_level parameter in compile() in nbs/00_individual.ipynb
- [ ] T025 [US1] Add unit tests for __init__() in nbs/00_individual_unittests.ipynb
- [ ] T026 [US1] Add unit tests for compile() and model structure in nbs/00_individual_unittests.ipynb
- [ ] T027 [US1] Add unit tests for run() with CartPole environment in nbs/00_individual_unittests.ipynb
- [ ] T027a [P] [US1] Add unit tests for different fitness calculation methods (cumulative reward, pct.errors.RMS, pct.errors.MAE) in nbs/00_individual_unittests.ipynb
- [ ] T028 [US1] Add unit tests for different weight types in nbs/00_individual_unittests.ipynb
- [ ] T028a [P] [US1] Add PCT library comparison test: linear activation layer vs pct.functions.WeightedSum in nbs/00_individual_unittests.ipynb
- [ ] T028b [P] [US1] Add PCT library comparison test: perception layer (linear activation) vs pct.functions.WeightedSum in nbs/00_individual_unittests.ipynb
- [ ] T028c [P] [US1] Add PCT library comparison test: comparator layer (reference - perception) vs pct.functions.Subtract in nbs/00_individual_unittests.ipynb
- [ ] T028d [P] [US1] Add PCT library comparison test: output layer (element-wise multiplication of inputs with weights) vs pct.functions.WeightedSum in nbs/00_individual_unittests.ipynb
- [ ] T029 [US1] Create 00_individual_usage.ipynb with basic usage examples in nbs/
- [ ] T030 [US1] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: At this point, User Story 1 should be fully functional - researchers can create, compile, and run hierarchies. All DPCT layers have been validated against equivalent PCT library functions to ensure behavioral correctness.

---

## Phase 4: User Story 2 - Save and Load Configurations (Priority: P1) 🎯 MVP

**Goal**: Enable researchers to save hierarchy configurations (structure + weights) and reload them for reproducibility

**Independent Test**: Create individual, save config to JSON, load from JSON, verify both produce identical outputs with same seed

### Implementation for User Story 2

- [ ] T031 [P] [US2] Implement DHPCTIndividual.config() to return complete configuration dictionary in nbs/00_individual.ipynb
- [ ] T032 [P] [US2] Implement DHPCTIndividual.save_config(filepath) to write JSON file in nbs/00_individual.ipynb
- [ ] T033 [US2] Implement DHPCTIndividual.load_config(filepath) class method in nbs/00_individual.ipynb
- [ ] T034 [US2] Implement DHPCTIndividual.from_config(config_dict) class method in nbs/00_individual.ipynb
- [ ] T035 [US2] Add support for pickle serialization in config() method in nbs/00_individual.ipynb
- [ ] T036 [US2] Implement to_legacy_config() method for backward compatibility in nbs/00_individual.ipynb
- [ ] T037 [US2] Implement from_legacy_config() class method in nbs/00_individual.ipynb
- [ ] T038 [US2] Add unit tests for config() and save_config() in nbs/00_individual_unittests.ipynb
- [ ] T039 [US2] Add unit tests for from_config() with deterministic behavior in nbs/00_individual_unittests.ipynb
- [ ] T040 [US2] Add unit tests for legacy config conversion in nbs/00_individual_unittests.ipynb
- [ ] T041 [US2] Add usage examples for save/load workflow in nbs/00_individual_usage.ipynb
- [ ] T042 [US2] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: At this point, User Stories 1 AND 2 work - researchers can persist and reload their hierarchies

---

## Phase 5: User Story 3 - Evolve Hierarchies with Evolutionary Algorithms (Priority: P1) 🎯 MVP

**Goal**: Enable automated discovery of effective hierarchies through evolution with fitness-based selection, crossover, and mutation

**Independent Test**: Initialize DHPCTEvolver with template individual, run 10 generations, verify best fitness improves and outperforms random individuals

### Implementation for User Story 3

- [ ] T043 [P] [US3] Create 01_evolver.ipynb with DHPCTEvolver class skeleton in nbs/
- [ ] T044 [P] [US3] Create 01_evolver_unittests.ipynb in nbs/ with test structure
- [ ] T045 [US3] Implement DHPCTEvolver.__init__() with evolution parameters in nbs/01_evolver.ipynb
- [ ] T046 [US3] Implement setup_evolution() to configure DEAP toolbox in nbs/01_evolver.ipynb
- [ ] T047 [US3] Register individual creation operator in DEAP toolbox in nbs/01_evolver.ipynb
- [ ] T048 [US3] Register population initialization in DEAP toolbox in nbs/01_evolver.ipynb
- [ ] T049 [US3] Register fitness evaluation operator in DEAP toolbox in nbs/01_evolver.ipynb
- [ ] T050 [US3] Register tournament selection operator in DEAP toolbox in nbs/01_evolver.ipynb
- [ ] T051 [US3] Implement run_evolution() main loop in nbs/01_evolver.ipynb
- [ ] T052 [US3] Add fitness evaluation for population in run_evolution() in nbs/01_evolver.ipynb
- [ ] T053 [US3] Add selection, crossover, mutation operations in run_evolution() in nbs/01_evolver.ipynb
- [ ] T054 [US3] Add generation statistics tracking (min/mean/max fitness) in nbs/01_evolver.ipynb
- [ ] T055 [US3] Add hall of fame tracking for best individuals in nbs/01_evolver.ipynb
- [ ] T056 [US3] Implement save_arch_best functionality in nbs/01_evolver.ipynb
- [ ] T057 [US3] Implement save_arch_all functionality in nbs/01_evolver.ipynb
- [ ] T058 [US3] Implement run_best option to display best individual in nbs/01_evolver.ipynb
- [ ] T059 [US3] Add early termination based on fitness_target in nbs/01_evolver.ipynb
- [ ] T060 [US3] Add early termination based on evolve_static_termination in nbs/01_evolver.ipynb
- [ ] T061 [US3] Implement save_results() to persist statistics and configs in nbs/01_evolver.ipynb
- [ ] T062 [US3] Implement get_best_individual() method in nbs/01_evolver.ipynb
- [ ] T063 [US3] Implement get_statistics() method in nbs/01_evolver.ipynb
- [ ] T064 [US3] Implement plot_evolution() for visualizing fitness over generations in nbs/01_evolver.ipynb
- [ ] T065 [US3] Add unit tests for setup_evolution() in nbs/01_evolver_unittests.ipynb
- [ ] T066 [US3] Add unit tests for run_evolution() with small population in nbs/01_evolver_unittests.ipynb
- [ ] T067 [US3] Add unit tests for statistics tracking in nbs/01_evolver_unittests.ipynb
- [ ] T068 [US3] Add unit tests for early termination conditions in nbs/01_evolver_unittests.ipynb
- [ ] T069 [US3] Create 01_evolver_usage.ipynb with evolution examples in nbs/
- [ ] T070 [US3] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: All P1 user stories complete - MVP functionality ready for researchers to evolve hierarchies

---

## Phase 6: User Story 4 - Apply Evolutionary Operators to Individuals (Priority: P2)

**Goal**: Enable manual experimentation with evolutionary operators (mating and mutation)

**Independent Test**: Create two parents, mate them, mutate offspring, verify offspring have characteristics from both parents and mutations differ from pre-mutation state

### Implementation for User Story 4

- [ ] T071 [P] [US4] Implement DHPCTIndividual.mate(other) for crossover in nbs/00_individual.ipynb
- [ ] T072 [P] [US4] Implement uniform crossover for float weights in mate() in nbs/00_individual.ipynb
- [ ] T073 [US4] Implement uniform crossover for boolean/ternary weights in mate() in nbs/00_individual.ipynb
- [ ] T074 [US4] Handle fixed_weights and fixed_levels in mate() in nbs/00_individual.ipynb
- [ ] T075 [US4] Implement DHPCTIndividual.mutate() for weight modification in nbs/00_individual.ipynb
- [ ] T076 [US4] Implement weight_prob parameter in mutate() in nbs/00_individual.ipynb
- [ ] T077 [US4] Implement struct_prob parameter for structural mutations in nbs/00_individual.ipynb
- [ ] T078 [US4] Add support for adding/removing levels in mutate() in nbs/00_individual.ipynb
- [ ] T079 [US4] Add support for adding/removing units in mutate() in nbs/00_individual.ipynb
- [ ] T080 [US4] Respect fixed_weights and fixed_levels in mutate() in nbs/00_individual.ipynb
- [ ] T081 [US4] Implement DHPCTIndividual.evaluate(nevals) for multiple runs in nbs/00_individual.ipynb
- [ ] T082 [US4] Add aggregation methods (mean, max, min, median) to evaluate() in nbs/00_individual.ipynb
- [ ] T083 [US4] Add unit tests for mate() operation in nbs/00_individual_unittests.ipynb
- [ ] T084 [US4] Add unit tests for mutate() with different probabilities in nbs/00_individual_unittests.ipynb
- [ ] T085 [US4] Add unit tests for structural mutations in nbs/00_individual_unittests.ipynb
- [ ] T086 [US4] Add unit tests for evaluate() with multiple runs in nbs/00_individual_unittests.ipynb
- [ ] T087 [US4] Add usage examples for manual evolution in nbs/00_individual_usage.ipynb
- [ ] T088 [US4] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: User Stories 1-4 complete - manual and automated evolution both functional

---

## Phase 7: User Story 5 - Optimize Evolution Hyperparameters (Priority: P2)

**Goal**: Enable automated hyperparameter tuning using Optuna to find best evolutionary algorithm settings

**Independent Test**: Configure DHPCTOptimizer to search pop_size [20-100] and mutation rate [0.05-0.3], run 10 trials, verify best parameters returned and optimization history shows improvement

### Implementation for User Story 5

- [ ] T089 [P] [US5] Create 02_optimizer.ipynb with DHPCTOptimizer class skeleton in nbs/
- [ ] T090 [P] [US5] Create 02_optimizer_unittests.ipynb in nbs/ with test structure
- [ ] T091 [US5] Implement DHPCTOptimizer.__init__() with optimization parameters in nbs/02_optimizer.ipynb
- [ ] T092 [US5] Parse parameters dict with fixed/variable flags in nbs/02_optimizer.ipynb
- [ ] T093 [US5] Implement define_objective() to configure Optuna objective function in nbs/02_optimizer.ipynb
- [ ] T094 [US5] Create objective function that creates and runs DHPCTEvolver in nbs/02_optimizer.ipynb
- [ ] T095 [US5] Add parameter suggestion logic (int, float, categorical) in objective in nbs/02_optimizer.ipynb
- [ ] T096 [US5] Implement run_optimization() to execute Optuna study in nbs/02_optimizer.ipynb
- [ ] T097 [US5] Add pruner support for early stopping trials in nbs/02_optimizer.ipynb
- [ ] T098 [US5] Add sampler configuration (TPE, CMA-ES, Grid, Random) in nbs/02_optimizer.ipynb
- [ ] T099 [US5] Add study persistence with storage parameter in nbs/02_optimizer.ipynb
- [ ] T100 [US5] Implement get_best_params() to retrieve optimal parameters in nbs/02_optimizer.ipynb
- [ ] T101 [US5] Implement get_best_value() to retrieve best fitness in nbs/02_optimizer.ipynb
- [ ] T102 [US5] Implement visualize_results() for parameter importance plots in nbs/02_optimizer.ipynb
- [ ] T103 [US5] Add optimization history visualization in visualize_results() in nbs/02_optimizer.ipynb
- [ ] T104 [US5] Implement save_results() to persist study data in nbs/02_optimizer.ipynb
- [ ] T105 [US5] Add unit tests for parameter parsing in nbs/02_optimizer_unittests.ipynb
- [ ] T106 [US5] Add unit tests for run_optimization() with small trials in nbs/02_optimizer_unittests.ipynb
- [ ] T107 [US5] Add unit tests for pruner functionality in nbs/02_optimizer_unittests.ipynb
- [ ] T108 [US5] Add unit tests for get_best_params() in nbs/02_optimizer_unittests.ipynb
- [ ] T109 [US5] Create 02_optimizer_usage.ipynb with optimization examples in nbs/
- [ ] T110 [US5] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: User Stories 1-5 complete - full evolution and optimization pipeline functional

---

## Phase 8: User Story 6 - Enable Online Learning During Execution (Priority: P3)

**Goal**: Enable evolved individuals to continue learning during environment interaction using gradient-based training

**Independent Test**: Create individual, enable online learning with learning_rate=0.01, run 1000 steps, verify error signals decrease compared to same individual without online learning

### Implementation for User Story 6

- [ ] T111 [US6] Add train parameter support to DHPCTIndividual.run() in nbs/00_individual.ipynb
- [ ] T112 [US6] Add learning_rate, optimizer, train_every_n_steps parameters to run() in nbs/00_individual.ipynb
- [ ] T113 [US6] Implement model.compile() with optimizer when train=True in nbs/00_individual.ipynb
- [ ] T114 [US6] Add error signal collection during execution in nbs/00_individual.ipynb
- [ ] T115 [US6] Implement gradient descent updates every N steps in nbs/00_individual.ipynb
- [ ] T116 [US6] Add error_weight_coefficients for per-level error weighting in nbs/00_individual.ipynb
- [ ] T117 [US6] Create training targets with zero error goal in nbs/00_individual.ipynb
- [ ] T118 [US6] Add unit tests for online learning in nbs/00_individual_unittests.ipynb
- [ ] T119 [US6] Add unit tests for error signal tracking in nbs/00_individual_unittests.ipynb
- [ ] T120 [US6] Add unit tests verifying error reduction with training in nbs/00_individual_unittests.ipynb
- [ ] T121 [US6] Add usage examples for online learning in nbs/00_individual_usage.ipynb
- [ ] T122 [US6] Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: All user stories complete - advanced online learning capability available

---

## Phase 9: Visualization and Advanced Features

**Purpose**: Cross-cutting visualization and advanced capabilities

- [ ] T123 [P] Create 03_visualization.ipynb with visualization functions in nbs/
- [ ] T124 [P] Create 03_visualization_unittests.ipynb in nbs/ with test structure
- [ ] T125 [P] Implement visualize_hierarchy_layers() with networkx in nbs/03_visualization.ipynb
- [ ] T126 [P] Implement visualize_pct_units() for control unit view in nbs/03_visualization.ipynb
- [ ] T127 [P] Implement visualize_weighted_network() with weight values in nbs/03_visualization.ipynb
- [ ] T128 [P] Implement visualize_execution_history() for time-series data in nbs/03_visualization.ipynb
- [ ] T129 [P] Implement visualize_layer_activations() for layer values over time in nbs/03_visualization.ipynb
- [ ] T130 [US1] Add record_history parameter to DHPCTIndividual.run() in nbs/00_individual.ipynb
- [ ] T131 [US1] Implement ExecutionHistory recording in run() in nbs/00_individual.ipynb
- [ ] T132 [US3] Add parallel fitness evaluation support in DHPCTEvolver in nbs/01_evolver.ipynb
- [ ] T133 [US3] Implement multiprocessing.Pool for parallel evaluation in nbs/01_evolver.ipynb
- [ ] T134 [US3] Add comet_ml logging support in DHPCTEvolver in nbs/01_evolver.ipynb
- [ ] T135 [US3] Add random_structure initialization support in nbs/01_evolver.ipynb
- [ ] T136 [US3] Add initial_individuals parameter for pre-trained individuals in nbs/01_evolver.ipynb
- [ ] T137 Add unit tests for visualization functions in nbs/03_visualization_unittests.ipynb
- [ ] T138 Create 03_visualization_usage.ipynb with visualization examples in nbs/
- [ ] T139 Run nbdev_prepare to export code and verify tests pass

**Checkpoint**: All advanced features and visualizations complete

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple components

- [ ] T140 [P] Update index.ipynb with comprehensive getting started guide in nbs/
- [ ] T141 [P] Add environment compatibility testing (CartPole, LunarLander) in nbs/00_individual_unittests.ipynb
- [ ] T142 [P] Verify deterministic behavior with random seeds across all components
- [ ] T143 [P] Add error handling and validation for invalid inputs
- [ ] T144 [P] Optimize model compilation performance for large hierarchies
- [ ] T145 [P] Add comprehensive docstrings following nbdev conventions
- [ ] T146 Run full nbdev_prepare and verify all notebooks execute
- [ ] T147 Run pytest and verify >90% code coverage
- [ ] T148 Validate quickstart.md examples work as documented
- [ ] T149 Build documentation with nbdev_docs and verify all pages render
- [ ] T150 Final smoke test with evolution scenario from quickstart.md

**Checkpoint**: Library complete, tested, and documented - ready for release

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - can start after Phase 2
- **User Story 2 (Phase 4)**: Depends on User Story 1 - requires DHPCTIndividual implementation
- **User Story 3 (Phase 5)**: Depends on User Stories 1 & 2 - requires DHPCTIndividual with config support
- **User Story 4 (Phase 6)**: Depends on User Story 1 - extends DHPCTIndividual with operators
- **User Story 5 (Phase 7)**: Depends on User Story 3 - requires DHPCTEvolver implementation
- **User Story 6 (Phase 8)**: Depends on User Story 1 - extends DHPCTIndividual with training
- **Visualization (Phase 9)**: Depends on User Stories 1 & 3 - visualizes individuals and evolution
- **Polish (Phase 10)**: Depends on all desired user stories being complete

### User Story Dependencies

```
Phase 2: Foundational ━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                         ┃                                  ┃
                         ┃                                  ┃
                         ┣━━━━► US1 (Phase 3) ━━━━━━━━━━━━┳┫
                         ┃         │                       ┃┃
                         ┃         │                       ┃┃
                         ┃         └━━► US2 (Phase 4) ━━━┳┃┃
                         ┃                │               ┃┃┃
                         ┃                │               ┃┃┃
                         ┃                └━━► US3 (Phase 5)┫┃
                         ┃                       │        ┃┃┃
                         ┃                       │        ┃┃┃
                         ┗━━━━► US4 (Phase 6) ◄──┘        ┃┃┃
                                    │                     ┃┃┃
                                    │                     ┃┃┃
                         US5 (Phase 7) ◄──────────────────┘┃┃
                                                            ┃┃
                         US6 (Phase 8) ◄────────────────────┘┃
                                                              ┃
                         Visualization (Phase 9) ◄────────────┛
```

### Critical Path (MVP)

For minimum viable product (MVP), complete in this order:

1. **Phase 1**: Setup (T001-T004)
2. **Phase 2**: Foundational (T005-T009) ← BLOCKS ALL STORIES
3. **Phase 3**: User Story 1 (T010-T030) ← Create and run hierarchies
4. **Phase 4**: User Story 2 (T031-T042) ← Save and load configs
5. **Phase 5**: User Story 3 (T043-T070) ← Evolution

**STOP HERE FOR MVP** - This gives researchers the core value proposition: create hierarchies, save them, and evolve them to discover better controllers.

### Parallel Opportunities

#### Within Phase 1 (Setup)
- T003 and T004 can run in parallel (different files)

#### Within Phase 2 (Foundational)
- T006, T007, T008 can run in parallel (different utilities)

#### Within Phase 3 (User Story 1)
- T010 and T011 can start in parallel (different notebooks)
- T025, T026, T027, T028 (tests) can run in parallel once implementation done
- T029 can be developed in parallel with tests

#### Within Phase 4 (User Story 2)
- T031 and T032 can run in parallel (different methods)
- T038, T039, T040 (tests) can run in parallel

#### Within Phase 5 (User Story 3)
- T043 and T044 can start in parallel (different notebooks)
- T065, T066, T067, T068 (tests) can run in parallel once evolver implemented

#### Within Phase 6 (User Story 4)
- T071 and T075 can run in parallel (different methods: mate vs mutate)
- T083, T084, T085, T086 (tests) can run in parallel

#### Within Phase 7 (User Story 5)
- T089 and T090 can start in parallel (different notebooks)
- T105, T106, T107, T108 (tests) can run in parallel

#### Within Phase 9 (Visualization)
- T123, T124, T125, T126, T127, T128, T129 can all start in parallel (different functions)

#### Within Phase 10 (Polish)
- T140, T141, T142, T143, T144, T145 can run in parallel (different concerns)

### Between Phases (with adequate staffing)

Once **Phase 2 (Foundational)** completes:
- Developer A: Phase 3 (User Story 1)
- Developer B: Phase 6 (User Story 4) - only depends on US1, not US2/US3

Once **Phase 4 (US2)** completes:
- Developer C: Phase 5 (User Story 3)

Once **Phase 3 (US1)** completes:
- Developer D: Phase 8 (User Story 6)

---

## Implementation Strategy

### MVP First (Recommended for Solo Developer)

**Goal**: Get core value proposition working ASAP

1. ✅ Complete **Phase 1: Setup** (4 tasks)
2. ✅ Complete **Phase 2: Foundational** (5 tasks) ← CRITICAL BLOCKER
3. ✅ Complete **Phase 3: User Story 1** (27 tasks) ← Create/run hierarchies + PCT library validation + fitness methods
4. ✅ Complete **Phase 4: User Story 2** (12 tasks) ← Save/load configs
5. ✅ Complete **Phase 5: User Story 3** (28 tasks) ← Evolution
6. **STOP and VALIDATE**: Test MVP independently
7. **Demo/Document**: Show researchers how to evolve hierarchies
8. **Continue if successful**: Add User Stories 4, 5, 6 as enhancements

**Total MVP Tasks**: 76 tasks (T001-T070, includes 4 PCT library comparison tests + 2 fitness calculation tasks)

### Incremental Delivery

Each user story adds independent value:

1. **Setup + Foundational** → Infrastructure ready
2. **+ User Story 1** → Can create and test hierarchies manually
3. **+ User Story 2** → Can save/share successful hierarchies
4. **+ User Story 3** → Can automatically discover good hierarchies (MVP complete!)
5. **+ User Story 4** → Can manually experiment with operators
6. **+ User Story 5** → Can automatically tune hyperparameters
7. **+ User Story 6** → Can enable online learning for fine-tuning
8. **+ Visualization** → Can visualize networks and results
9. **+ Polish** → Production-ready library

### Parallel Team Strategy

With 3 developers after Foundational phase:

- **Developer A**: Owns DHPCTIndividual (US1, US2, US4, US6)
- **Developer B**: Owns DHPCTEvolver (US3)
- **Developer C**: Owns DHPCTOptimizer (US5) and Visualization (Phase 9)

**Coordination points**:
- All wait for Phase 2 completion
- Dev B needs US1 & US2 done before starting US3
- Dev C needs US3 done before starting US5
- Daily syncs to coordinate integration

---

## Validation & Testing

### After Each Phase

- Run `nbdev_prepare` to export code
- Run `pytest` to execute all unit tests
- Verify notebook execution in Jupyter
- Check code coverage with `pytest --cov`

### Independent Story Tests

**User Story 1**: Create DHPCTIndividual([4,3,2]), compile(), run(500 steps), verify fitness returned

**User Story 2**: Save config → load config → run both, verify identical behavior with same seed

**User Story 3**: Run evolution for 10 generations, verify best fitness in gen 10 > gen 1

**User Story 4**: parent1.mate(parent2) → 2 offspring, mutate(0.2) → verify changes

**User Story 5**: Run optimizer with 10 trials, verify best_params returned and fitness improved

**User Story 6**: run(train=True) → verify error signals decrease over 1000 steps

### Success Criteria Verification

- **SC-001**: Create 3-level hierarchy, run CartPole 500 steps in <30s setup time ✓
- **SC-002**: Config save/load produces identical behavior with same seed ✓
- **SC-003**: Evolve 50 individuals × 100 generations in <10 minutes ✓
- **SC-004**: Generation 100 best fitness ≥ 50% better than generation 1 ✓
- **SC-005**: Config files <100KB for 3-5 levels, 2-10 units/level ✓
- **SC-006**: Optimizer with 20 trials finds 20% better params than default ✓
- **SC-007**: Unit tests achieve >90% code coverage ✓
- **SC-008**: All notebooks execute without errors in nbdev_prepare ✓
- **SC-009**: Documentation includes working examples for each feature ✓
- **SC-010**: Library works with CartPole-v1 and LunarLanderContinuous-v2 ✓

---

## Notes

- All development in Jupyter notebooks following nbdev literate programming
- Three-notebook pattern per component: implementation, usage, unittests
- Run `nbdev_prepare` after each modification to export and test
- Tests are written in `*_unittests.ipynb` notebooks (not separate pytest files)
- **PCT library comparison**: Individual layer behaviors are validated against equivalent functions from the PCT library (https://pypi.org/project/pct/) to ensure correctness (e.g., linear layers vs pct.functions.WeightedSum, comparators vs pct.functions.Subtract)
- **Fitness calculation**: Support multiple fitness metrics including cumulative reward (default) and error functions from pct.errors module (e.g., RMS, MAE)
- Install PCT library with: `pip install pct`
- [P] tasks are in different files and can run in parallel
- [Story] label maps each task to its user story for traceability
- Commit after each task or logical group
- File paths use `nbs/` for notebooks, `dpct/` for auto-generated Python modules
- Each user story checkpoint enables independent validation
- MVP = first 3 user stories (create, save, evolve)
