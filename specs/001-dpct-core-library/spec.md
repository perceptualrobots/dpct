# Feature Specification: DPCT Core Library

**Feature Branch**: `001-dpct-core-library`  
**Created**: 2026-01-15  
**Status**: Draft  
**Input**: User description: "Implement Deep Perceptual Control Theory (DPCT) library with hierarchical control systems, evolutionary algorithms, and optimization capabilities"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create and Run PCT Hierarchies (Priority: P1)

A researcher wants to create a hierarchical control system that can control a simulated environment (e.g., CartPole). They need to define the hierarchy structure, compile it into a runnable model, execute it in the environment, and observe its behavior.

**Why this priority**: This is the core value proposition of the library - enabling users to create and test PCT control hierarchies. Without this, no other functionality matters.

**Independent Test**: Can be fully tested by creating a DHPCTIndividual with a specific hierarchy configuration, running it in CartPole environment for 500 steps, and verifying it produces control actions that affect the environment state.

**Acceptance Scenarios**:

1. **Given** environment name "CartPole-v1" and hierarchy levels [4, 3, 2], **When** user creates DHPCTIndividual and calls compile(), **Then** a Keras model is created with correct layer structure matching PCT principles
2. **Given** a compiled DHPCTIndividual, **When** user calls run(steps=500), **Then** the individual interacts with the environment for 500 timesteps and returns fitness score
3. **Given** a DHPCTIndividual with hierarchy levels [4, 3, 2], **When** examining the model structure, **Then** perception layers receive inputs from lower levels, reference layers from upper level outputs, and comparators compute reference minus perception
4. **Given** a running individual, **When** observing model outputs, **Then** both action values and error signals (comparators) are accessible

---

### User Story 2 - Save and Load Configurations (Priority: P1)

A researcher discovers a well-performing hierarchy through experimentation and needs to save its complete configuration (structure + weights) to a file, then later reload it to continue testing or share with colleagues.

**Why this priority**: Configuration persistence is essential for reproducibility, experimentation tracking, and collaboration. Users need this immediately after creating their first working hierarchy.

**Independent Test**: Create an individual, save its configuration to JSON, create a new individual from that JSON, and verify both produce identical outputs when run in the same environment with the same random seed.

**Acceptance Scenarios**:

1. **Given** a DHPCTIndividual with trained weights, **When** user calls config(), **Then** a complete dictionary is returned containing environment settings, hierarchy structure, and all weight values
2. **Given** a DHPCTIndividual, **When** user calls save_config("individual.json"), **Then** a JSON file is created with all configuration data
3. **Given** a saved configuration file "individual.json", **When** user calls DHPCTIndividual.from_config(config_dict), **Then** a new individual is instantiated with identical structure and weights
4. **Given** two individuals created from the same configuration with the same random seed, **When** both are run for 100 steps, **Then** they produce identical action sequences and fitness scores

---

### User Story 3 - Evolve Hierarchies with Evolutionary Algorithms (Priority: P1)

A researcher wants to automatically discover effective hierarchy configurations by evolving a population of individuals over multiple generations, with fitness-based selection, crossover, and mutation.

**Why this priority**: Evolution is a primary method for optimizing PCT hierarchies. This represents the library's key value for discovering control strategies without manual tuning.

**Independent Test**: Initialize DHPCTEvolver with a template individual, run evolution for 10 generations, and verify that the best individual's fitness improves over generations and outperforms random individuals.

**Acceptance Scenarios**:

1. **Given** a template DHPCTIndividual and fitness function, **When** user initializes DHPCTEvolver with pop_size=20 and generations=10, **Then** evolver is configured with DEAP toolbox
2. **Given** a configured evolver, **When** user calls run_evolution(), **Then** evolution runs for specified generations with selection, crossover, and mutation
3. **Given** an evolution run, **When** examining generation statistics, **Then** min/mean/max fitness values and mutation percentages are reported each generation
4. **Given** evolution configured with early termination fitness_target=500, **When** an individual achieves fitness ≥500, **Then** evolution terminates early and reports the best individual
5. **Given** evolution with save_arch_best=True, **When** each generation completes, **Then** the best individual's configuration is saved to a file

---

### User Story 4 - Apply Evolutionary Operators to Individuals (Priority: P2)

A researcher wants to manually experiment with evolutionary operators by taking existing individuals and creating offspring through mating (crossover) or generating variants through mutation.

**Why this priority**: This enables manual experimentation and understanding of how evolution works, supporting both automated and semi-supervised optimization workflows.

**Independent Test**: Create two parent individuals, mate them to produce two offspring, verify offspring have characteristics from both parents, then mutate one offspring and verify it differs from its pre-mutation state.

**Acceptance Scenarios**:

1. **Given** two DHPCTIndividual instances with different weight configurations, **When** user calls parent1.mate(parent2), **Then** two new offspring individuals are created with blended/crossed characteristics
2. **Given** a DHPCTIndividual, **When** user calls mutate(weight_prob=0.2), **Then** approximately 20% of the weights are modified
3. **Given** a DHPCTIndividual, **When** user calls mutate(struct_prob=0.1), **Then** structural changes may occur in the hierarchy
4. **Given** a DHPCTIndividual, **When** user calls evaluate(nevals=5), **Then** the individual is run 5 times and average or maximum fitness is returned

---

### User Story 5 - Optimize Evolution Hyperparameters (Priority: P2)

A researcher wants to find the best evolutionary algorithm settings (population size, mutation rates, selection pressure) by running multiple evolutionary trials with different hyperparameter combinations using automated optimization.

**Why this priority**: Hyperparameter tuning is time-consuming when done manually. Automated optimization accelerates research by finding effective settings faster.

**Independent Test**: Configure DHPCTOptimizer to search population size [20-100] and mutation rate [0.05-0.3], run 10 optimization trials, and verify that best parameters are returned and that optimization history shows improvement.

**Acceptance Scenarios**:

1. **Given** evolution parameters with fixed and variable flags, **When** user initializes DHPCTOptimizer with n_trials=20, **Then** optimizer is configured to search variable parameters
2. **Given** a configured optimizer, **When** user calls run_optimization(), **Then** Optuna runs specified trials testing different parameter combinations
3. **Given** a completed optimization, **When** user calls get_best_params(), **Then** the parameter combination that achieved best fitness is returned
4. **Given** a completed optimization, **When** user calls visualize_results(), **Then** parameter importance and optimization history plots are generated
5. **Given** optimizer configured with pruner, **When** a trial performs poorly early, **Then** the trial is pruned to save computational resources

---

### User Story 6 - Enable Online Learning During Execution (Priority: P3)

A researcher wants an evolved individual to continue learning while interacting with its environment by using gradient-based training on the Keras model to minimize error signals in real-time.

**Why this priority**: Online learning extends evolved individuals' capabilities, but requires basic evolution to be working first. This is an advanced feature for fine-tuning.

**Independent Test**: Create an individual, enable online learning with learning_rate=0.01, run for 1000 steps, and verify that error signals (comparator values) decrease over time compared to the same individual run without online learning.

**Acceptance Scenarios**:

1. **Given** a DHPCTIndividual, **When** user calls run(steps=1000, train=True), **Then** online learning is enabled during execution
2. **Given** online learning enabled, **When** the individual runs, **Then** comparator values are used as training targets with goal of zero error
3. **Given** online learning configuration with train_every_n_steps=10, **When** running, **Then** model weights are updated every 10 steps based on accumulated errors
4. **Given** error_weight_coefficients setting different weights per level, **When** training, **Then** higher-level errors have configurable influence on learning

---

### Edge Cases

- What happens when an environment terminates early (e.g., CartPole falls)?
  - If early_termination=True, run() stops when environment returns Done=True; otherwise continues resetting environment
- What happens when loading a configuration for an environment that's not installed?
  - from_config() should fail gracefully with clear error message indicating missing environment
- What happens when mate() is called with incompatible hierarchies (different level counts)?
  - Should either pad/adapt or raise clear error explaining incompatibility
- What happens when running evolution with population size less than 4?
  - Should warn user or use minimum viable population size for DEAP algorithms
- What happens when a hierarchy has only one level?
  - Should work correctly with perceptions from observations, references from input layer
- What happens when all individuals in a generation have identical fitness?
  - Selection still operates (possibly random), mutation still introduces variation
- What happens when saving configuration to a path that doesn't exist?
  - Should create parent directories or fail with clear error
- What happens during mutation if struct_prob is set to 1.0?
  - Every structure element could mutate - should still maintain valid hierarchy structure

## Requirements *(mandatory)*

### Functional Requirements

#### Core Individual Requirements

- **FR-001**: System MUST support creating DHPCTIndividual with environment name, hierarchy levels array, activation functions, and weight types
- **FR-002**: System MUST support creating DHPCTIndividual from configuration dictionary via from_config() class method
- **FR-003**: DHPCTIndividual MUST compile() to create a Keras Functional API model representing the PCT hierarchy
- **FR-004**: DHPCTIndividual MUST support run(steps, train, early_termination) to execute in environment and return fitness
- **FR-005**: DHPCTIndividual MUST support config() to return complete configuration dictionary including env, hierarchy, and weights
- **FR-005a**: Configuration dictionaries MUST be pickleable for serialization beyond JSON
- **FR-006**: DHPCTIndividual MUST support save_config(filepath) to persist configuration as JSON
- **FR-007**: DHPCTIndividual MUST support mate(other) to create two offspring via crossover
- **FR-008**: DHPCTIndividual MUST support mutate(struct_prob, weight_prob) to modify structure and/or weights
- **FR-009**: DHPCTIndividual MUST support evaluate(nevals) to run multiple trials and return fitness score
- **FR-009a**: DHPCTIndividual MUST support converting to and from legacy configuration formats via to_legacy_config() and from_legacy_config() methods
- **FR-009b**: DHPCTIndividual.run() MUST support optional history recording of all observations, layer values, and actions
- **FR-009c**: System MUST provide visualization functions for execution history showing observations, layer activations, and actions over time

#### Hierarchy Structure Requirements

- **FR-010**: Each level MUST have perception, reference, comparator, and output layers following PCT principles
- **FR-011**: Lowest level perception layer MUST receive inputs from environment observations, unless another level is explicitly specified for connection to observations
- **FR-012**: Higher level perception layers MUST receive inputs from all perception values of the level below
- **FR-013**: Highest level reference layer MUST receive inputs from external reference input layer
- **FR-014**: Lower level reference layers MUST receive inputs from all output values of the level above
- **FR-015**: Comparator at each level MUST compute reference minus perception (element-wise subtraction)
- **FR-016**: Output at each level MUST be element-wise multiplication of weights and comparator values
- **FR-017**: Actions MUST be computed from Level 0 output layer
- **FR-018**: Errors output MUST collect all comparator values across all levels
- **FR-019**: Model layers MUST follow naming convention: PL## (perception), RL## (reference), CL## (comparator), OL## (output), Observations, Actions, Errors
- **FR-019a**: System MUST provide visualization function to display network diagram of hierarchy showing all layers, nodes, and connections
- **FR-019b**: System MUST provide visualization function to display PCT control units as single nodes (combining reference, perception, comparator, output)
- **FR-019c**: System MUST provide visualization function to display hierarchy network with weight values shown on connections

#### Weight and Activation Requirements

- **FR-020**: System MUST support float weight type as default
- **FR-021**: System MUST support boolean weight type (binary 0/1 values)
- **FR-022**: System MUST support ternary weight type (-1, 0, 1 values)
- **FR-023**: Default activation function MUST be linear unless specified in activation_funcs
- **FR-024**: System MUST support per-level activation function configuration

#### Evolution Requirements

- **FR-025**: DHPCTEvolver MUST initialize with population size, generation count, and termination criteria
- **FR-026**: DHPCTEvolver MUST support setup_evolution(template_individual, fitness_function, minimize) to configure DEAP toolbox
- **FR-027**: DHPCTEvolver MUST support run_evolution(verbose) to execute evolutionary algorithm
- **FR-028**: DHPCTEvolver MUST track generation statistics: min/mean/max fitness, mutation percentages, elapsed time
- **FR-029**: DHPCTEvolver MUST support early termination based on evolve_static_termination (unchanged generations)
- **FR-030**: DHPCTEvolver MUST support parallelization of fitness evaluations during evolution
- **FR-031**: DHPCTEvolver MUST support save_arch_best option to save best individual configuration each generation
- **FR-032**: DHPCTEvolver MUST support save_arch_all option to save all individual configurations each generation
- **FR-033**: DHPCTEvolver MUST support run_best option to evaluate and display best individual each generation
- **FR-034**: DHPCTEvolver MUST support save_results(path) to persist evolution statistics and configurations
- **FR-034a**: DHPCTEvolver MUST support optional logging to comet_ml experiments for tracking evolution progress
- **FR-034b**: DHPCTEvolver MUST support initialization with random hierarchy structures (random level counts and units per level within specified limits)
- **FR-034c**: DHPCTEvolver MUST support initialization with existing or pre-trained individuals in addition to random initialization
- **FR-034d**: DHPCTEvolver MUST support marking specific weights, nodes, or levels as FIXED to prevent modification during evolution

#### Optimization Requirements

- **FR-035**: DHPCTOptimizer MUST support flexible parameter specification with fixed/variable flags
- **FR-036**: DHPCTOptimizer MUST initialize with n_trials, timeout, pruner, sampler, study_name, and storage
- **FR-037**: DHPCTOptimizer MUST support define_objective(template_individual, fitness_function, evaluation_budget)
- **FR-038**: DHPCTOptimizer MUST support run_optimization(verbose) to execute Optuna hyperparameter search
- **FR-039**: DHPCTOptimizer MUST support get_best_params() to retrieve optimal parameter combination
- **FR-040**: DHPCTOptimizer MUST support visualize_results() to generate parameter importance and history plots
- **FR-041**: DHPCTOptimizer MUST support save_results(path) to persist optimization study data

#### Online Learning Requirements

- **FR-042**: DHPCTIndividual.run() MUST support train=True to enable online learning during execution
- **FR-043**: Online learning MUST use comparator values as training outputs with target of zero
- **FR-044**: Online learning MUST support configurable learning rate and optimizer (e.g., adam)
- **FR-045**: Online learning MUST support train_every_n_steps to control training frequency
- **FR-046**: Online learning MUST support error_weight_coefficients to weight different levels' errors

#### Testing and Validation Requirements

- **FR-047**: System MUST be compatible with Gymnasium environments and any environments following the same interface pattern (CartPole-v1, LunarLanderContinuous-v2, etc.)
- **FR-048**: All operations MUST support deterministic behavior via random seed configuration
- **FR-049**: System MUST validate configurations and raise clear errors for invalid inputs

### Key Entities

- **DHPCTIndividual**: Represents a single hierarchical control system with an environment and Keras model. Attributes include environment name/properties, hierarchy structure (levels, activations, weight types), compiled Keras model, and current weights.

- **DHPCTEvolver**: Manages evolutionary optimization of a population of individuals. Attributes include population size, generation count, termination criteria, DEAP toolbox configuration, and evolution statistics.

- **DHPCTOptimizer**: Manages hyperparameter optimization using Optuna. Attributes include trial count, timeout settings, parameter search space (fixed vs variable), Optuna study, and optimization results.

- **Hierarchy Configuration**: Dictionary containing complete specification of a hierarchy. Includes environment settings, level definitions, activation functions, weight types, and all weight/bias values.

- **Generation Statistics**: Data collected during evolution. Includes generation number, min/mean/max fitness, mutation percentages, best individual, and elapsed time.

- **PCT Control Unit**: Conceptual entity (implemented as Keras layers). Each unit has reference input, perception input, comparator (reference - perception), and output (weighted comparator).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can create a 3-level hierarchy and successfully run it in CartPole environment for 500 steps within 30 seconds of setup time
- **SC-002**: Configuration save/load round-trip produces bit-identical behavior when running same individual with same random seed
- **SC-003**: Evolution of population of 50 individuals over 100 generations completes within 10 minutes for CartPole environment
- **SC-004**: Evolution shows measurable fitness improvement: best fitness in generation 100 is at least 50% better than generation 1
- **SC-005**: Saved configurations are human-readable JSON files under 100KB for typical hierarchies (3-5 levels, 2-10 units per level)
- **SC-006**: Hyperparameter optimization with 20 trials finds parameter combination that achieves 20% better fitness than default parameters
- **SC-007**: Unit tests achieve >90% code coverage across all three main classes
- **SC-008**: All nbdev notebooks execute without errors when running nbdev_prepare
- **SC-009**: Documentation automatically generated from notebooks includes working code examples for each major feature
- **SC-010**: Library successfully controls both CartPole-v1 and LunarLanderContinuous-v2 environments demonstrating generalizability

## Assumptions *(mandatory)*

- Users have Python 3.8+ installed with ability to install pip packages
- Users are familiar with basic Python programming and Jupyter notebooks
- Users have basic understanding of Perceptual Control Theory concepts or are willing to learn from documentation
- Gymnasium or compatible environments can be installed and run successfully in the user's environment
- TensorFlow/Keras runs successfully on the user's hardware (CPU is sufficient, GPU optional, parallelization benefits from multiple cores)
- Users developing the library are familiar with nbdev framework and literate programming
- DEAP library provides sufficient evolutionary algorithm primitives for the use cases
- Optuna library provides sufficient hyperparameter optimization capabilities
- JSON is an acceptable format for configuration persistence (no binary format required initially)
- Hierarchies with 2-10 levels and 2-20 units per level represent the typical use case (not extremely deep/wide networks)
- Default linear activations are reasonable starting point (users can customize as needed)
- Float weights are sufficient for most use cases (boolean/ternary are specialized options)
- Fitness functions can be user-defined and environment-specific
- Evolution population sizes of 20-100 are practical for most use cases
- Online learning uses standard gradient descent without requiring advanced RL algorithms

## Dependencies

### External Dependencies

- **nbdev**: Literate programming framework for notebook-based development (MANDATORY)
- **tensorflow**: Keras model implementation for hierarchies (SUGGESTED)
- **gymnasium**: Environment interface (successor to OpenAI Gym) (SUGGESTED)
- **deap**: Evolutionary algorithm framework (SUGGESTED)
- **numpy**: Numerical operations (SUGGESTED)
- **matplotlib**: Visualization of evolution progress and results (SUGGESTED)
- **optuna**: Hyperparameter optimization (SUGGESTED)
- **networkx**: For visualizing hierarchy structure as graphs (SUGGESTED)
- **comet_ml**: Optional for logging evolution experiments
- **json**: Configuration persistence (Python standard library)
- **pickle**: Configuration serialization (Python standard library)

### Internal Dependencies

- DHPCTEvolver depends on DHPCTIndividual for creating/managing individuals
- DHPCTOptimizer depends on DHPCTEvolver for running optimization trials
- Online learning depends on DHPCTIndividual having compiled Keras model
- All components depend on Gymnasium environments being available

### Environment Dependencies

- Python 3.8 or higher
- Sufficient memory for TensorFlow operations and population of individuals
- No GPU required but can accelerate training if available

## Out of Scope *(mandatory)*

The following are explicitly NOT included in this feature:

- **Custom environment creation tools**: Users must use existing Gymnasium environments or create their own separately
- **GUI/web interface**: Library is Python-only, no graphical interface provided
- **Advanced RL algorithms**: No integration with PPO, A3C, or other deep RL methods
- **Model compression/quantization**: No optimization for deployment or edge devices
- **Multi-objective optimization**: Only single fitness value supported
- **Automatic environment detection**: Users must explicitly specify environment name
- **Cloud deployment tools**: No built-in support for AWS, GCP, or Azure
- **Benchmark suite**: No standardized benchmark environments or comparison tools
- **Legacy support**: No support for original OpenAI Gym (only Gymnasium)

## Notes

This specification is derived from the comprehensive PROJECT_REQUIREMENTS.md document which provides detailed implementation guidance including:

- Specific class structures and method signatures
- Keras model architecture details
- Layer naming conventions
- Configuration dictionary format
- nbdev workflow and file naming patterns
- Unit testing requirements

Implementers should refer to PROJECT_REQUIREMENTS.md for technical implementation details while using this specification for understanding user needs and acceptance criteria.

The specification focuses on WHAT the system must do (user value) while PROJECT_REQUIREMENTS.md describes HOW to implement it (technical approach).
