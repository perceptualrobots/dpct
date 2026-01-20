# Data Model: DPCT Core Library

**Date**: 2026-01-20  
**Purpose**: Define entities, their attributes, relationships, and state transitions

## Core Entities

### 1. DHPCTIndividual

**Description**: Represents a single hierarchical Perceptual Control Theory system with an associated environment and Keras model.

**Attributes**:
- `env_name` (str): Gymnasium environment identifier (e.g., "CartPole-v1")
- `env_properties` (dict): Environment-specific configuration (observation/action spaces, etc.)
- `levels` (list[int]): Number of units in each hierarchy level, bottom to top (e.g., [4, 3, 2])
- `activation_funcs` (list[str] | str): Activation function per level or single default (e.g., "linear")
- `weight_types` (list[str] | str): Weight type per level: "float", "boolean", or "ternary"
- `model` (keras.Model | None): Compiled Keras Functional API model (None before compile())
- `weights` (dict[str, np.ndarray]): Current weight values keyed by layer name
- `fixed_weights` (set[str]): Layer names that should not be mutated
- `fixed_levels` (set[int]): Level indices that should not be mutated
- `obs_connection_level` (int): Which level connects to observations (default: 0)
- `fitness` (float | None): Most recent fitness evaluation result

**Relationships**:
- Creates and owns one Keras `Model` instance
- Interacts with one Gymnasium `Environment` instance during execution
- Member of zero or one `DHPCTEvolver` populations
- Configuration can be serialized to/from `HierarchyConfiguration`

**State Transitions**:
1. **Uncompiled** → `compile()` → **Compiled**: Keras model is built from hierarchy specification
2. **Compiled** → `run()` → **Evaluated**: Fitness computed from environment interaction
3. **Evaluated** → `mutate()` → **Compiled**: Weights modified, stays compiled
4. **Compiled** → `mate(other)` → **Offspring Created**: Two new compiled individuals
5. **Any** → `save_config()` → **Persisted**: Configuration saved to disk
6. **None** → `from_config()` → **Compiled**: New individual loaded from configuration

**Validation Rules**:
- `levels` must have at least 1 element, each element ≥ 1
- `activation_funcs` length must be 1 or match `len(levels)`
- `weight_types` length must be 1 or match `len(levels)`
- `obs_connection_level` must be < `len(levels)`
- `model` inputs must include "Observations" and "ReferencesInput"
- `model` outputs must include "Actions" and "Errors"

---

### 2. DHPCTEvolver

**Description**: Manages evolutionary optimization of a population of `DHPCTIndividual` instances using genetic algorithms.

**Attributes**:
- `pop_size` (int): Number of individuals in population
- `generations` (int): Number of generations to evolve
- `cxpb` (float): Crossover probability (0.0-1.0)
- `mutpb` (float): Mutation probability (0.0-1.0)
- `tournsize` (int): Tournament selection size
- `template_individual` (DHPCTIndividual): Blueprint for creating initial population
- `fitness_function` (callable): Function mapping individual to fitness score
- `minimize` (bool): Whether to minimize (True) or maximize (False) fitness
- `population` (list[DHPCTIndividual]): Current generation's individuals
- `hall_of_fame` (deap.HallOfFame): Best individuals across all generations
- `statistics` (list[dict]): Per-generation statistics (min/mean/max fitness, etc.)
- `toolbox` (deap.Toolbox): DEAP toolbox with registered operators
- `save_arch_best` (bool): Whether to save best individual config each generation
- `save_arch_all` (bool): Whether to save all individual configs each generation
- `run_best` (bool): Whether to evaluate and display best individual each generation
- `evolve_static_termination` (int | None): Terminate if no improvement for N generations
- `parallel` (bool): Whether to use multiprocessing for fitness evaluations
- `use_comet` (bool): Whether to log to comet_ml
- `comet_experiment` (comet_ml.Experiment | None): Experiment instance if logging
- `random_structure` (dict | None): Config for random structure initialization (min/max levels/units)
- `initial_individuals` (list[DHPCTIndividual] | None): Pre-trained individuals to include in initial population

**Relationships**:
- Creates and manages population of `DHPCTIndividual` instances
- Uses `fitness_function` to evaluate individuals
- May interact with `DHPCTOptimizer` (optimizer runs multiple evolvers)
- Optionally logs to comet_ml `Experiment`

**State Transitions**:
1. **Unconfigured** → `setup_evolution()` → **Configured**: DEAP toolbox initialized
2. **Configured** → `run_evolution()` → **Evolving**: Generation loop executing
3. **Evolving** → (per generation) → **Evaluated**: Population fitness computed
4. **Evaluated** → (selection/crossover/mutation) → **Evolved**: New generation created
5. **Evolving** → (termination condition) → **Completed**: Evolution finished
6. **Completed** → `save_results()` → **Persisted**: Statistics and configs saved

**Validation Rules**:
- `pop_size` ≥ 4 (minimum for tournament selection)
- `0.0 ≤ cxpb, mutpb ≤ 1.0`
- `tournsize` ≥ 2 and ≤ `pop_size`
- `fitness_function` must return numeric value
- `template_individual` must be compiled before use
- If `random_structure` provided, must have min_levels, max_levels, min_units, max_units

---

### 3. DHPCTOptimizer

**Description**: Manages hyperparameter optimization using Optuna to find best evolutionary algorithm settings.

**Attributes**:
- `n_trials` (int): Number of optimization trials to run
- `timeout` (int | None): Maximum time in seconds for optimization
- `parameters` (dict): Parameter definitions with 'fixed' flag and search space
- `template_individual` (DHPCTIndividual): Blueprint for evolution runs
- `fitness_function` (callable): Fitness function for evaluating individuals
- `evaluation_budget` (dict): Resources per trial (generations, population size)
- `study` (optuna.Study): Optuna study instance
- `pruner` (optuna.pruners.BasePruner | None): Pruner for early stopping trials
- `sampler` (optuna.samplers.BaseSampler | None): Sampling strategy (default: TPE)
- `study_name` (str | None): Name for persistent study
- `storage` (str | None): Database URL for study persistence
- `best_params` (dict | None): Best parameter combination found
- `best_value` (float | None): Best fitness achieved

**Relationships**:
- Creates multiple `DHPCTEvolver` instances (one per trial)
- Uses `template_individual` to configure evolvers
- Uses Optuna `Study` for optimization management

**State Transitions**:
1. **Unconfigured** → `define_objective()` → **Configured**: Objective function defined
2. **Configured** → `run_optimization()` → **Optimizing**: Trials executing
3. **Optimizing** → (per trial) → **Trial Running**: Evolver running for trial
4. **Trial Running** → (pruner check) → **Trial Pruned** or **Trial Completed**
5. **Optimizing** → (all trials done) → **Completed**: Optimization finished
6. **Completed** → `get_best_params()` → **Best Returned**: Optimal params extracted
7. **Completed** → `visualize_results()` → **Visualized**: Plots generated
8. **Completed** → `save_results()` → **Persisted**: Study saved to storage

**Validation Rules**:
- `n_trials` ≥ 1
- `timeout` > 0 if provided
- `parameters` must have at least one variable parameter
- Each parameter must specify type (int/float/categorical) and range/choices
- `evaluation_budget` must specify sufficient resources for meaningful evolution

---

### 4. HierarchyConfiguration

**Description**: Dictionary representation of a complete hierarchy specification, used for serialization and deserialization.

**Structure**:
```python
{
    "env_name": str,                    # e.g., "CartPole-v1"
    "env_properties": {
        "observation_space": {...},     # Gymnasium space spec
        "action_space": {...}            # Gymnasium space spec
    },
    "hierarchy": {
        "levels": list[int],             # e.g., [4, 3, 2]
        "activation_funcs": list[str],   # e.g., ["linear", "linear", "linear"]
        "weight_types": list[str],       # e.g., ["float", "float", "float"]
        "obs_connection_level": int      # e.g., 0
    },
    "weights": {
        "PL00": np.ndarray | list,      # Perception layer 0 weights
        "RL00": np.ndarray | list,      # Reference layer 0 weights
        "CL00": None,                    # Comparators have no weights
        "OL00": np.ndarray | list,      # Output layer 0 weights
        # ... for each level
    },
    "biases": {
        # Similar structure for biases
    },
    "fixed_weights": list[str],         # e.g., ["PL01", "RL01"]
    "fixed_levels": list[int],          # e.g., [2]
    "metadata": {
        "created": str,                  # ISO timestamp
        "version": str,                  # Library version
        "fitness": float | None          # Last known fitness
    }
}
```

**Validation Rules**:
- All layer names in `weights` must follow naming convention (PL##, RL##, OL##)
- Weight array shapes must match layer dimensions
- `levels` length must match number of layer sets in `weights`
- Must be JSON-serializable (or pickleable)

---

### 5. GenerationStatistics

**Description**: Metrics collected for a single generation during evolution.

**Attributes**:
- `generation` (int): Generation number (0-indexed)
- `min_fitness` (float): Minimum fitness in population
- `mean_fitness` (float): Mean fitness in population
- `max_fitness` (float): Maximum fitness in population
- `std_fitness` (float): Standard deviation of fitness
- `mutation_pct` (float): Percentage of population that underwent mutation
- `crossover_pct` (float): Percentage of population that underwent crossover
- `best_individual_config` (HierarchyConfiguration | None): Config of best individual
- `elapsed_time` (float): Time in seconds for this generation
- `timestamp` (str): ISO timestamp when generation completed

**Relationships**:
- Collected by `DHPCTEvolver` during `run_evolution()`
- Multiple statistics form evolution history
- Can be logged to comet_ml experiment

**Validation Rules**:
- `min_fitness ≤ mean_fitness ≤ max_fitness`
- `0.0 ≤ mutation_pct, crossover_pct ≤ 1.0`
- `elapsed_time ≥ 0`

---

### 6. PCTControlUnit (Conceptual)

**Description**: Conceptual entity representing the PCT control loop at one level. Implemented as a group of Keras layers, not a standalone class.

**Components**:
- **Perception Function**: Dense layer transforming inputs to perceptual signal
- **Reference Function**: Dense layer transforming higher-level inputs to reference signal
- **Comparator**: Subtraction layer computing error (reference - perception)
- **Output Function**: Multiplication layer applying weights to error signal

**Layer Names** (for level L with U units):
- Perception: `PL{L:02d}` (e.g., PL00, PL01)
- Reference: `RL{L:02d}`
- Comparator: `CL{L:02d}`
- Output: `OL{L:02d}`

**Connections**:
- **Level 0 (lowest)**:
  - Perception input: Environment observations
  - Reference input: Output from Level 1 (or external reference if only one level)
  - Output: Actions to environment
- **Level N (highest)**:
  - Perception input: All perception values from Level N-1
  - Reference input: External reference input layer
  - Output: Reference signals for Level N-1
- **Middle levels**:
  - Perception input: All perception values from level below
  - Reference input: All output values from level above
  - Output: Reference signals for level below

**Validation Rules**:
- Perception input dimension matches source dimension
- Reference input dimension matches source dimension
- Comparator inputs have matching dimensions (element-wise subtraction)
- Output dimension matches number of units in this level

---

### 7. ExecutionHistory

**Description**: Record of all state transitions during an individual's execution in an environment.

**Attributes**:
- `observations` (list[np.ndarray]): Environment observations at each step
- `actions` (list[np.ndarray]): Actions taken at each step
- `rewards` (list[float]): Rewards received at each step
- `errors` (list[np.ndarray]): Comparator values (error signals) at each step
- `layer_activations` (dict[str, list[np.ndarray]]): Activation values for each layer at each step
- `terminated` (list[bool]): Whether episode terminated at each step
- `truncated` (list[bool]): Whether episode truncated at each step
- `timestamps` (list[float]): Relative time in seconds for each step

**Relationships**:
- Created by `DHPCTIndividual.run()` when `record_history=True`
- Used by visualization functions to generate plots

**Methods**:
- `plot_observations()`: Time series of observation values
- `plot_actions()`: Time series of action values
- `plot_errors()`: Time series of error signals
- `plot_layer(layer_name)`: Time series of specific layer activations
- `to_dataframe()`: Convert to pandas DataFrame for analysis

**Validation Rules**:
- All lists must have same length (number of steps)
- Array dimensions must match environment/model specifications

---

## Entity Relationships Diagram

```
┌─────────────────────┐
│ DHPCTOptimizer      │
│                     │
│ - n_trials          │
│ - parameters        │
│ - study             │
└──────┬──────────────┘
       │ creates multiple
       │
       ▼
┌─────────────────────┐      manages     ┌─────────────────────┐
│ DHPCTEvolver        ├─────────────────►│ DHPCTIndividual     │
│                     │   population of  │                     │
│ - pop_size          │                  │ - env_name          │
│ - generations       │                  │ - levels            │
│ - toolbox           │                  │ - model             │
│ - statistics        │                  │ - weights           │
└──────┬──────────────┘                  └──────┬──────────────┘
       │ collects                               │
       │                                        │ serializes to/from
       ▼                                        ▼
┌─────────────────────┐                  ┌─────────────────────┐
│ GenerationStatistics│                  │HierarchyConfiguration│
│                     │                  │                     │
│ - generation        │                  │ - env_name          │
│ - min/mean/max      │                  │ - hierarchy         │
│ - elapsed_time      │                  │ - weights           │
└─────────────────────┘                  └─────────────────────┘

       DHPCTIndividual
             │
             │ optionally records
             ▼
       ┌─────────────────────┐
       │ ExecutionHistory    │
       │                     │
       │ - observations      │
       │ - actions           │
       │ - errors            │
       │ - layer_activations │
       └─────────────────────┘
```

## Data Flow

### Creating and Running an Individual
```
1. User creates DHPCTIndividual(env_name="CartPole-v1", levels=[4,3,2])
2. Individual.compile() → builds Keras model with PCT structure
3. Individual.run(steps=500) → 
   - Creates Gymnasium environment
   - For each step:
     - Observations → Model input
     - Model predicts Actions and Errors
     - Environment.step(Actions) → new observations, rewards
   - Returns fitness (sum of rewards)
```

### Evolution Process
```
1. User creates template DHPCTIndividual
2. User creates DHPCTEvolver(template, pop_size=50, generations=100)
3. Evolver.setup_evolution(fitness_function) → configures DEAP toolbox
4. Evolver.run_evolution() →
   For each generation:
     - Evaluate population fitness (parallel if enabled)
     - Select parents via tournament selection
     - Create offspring via crossover
     - Mutate offspring
     - Replace population with offspring
     - Collect GenerationStatistics
     - Optionally save best individual config
     - Optionally log to comet_ml
   - Returns best individual and statistics
```

### Hyperparameter Optimization
```
1. User creates template DHPCTIndividual
2. User creates DHPCTOptimizer(parameters={...}, n_trials=20)
3. Optimizer.define_objective(template, fitness_function, budget={...})
4. Optimizer.run_optimization() →
   For each trial:
     - Optuna suggests parameter values
     - Create DHPCTEvolver with suggested parameters
     - Run evolution with budget constraints
     - Return best fitness from evolution
     - Optuna updates search strategy
   - Returns best parameters and study results
```

## Persistence Formats

### JSON Configuration Example
```json
{
  "env_name": "CartPole-v1",
  "hierarchy": {
    "levels": [4, 3, 2],
    "activation_funcs": ["linear", "linear", "linear"],
    "weight_types": ["float", "float", "float"]
  },
  "weights": {
    "PL00": [[0.5, -0.3, 0.1, 0.7], ...],
    "RL00": [[0.2, 0.4, -0.1, 0.3], ...],
    ...
  },
  "metadata": {
    "created": "2026-01-20T10:30:00Z",
    "fitness": 487.5
  }
}
```

### Evolution Results Structure
```python
{
    "parameters": {
        "pop_size": 50,
        "generations": 100,
        "cxpb": 0.5,
        "mutpb": 0.2
    },
    "statistics": [
        {"generation": 0, "min": 10.5, "mean": 45.2, "max": 98.7, ...},
        {"generation": 1, "min": 15.3, "mean": 52.1, "max": 124.5, ...},
        ...
    ],
    "best_individual": {...},  # HierarchyConfiguration
    "total_time": 542.3
}
```

## Next Steps

1. Define API contracts in `contracts/` directory
2. Create quickstart guide demonstrating entity usage
3. Implement entities in notebooks following nbdev patterns
