# API Contract: DHPCTEvolver

**Module**: `dpct.evolver`  
**Description**: Manages evolutionary optimization of DHPCTIndividual populations

## Constructor

```python
DHPCTEvolver(
    pop_size: int = 50,
    generations: int = 100,
    cxpb: float = 0.5,
    mutpb: float = 0.2,
    tournsize: int = 3,
    evolve_static_termination: int | None = None,
    save_arch_best: bool = False,
    save_arch_all: bool = False,
    run_best: bool = False,
    parallel: bool = False,
    use_comet: bool = False,
    comet_config: dict | None = None,
    random_structure: dict | None = None,
    initial_individuals: list[DHPCTIndividual] | None = None,
    random_seed: int | None = None
)
```

**Parameters**:
- `pop_size`: Number of individuals in population (≥4)
- `generations`: Number of generations to evolve
- `cxpb`: Crossover probability [0.0, 1.0]
- `mutpb`: Mutation probability [0.0, 1.0]
- `tournsize`: Tournament selection size (2 ≤ size ≤ pop_size)
- `evolve_static_termination`: Stop if no improvement for N generations
- `save_arch_best`: Save best individual config each generation
- `save_arch_all`: Save all individual configs each generation
- `run_best`: Evaluate and display best individual each generation
- `parallel`: Use multiprocessing for fitness evaluation
- `use_comet`: Log evolution to comet_ml
- `comet_config`: Configuration dict for comet_ml Experiment
- `random_structure`: Dict with min/max levels/units for random initialization
- `initial_individuals`: Pre-trained individuals to include in initial population
- `random_seed`: Seed for reproducible evolution

**Returns**: New DHPCTEvolver instance (unconfigured)

**Raises**:
- `ValueError`: If pop_size < 4, probabilities out of range, or tournsize invalid

---

## Instance Methods

### setup_evolution

```python
def setup_evolution(
    self,
    template_individual: DHPCTIndividual,
    fitness_function: callable,
    minimize: bool = False,
    weight_mutation_strength: float = 1.0,
    struct_mutation_prob: float = 0.0
) -> None
```

**Description**: Configure DEAP toolbox for evolution

**Parameters**:
- `template_individual`: Blueprint for creating population
- `fitness_function`: Function mapping DHPCTIndividual → float
- `minimize`: Whether to minimize (True) or maximize (False) fitness
- `weight_mutation_strength`: Scale factor for weight mutations
- `struct_mutation_prob`: Probability of structural mutations

**Side Effects**:
- Creates `self.toolbox` with registered operators
- Creates `self.hall_of_fame` for tracking best individuals
- Initializes `self.population` based on template and random_structure
- If `initial_individuals` provided, includes them in population
- If `use_comet=True`, creates comet_ml Experiment

**Raises**:
- `ValueError`: If template_individual not compiled
- `RuntimeError`: If already configured

**Post-conditions**:
- `self.toolbox` has operators: individual, population, evaluate, mate, mutate, select, map
- `self.population` has `pop_size` individuals
- If `random_structure` specified, population has diverse structures

---

### run_evolution

```python
def run_evolution(self, verbose: bool = True) -> tuple[DHPCTIndividual, list[dict]]
```

**Description**: Execute evolutionary algorithm

**Parameters**:
- `verbose`: Print generation statistics to console

**Returns**: Tuple of (best_individual, statistics_list)
- `best_individual`: Highest fitness individual across all generations
- `statistics_list`: List of GenerationStatistics dicts

**Side Effects**:
- Updates `self.population` each generation
- Appends to `self.statistics` each generation
- If `save_arch_best=True`, writes configs to files
- If `save_arch_all=True`, writes all configs to files
- If `run_best=True`, executes and displays best individual
- If `use_comet=True`, logs metrics to comet_ml
- If `parallel=True`, uses multiprocessing.Pool for evaluation

**Termination Conditions**:
1. Reaches `self.generations` count
2. If `evolve_static_termination` set: no improvement for N generations

**Raises**:
- `RuntimeError`: If not configured (setup_evolution not called)

**Algorithm**:
```
For each generation:
  1. Evaluate population fitness (parallel if enabled)
  2. Record statistics (min, mean, max, std)
  3. Select parents via tournament selection
  4. Apply crossover with probability cxpb
  5. Apply mutation with probability mutpb
  6. Replace population with offspring
  7. Update hall of fame
  8. Check termination conditions
  9. Save configs if requested
  10. Log to comet_ml if enabled
```

---

### save_results

```python
def save_results(self, dirpath: str) -> None
```

**Description**: Save evolution results to directory

**Parameters**:
- `dirpath`: Directory path to save results

**Side Effects**:
- Creates directory if not exists
- Writes `statistics.json` with generation statistics
- Writes `best_individual.json` with best config
- Writes `parameters.json` with evolution parameters
- Writes `hall_of_fame.json` with top individuals

**Raises**:
- `IOError`: If files cannot be written
- `RuntimeError`: If evolution not yet run

---

### get_best_individual

```python
def get_best_individual(self) -> DHPCTIndividual
```

**Description**: Get highest fitness individual from current population

**Returns**: Best DHPCTIndividual

**Raises**:
- `RuntimeError`: If population not initialized

---

### get_statistics

```python
def get_statistics(self) -> list[dict]
```

**Description**: Get generation statistics history

**Returns**: List of GenerationStatistics dicts

---

### plot_evolution

```python
def plot_evolution(
    self,
    metrics: list[str] = ["min", "mean", "max"],
    save_path: str | None = None
) -> None
```

**Description**: Plot evolution progress

**Parameters**:
- `metrics`: Which metrics to plot ("min", "mean", "max", "std")
- `save_path`: If provided, save plot to file instead of displaying

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `RuntimeError`: If evolution not yet run
- `ValueError`: If unknown metric requested

---

## Properties

### is_configured

```python
@property
def is_configured(self) -> bool
```

**Returns**: True if setup_evolution() has been called

---

### current_generation

```python
@property
def current_generation(self) -> int
```

**Returns**: Current generation number (0 if not started)

---

### best_fitness

```python
@property
def best_fitness(self) -> float | None
```

**Returns**: Best fitness achieved so far (None if not run)

---

## Usage Example

```python
# Create template individual
template = DHPCTIndividual(
    env_name="CartPole-v1",
    levels=[4, 3, 2],
    random_seed=42
)
template.compile()

# Define fitness function
def fitness_function(individual):
    return individual.run(steps=500, early_termination=True)

# Create evolver with random structure initialization
evolver = DHPCTEvolver(
    pop_size=50,
    generations=100,
    parallel=True,
    random_structure={
        'min_levels': 2,
        'max_levels': 5,
        'min_units': 2,
        'max_units': 10
    },
    use_comet=True,
    comet_config={'project_name': 'dpct-evolution'}
)

# Setup and run evolution
evolver.setup_evolution(template, fitness_function)
best, stats = evolver.run_evolution(verbose=True)

print(f"Best fitness: {best.fitness}")

# Save results
evolver.save_results("results/evolution_run_1")

# Plot progress
evolver.plot_evolution(metrics=["min", "mean", "max"])
```

## Related Requirements

- FR-025: Initialize with pop_size, generations, termination criteria
- FR-026: setup_evolution() configures DEAP toolbox
- FR-027: run_evolution() executes algorithm
- FR-028: Track generation statistics
- FR-029: evolve_static_termination for early stopping
- FR-030: Parallelization support
- FR-031: save_arch_best option
- FR-032: save_arch_all option
- FR-033: run_best option
- FR-034: save_results() method
- FR-034a: comet_ml logging
- FR-034b: random_structure initialization
- FR-034c: initial_individuals support
- FR-034d: Fixed weights/levels support (inherited from DHPCTIndividual)
