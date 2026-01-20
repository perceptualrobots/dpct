# API Contract: DHPCTOptimizer

**Module**: `dpct.optimizer`  
**Description**: Manages hyperparameter optimization using Optuna

## Constructor

```python
DHPCTOptimizer(
    parameters: dict,
    n_trials: int = 20,
    timeout: int | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    random_seed: int | None = None
)
```

**Parameters**:
- `parameters`: Parameter specification dict (see Parameters Format below)
- `n_trials`: Number of optimization trials to run
- `timeout`: Maximum time in seconds (None = no timeout)
- `pruner`: Optuna pruner for early stopping (default: MedianPruner)
- `sampler`: Optuna sampler (default: TPESampler)
- `study_name`: Name for study (for persistence)
- `storage`: Database URL for study storage (e.g., "sqlite:///optuna.db")
- `random_seed`: Seed for reproducible optimization

**Returns**: New DHPCTOptimizer instance (unconfigured)

**Raises**:
- `ValueError`: If parameters invalid or n_trials < 1

### Parameters Format

```python
{
    "pop_size": {
        "type": "int",
        "low": 20,
        "high": 100,
        "fixed": False
    },
    "generations": {
        "type": "int",
        "value": 50,
        "fixed": True  # Not optimized
    },
    "cxpb": {
        "type": "float",
        "low": 0.3,
        "high": 0.9,
        "fixed": False
    },
    "mutpb": {
        "type": "float",
        "low": 0.05,
        "high": 0.5,
        "fixed": False
    },
    "tournsize": {
        "type": "categorical",
        "choices": [2, 3, 5, 7],
        "fixed": False
    },
    "weight_mutation_strength": {
        "type": "float",
        "low": 0.1,
        "high": 2.0,
        "log": True,  # Log scale
        "fixed": False
    }
}
```

**Parameter Types**:
- `int`: Integer parameter with low/high bounds
- `float`: Float parameter with low/high bounds, optional log scale
- `categorical`: Discrete choices from list

**Fixed Parameters**:
- If `fixed=True`, parameter uses `value` and is not optimized
- If `fixed=False`, parameter is optimized within specified range

---

## Instance Methods

### define_objective

```python
def define_objective(
    self,
    template_individual: DHPCTIndividual,
    fitness_function: callable,
    evaluation_budget: dict,
    minimize: bool = False
) -> None
```

**Description**: Define optimization objective function

**Parameters**:
- `template_individual`: Blueprint for creating individuals
- `fitness_function`: Function mapping DHPCTIndividual → float
- `evaluation_budget`: Resource limits per trial (see Budget Format below)
- `minimize`: Whether to minimize (True) or maximize (False) fitness

**Side Effects**:
- Creates `self.objective` function for Optuna
- Creates `self.study` if not exists
- Stores template and fitness function for trials

**Raises**:
- `ValueError`: If template not compiled or budget invalid
- `RuntimeError`: If already configured

### Evaluation Budget Format

```python
{
    "generations": 50,      # Max generations per trial
    "pop_size": 30,         # Population size per trial (or use optimized value)
    "max_time": 300,        # Max seconds per trial
    "nevals": 1             # Evaluations per individual fitness
}
```

---

### run_optimization

```python
def run_optimization(self, verbose: bool = True) -> optuna.Study
```

**Description**: Execute hyperparameter optimization

**Parameters**:
- `verbose`: Print trial progress to console

**Returns**: Completed Optuna Study object

**Side Effects**:
- Runs `n_trials` optimization trials
- Updates `self.study` with results
- If `storage` provided, persists study to database
- If `pruner` enabled, may prune underperforming trials

**Raises**:
- `RuntimeError`: If objective not defined

**Trial Process**:
```
For each trial:
  1. Optuna suggests parameter values
  2. Create DHPCTEvolver with suggested parameters
  3. Setup evolution with template and fitness function
  4. Run evolution within evaluation budget
  5. Report best fitness to Optuna
  6. Pruner checks if trial should be stopped early
  7. Update study with trial results
```

---

### get_best_params

```python
def get_best_params(self) -> dict
```

**Description**: Get optimal parameter combination

**Returns**: Dict with best parameter values (includes both fixed and optimized)

**Raises**:
- `RuntimeError`: If optimization not yet run

**Example Return**:
```python
{
    "pop_size": 75,
    "generations": 50,  # Fixed
    "cxpb": 0.65,
    "mutpb": 0.18,
    "tournsize": 5,
    "weight_mutation_strength": 0.85
}
```

---

### get_best_value

```python
def get_best_value(self) -> float
```

**Description**: Get best fitness achieved across all trials

**Returns**: Best fitness value

**Raises**:
- `RuntimeError`: If optimization not yet run

---

### visualize_results

```python
def visualize_results(
    self,
    plots: list[str] = ["importance", "history", "parallel_coordinate"],
    save_dir: str | None = None
) -> None
```

**Description**: Generate visualization plots

**Parameters**:
- `plots`: Which plots to generate:
  - "importance": Parameter importance
  - "history": Optimization history
  - "parallel_coordinate": Parameter relationships
  - "contour": 2D parameter contours
  - "slice": Parameter slices
- `save_dir`: If provided, save plots to directory instead of displaying

**Side Effects**:
- Displays matplotlib plots or saves to files
- Uses Optuna's built-in visualization functions

**Raises**:
- `RuntimeError`: If optimization not yet run
- `ValueError`: If unknown plot type requested

---

### save_results

```python
def save_results(self, dirpath: str) -> None
```

**Description**: Save optimization results to directory

**Parameters**:
- `dirpath`: Directory path to save results

**Side Effects**:
- Creates directory if not exists
- Writes `best_params.json` with optimal parameters
- Writes `study_results.json` with all trial results
- Writes `parameter_importance.json` with importance scores
- If `storage` not used, writes `study.pkl` with pickled study

**Raises**:
- `IOError`: If files cannot be written
- `RuntimeError`: If optimization not yet run

---

### get_study

```python
def get_study(self) -> optuna.Study
```

**Description**: Get Optuna Study object for advanced analysis

**Returns**: optuna.Study instance

---

### get_trials_dataframe

```python
def get_trials_dataframe(self) -> pd.DataFrame
```

**Description**: Get trials as pandas DataFrame

**Returns**: DataFrame with columns for parameters, fitness, state, duration

**Raises**:
- `ImportError`: If pandas not available
- `RuntimeError`: If optimization not yet run

---

## Properties

### is_configured

```python
@property
def is_configured(self) -> bool
```

**Returns**: True if define_objective() has been called

---

### n_completed_trials

```python
@property
def n_completed_trials(self) -> int
```

**Returns**: Number of successfully completed trials

---

### n_pruned_trials

```python
@property
def n_pruned_trials(self) -> int
```

**Returns**: Number of pruned trials

---

## Usage Example

```python
# Create template individual
template = DHPCTIndividual(
    env_name="LunarLanderContinuous-v2",
    levels=[8, 6, 4],
    random_seed=42
)
template.compile()

# Define fitness function
def fitness_function(individual):
    return individual.evaluate(nevals=3, aggregate="mean")

# Define parameter search space
parameters = {
    "pop_size": {"type": "int", "low": 30, "high": 100, "fixed": False},
    "generations": {"type": "int", "value": 50, "fixed": True},
    "cxpb": {"type": "float", "low": 0.4, "high": 0.9, "fixed": False},
    "mutpb": {"type": "float", "low": 0.05, "high": 0.4, "fixed": False},
    "tournsize": {"type": "categorical", "choices": [3, 5, 7], "fixed": False}
}

# Create optimizer
optimizer = DHPCTOptimizer(
    parameters=parameters,
    n_trials=30,
    timeout=3600,  # 1 hour max
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    study_name="lunar_lander_optimization",
    storage="sqlite:///optuna.db"
)

# Define objective with budget
budget = {
    "generations": 50,
    "max_time": 300,  # 5 min per trial
    "nevals": 3
}
optimizer.define_objective(template, fitness_function, budget, minimize=False)

# Run optimization
study = optimizer.run_optimization(verbose=True)

# Get results
best_params = optimizer.get_best_params()
print(f"Best parameters: {best_params}")
print(f"Best fitness: {optimizer.get_best_value()}")

# Visualize
optimizer.visualize_results(
    plots=["importance", "history", "parallel_coordinate"],
    save_dir="results/optimization"
)

# Save results
optimizer.save_results("results/optimization")
```

## Integration with DHPCTEvolver

After optimization, use best parameters to run final evolution:

```python
best = optimizer.get_best_params()

final_evolver = DHPCTEvolver(
    pop_size=best["pop_size"],
    generations=200,  # More generations for final run
    cxpb=best["cxpb"],
    mutpb=best["mutpb"],
    tournsize=best["tournsize"],
    parallel=True,
    use_comet=True
)

final_evolver.setup_evolution(template, fitness_function)
best_individual, stats = final_evolver.run_evolution()
```

## Related Requirements

- FR-035: Flexible parameter specification with fixed/variable flags
- FR-036: Initialize with n_trials, timeout, pruner, sampler, study_name, storage
- FR-037: define_objective() with template, fitness, budget
- FR-038: run_optimization() executes Optuna search
- FR-039: get_best_params() retrieves optimal combination
- FR-040: visualize_results() generates plots
- FR-041: save_results() persists study data
