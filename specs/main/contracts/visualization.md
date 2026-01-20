# API Contract: Visualization Functions

**Module**: `dpct.visualization`  
**Description**: Functions for visualizing hierarchies, execution history, and network structures

## Network Visualization Functions

### visualize_hierarchy_layers

```python
def visualize_hierarchy_layers(
    individual: DHPCTIndividual,
    layout: str = "hierarchical",
    show_weights: bool = False,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8)
) -> None
```

**Description**: Display full layer-level network diagram showing all layers, nodes, and connections

**Parameters**:
- `individual`: Compiled DHPCTIndividual to visualize
- `layout`: Layout algorithm ("hierarchical", "spring", "circular")
- `show_weights`: Whether to display weight values on edges
- `save_path`: If provided, save plot to file instead of displaying
- `figsize`: Figure size in inches (width, height)

**Visualization**:
- Nodes: All layers (PL##, RL##, CL##, OL##, Observations, Actions, Errors)
- Edges: Connections between layers
- Colors: Different colors for layer types (perception, reference, comparator, output)
- Labels: Layer names and optionally weight values

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `RuntimeError`: If individual not compiled
- `ImportError`: If networkx not available

**Related**: FR-019a

---

### visualize_pct_units

```python
def visualize_pct_units(
    individual: DHPCTIndividual,
    layout: str = "hierarchical",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6)
) -> None
```

**Description**: Display PCT control units as single nodes (combining perception, reference, comparator, output)

**Parameters**:
- `individual`: Compiled DHPCTIndividual to visualize
- `layout`: Layout algorithm
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- Nodes: One node per control unit per level (e.g., "Level 0 Unit 0")
- Edges: Connections between units
- Hierarchy: Clear level structure from observations → Level 0 → Level 1 → ... → References

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `RuntimeError`: If individual not compiled
- `ImportError`: If networkx not available

**Related**: FR-019b

---

### visualize_weighted_network

```python
def visualize_weighted_network(
    individual: DHPCTIndividual,
    layout: str = "hierarchical",
    weight_threshold: float | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 10)
) -> None
```

**Description**: Display hierarchy network with weight values shown on connections

**Parameters**:
- `individual`: Compiled DHPCTIndividual to visualize
- `layout`: Layout algorithm
- `weight_threshold`: If provided, only show connections with |weight| > threshold
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- Nodes: Layers or units
- Edges: Connections with weight values as labels
- Edge width: Proportional to absolute weight value
- Edge color: Positive weights (blue) vs negative weights (red)

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `RuntimeError`: If individual not compiled
- `ImportError`: If networkx not available

**Related**: FR-019c

---

## Execution History Visualization

### visualize_execution_history

```python
def visualize_execution_history(
    history: ExecutionHistory,
    metrics: list[str] = ["observations", "actions", "errors"],
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8)
) -> None
```

**Description**: Plot time series of observations, actions, and errors during execution

**Parameters**:
- `history`: ExecutionHistory object from individual.run(record_history=True)
- `metrics`: Which metrics to plot ("observations", "actions", "errors", "rewards", "layer_NAME")
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- Multiple subplots, one per metric
- X-axis: Time step
- Y-axis: Metric values
- Legend: Dimension names if multi-dimensional

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `ValueError`: If history empty or metric unknown

**Related**: FR-009c

---

### visualize_layer_activations

```python
def visualize_layer_activations(
    history: ExecutionHistory,
    layer_names: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 6)
) -> None
```

**Description**: Plot activation values for specific layers over time

**Parameters**:
- `history`: ExecutionHistory object
- `layer_names`: List of layer names to plot (e.g., ["PL00", "CL01"])
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- Heatmap or line plot showing layer activations over time
- Different colors for different units within layer

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `ValueError`: If layer_names not in history

**Related**: FR-009c

---

### visualize_error_propagation

```python
def visualize_error_propagation(
    history: ExecutionHistory,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 6)
) -> None
```

**Description**: Visualize how error signals propagate across hierarchy levels

**Parameters**:
- `history`: ExecutionHistory object
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- Stacked plot or heatmap showing error magnitudes at each level
- Highlights error flow from low to high levels

**Side Effects**:
- Displays matplotlib plot or saves to file

---

## Evolution Visualization

### plot_evolution_progress

```python
def plot_evolution_progress(
    statistics: list[dict],
    metrics: list[str] = ["min", "mean", "max"],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6)
) -> None
```

**Description**: Plot evolution fitness over generations

**Parameters**:
- `statistics`: List of GenerationStatistics dicts from evolver
- `metrics`: Which metrics to plot ("min", "mean", "max", "std")
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Visualization**:
- X-axis: Generation number
- Y-axis: Fitness
- Multiple lines for different metrics
- Shaded region for std if included

**Side Effects**:
- Displays matplotlib plot or saves to file

**Raises**:
- `ValueError`: If statistics empty or metric unknown

---

### plot_parameter_importance

```python
def plot_parameter_importance(
    study: optuna.Study,
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 6)
) -> None
```

**Description**: Wrapper for Optuna parameter importance visualization

**Parameters**:
- `study`: Optuna Study object from optimizer
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Side Effects**:
- Displays matplotlib plot or saves to file
- Uses optuna.visualization.plot_param_importances()

**Related**: FR-040

---

### plot_optimization_history

```python
def plot_optimization_history(
    study: optuna.Study,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6)
) -> None
```

**Description**: Wrapper for Optuna optimization history visualization

**Parameters**:
- `study`: Optuna Study object
- `save_path`: If provided, save plot to file
- `figsize`: Figure size

**Side Effects**:
- Displays matplotlib plot or saves to file
- Uses optuna.visualization.plot_optimization_history()

**Related**: FR-040

---

## Usage Examples

```python
# Visualize hierarchy structure
individual = DHPCTIndividual("CartPole-v1", [4, 3, 2])
individual.compile()

# Full layer view
visualize_hierarchy_layers(individual, layout="hierarchical")

# PCT unit view
visualize_pct_units(individual, layout="hierarchical")

# Weighted network
visualize_weighted_network(individual, weight_threshold=0.1)

# Run and visualize execution
fitness = individual.run(steps=500, record_history=True)
visualize_execution_history(
    individual.history,
    metrics=["observations", "actions", "errors"]
)

# Visualize specific layers
visualize_layer_activations(
    individual.history,
    layer_names=["PL00", "CL00", "OL00"]
)

# Evolution progress
evolver = DHPCTEvolver(pop_size=50, generations=100)
evolver.setup_evolution(individual, fitness_fn)
best, stats = evolver.run_evolution()

plot_evolution_progress(stats, metrics=["min", "mean", "max"])

# Optimization results
optimizer = DHPCTOptimizer(parameters, n_trials=20)
optimizer.define_objective(individual, fitness_fn, budget)
study = optimizer.run_optimization()

plot_parameter_importance(study)
plot_optimization_history(study)
```

## Related Requirements

- FR-009c: Execution history visualization
- FR-019a: Layer-level network diagram
- FR-019b: PCT unit network diagram
- FR-019c: Weighted network diagram
- FR-040: Optimization result visualization
