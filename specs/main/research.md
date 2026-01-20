# Research: DPCT Core Library

**Date**: 2026-01-20  
**Purpose**: Resolve technical unknowns and establish best practices for implementation

## Technology Decisions

### 1. Keras Functional API for Hierarchical Models

**Decision**: Use TensorFlow's Keras Functional API for building PCT hierarchies

**Rationale**:
- Functional API supports complex multi-input/multi-output models needed for PCT architecture
- Allows explicit definition of layer connections (perception → comparator ← reference)
- Enables named layers following convention (PL##, RL##, CL##, OL##)
- Supports custom layer operations (element-wise subtraction for comparators, multiplication for outputs)
- Compatible with model serialization via `get_weights()`/`set_weights()`
- Well-documented and stable API

**Alternatives Considered**:
- **Sequential API**: Rejected - cannot handle multiple inputs/outputs or complex connection patterns
- **PyTorch**: Rejected - would require complete rewrite of existing infrastructure
- **Custom neural network from scratch**: Rejected - reinventing the wheel, loses ecosystem benefits

**Implementation Approach**:
```python
# Pseudo-code structure
observations = Input(shape=obs_dim, name='Observations')
references_input = Input(shape=ref_dim, name='ReferencesInput')

# For each level (from bottom to top):
#   perception = Dense(..., name=f'PL{level}')(perception_inputs)
#   reference = Dense(..., name=f'RL{level}')(reference_inputs)
#   comparator = Subtract(name=f'CL{level}')([reference, perception])
#   output = Multiply(name=f'OL{level}')([weights, comparator])

model = Model(inputs=[observations, references_input], 
              outputs=[actions, errors])
```

### 2. DEAP for Evolutionary Algorithms

**Decision**: Use DEAP (Distributed Evolutionary Algorithms in Python) library

**Rationale**:
- Provides toolbox pattern for flexible evolutionary algorithm configuration
- Supports standard genetic operators: selection, crossover, mutation
- Built-in statistics tracking (min, mean, max, std)
- Mature library with extensive documentation
- Compatible with custom fitness functions and individual representations
- Supports parallelization via `multiprocessing` or `scoop`

**Alternatives Considered**:
- **Custom evolution from scratch**: Rejected - complex to implement correctly, DEAP is well-tested
- **PyGAD**: Rejected - less flexible for custom individual representations
- **NEAT-Python**: Rejected - specialized for topology evolution, not weight-focused evolution

**Implementation Approach**:
```python
from deap import base, creator, tools, algorithms

# Define fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", DHPCTIndividual, fitness=creator.FitnessMax)

# Configure toolbox
toolbox = base.Toolbox()
toolbox.register("individual", create_individual, template)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", crossover_operator)
toolbox.register("mutate", mutation_operator)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run evolution
pop = toolbox.population(n=pop_size)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations)
```

### 3. Optuna for Hyperparameter Optimization

**Decision**: Use Optuna for automated hyperparameter tuning

**Rationale**:
- Supports multiple sampling strategies (TPE, CMA-ES, Grid, Random)
- Built-in pruning for early stopping of unpromising trials
- Visualization functions for parameter importance and optimization history
- Persistent storage via RDB backend or in-memory
- Clean API for defining search spaces
- Compatible with parallel execution

**Alternatives Considered**:
- **Hyperopt**: Rejected - less active development, less user-friendly API
- **Grid search**: Rejected - not scalable for high-dimensional search spaces
- **Ray Tune**: Rejected - heavier dependency, overkill for this use case

**Implementation Approach**:
```python
import optuna

def objective(trial):
    # Suggest parameters
    pop_size = trial.suggest_int('pop_size', 20, 100)
    mut_rate = trial.suggest_float('mut_rate', 0.05, 0.3)
    
    # Run evolution with these parameters
    evolver = DHPCTEvolver(pop_size=pop_size, mutation_rate=mut_rate)
    best_fitness = evolver.run_evolution()
    
    return best_fitness

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 4. Gymnasium Environment Interface

**Decision**: Support Gymnasium (OpenAI Gym successor) and compatible interfaces

**Rationale**:
- Gymnasium is the actively maintained successor to OpenAI Gym
- Standard interface: `reset()`, `step(action)`, `render()`, `close()`
- Large ecosystem of environments (CartPole, LunarLander, Atari, etc.)
- Any custom environment can implement the same interface
- Version 0.26+ has improved typing and API consistency

**Alternatives Considered**:
- **OpenAI Gym**: Rejected - deprecated, no longer maintained
- **Custom environment interface**: Rejected - limits compatibility with existing environments

**Implementation Approach**:
```python
import gymnasium as gym

env = gym.make('CartPole-v1')
obs, info = env.reset(seed=42)

for step in range(steps):
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    if done and early_termination:
        break
    elif done:
        obs, info = env.reset()
```

### 5. Parallelization Strategy

**Decision**: Use Python's `multiprocessing.Pool` for parallel fitness evaluations

**Rationale**:
- CPU-bound fitness evaluations benefit from true parallelism (not threading)
- Compatible with DEAP's `toolbox.map` pattern
- Simple to implement: `toolbox.register("map", pool.map)`
- Works across Windows, Linux, macOS (with proper `if __name__ == '__main__'` guards)
- Can limit to `cpu_count()` or user-specified number of processes

**Alternatives Considered**:
- **Threading**: Rejected - GIL limits parallelism for CPU-bound tasks
- **Ray**: Rejected - heavy dependency for simple parallelization
- **Dask**: Rejected - overkill for this use case
- **SCOOP**: Considered - DEAP integration, but multiprocessing is simpler

**Implementation Approach**:
```python
from multiprocessing import Pool, cpu_count

def parallel_evolution(toolbox, population, ngen, parallel=True):
    if parallel:
        with Pool(processes=cpu_count()) as pool:
            toolbox.register("map", pool.map)
            result = algorithms.eaSimple(population, toolbox, ...)
    else:
        result = algorithms.eaSimple(population, toolbox, ...)
    
    return result
```

### 6. Visualization Approaches

**Decision**: Use NetworkX + Matplotlib for network diagrams, Matplotlib for execution history

**Rationale**:
- NetworkX provides graph structures for representing hierarchy networks
- Matplotlib integration for rendering graphs with custom layouts
- Can extract layer information from Keras model and build graph representation
- Different visualization modes via different node/edge attributes
- Matplotlib sufficient for time-series plots of execution history

**Alternatives Considered**:
- **Plotly**: Considered for interactivity, but adds dependency complexity
- **Graphviz**: Rejected - requires external installation
- **Custom rendering**: Rejected - NetworkX handles layout algorithms well

**Implementation Approach**:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Layer view: full detail
G = nx.DiGraph()
for layer in model.layers:
    G.add_node(layer.name, type=extract_type(layer.name))
for connection in model.connections:
    G.add_edge(connection.source, connection.target)

# PCT unit view: collapse PL/RL/CL/OL into single nodes per level
# Weighted view: add edge labels with weight values

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.show()

# Execution history: time series plots
plt.plot(history['observations'])
plt.plot(history['actions'])
plt.plot(history['errors'])
plt.legend(['Observations', 'Actions', 'Errors'])
plt.show()
```

### 7. Configuration Serialization

**Decision**: Primary format is JSON, with pickle support for Python-specific serialization

**Rationale**:
- JSON is human-readable, editable, version-controllable, language-agnostic
- NumPy arrays can be converted to lists for JSON serialization
- Pickle handles Python objects natively (including NumPy arrays) but is Python-specific
- Both formats satisfy different use cases (JSON for sharing, pickle for Python-to-Python)

**Implementation Approach**:
```python
import json
import pickle
import numpy as np

def config_to_json(config):
    """Convert config dict to JSON, handling NumPy arrays"""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    return json.dumps(config, default=convert, indent=2)

def config_from_json(json_str):
    """Load config from JSON, converting lists back to arrays"""
    config = json.loads(json_str)
    # Convert weight lists back to NumPy arrays
    return config

def config_to_pickle(config, filepath):
    """Save config using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)

def config_from_pickle(filepath):
    """Load config from pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
```

### 8. Legacy Configuration Format

**Decision**: Provide adapter functions to convert to/from legacy format specification

**Rationale**:
- Maintains compatibility with existing systems
- Adapters isolate format differences from core logic
- Can be versioned independently if legacy format changes

**Implementation Approach**:
```python
def to_legacy_config(config):
    """Convert current config format to legacy format"""
    # Map field names, restructure data, handle differences
    legacy = {
        'old_field_name': config['new_field_name'],
        # ... mapping logic
    }
    return legacy

def from_legacy_config(legacy_config):
    """Convert legacy format to current config format"""
    config = {
        'new_field_name': legacy_config['old_field_name'],
        # ... reverse mapping logic
    }
    return config
```

Note: Actual legacy format specification needs to be provided before implementation.

### 9. Comet.ml Integration

**Decision**: Optional integration via environment variables or explicit configuration

**Rationale**:
- Comet.ml provides experiment tracking, metric logging, model versioning
- Optional dependency - library works without it
- Can log evolution progress, best individuals, hyperparameter searches
- Integrates via simple API: create experiment, log metrics, log models

**Implementation Approach**:
```python
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

class DHPCTEvolver:
    def __init__(self, ..., use_comet=False, comet_config=None):
        self.use_comet = use_comet and COMET_AVAILABLE
        
        if self.use_comet:
            self.experiment = Experiment(**comet_config)
    
    def run_evolution(self, ...):
        for gen in range(generations):
            # ... evolution logic
            
            if self.use_comet:
                self.experiment.log_metric("best_fitness", stats['max'], step=gen)
                self.experiment.log_metric("mean_fitness", stats['mean'], step=gen)
```

### 10. Weight Type Implementations

**Decision**: Use NumPy arrays with constraints for different weight types

**Rationale**:
- Float: standard NumPy float arrays
- Boolean: NumPy arrays with dtype=bool or constrained to {0, 1}
- Ternary: NumPy arrays constrained to {-1, 0, 1}
- Mutation operators respect constraints when modifying weights

**Implementation Approach**:
```python
def create_weights(shape, weight_type='float'):
    if weight_type == 'float':
        return np.random.randn(*shape)
    elif weight_type == 'boolean':
        return np.random.randint(0, 2, size=shape).astype(float)
    elif weight_type == 'ternary':
        return np.random.choice([-1, 0, 1], size=shape).astype(float)
    
def mutate_weights(weights, weight_type, prob=0.1):
    mask = np.random.random(weights.shape) < prob
    
    if weight_type == 'float':
        weights[mask] += np.random.randn(mask.sum())
    elif weight_type == 'boolean':
        weights[mask] = 1 - weights[mask]  # flip
    elif weight_type == 'ternary':
        weights[mask] = np.random.choice([-1, 0, 1], size=mask.sum())
    
    return weights
```

### 11. Random Structure Initialization

**Decision**: Parameterize level count and units per level with min/max bounds

**Rationale**:
- Supports evolutionary search over architectures, not just weights
- Min/max bounds prevent degenerate structures (too few levels/units)
- Random initialization explores structural diversity in initial population

**Implementation Approach**:
```python
def create_random_structure(min_levels=2, max_levels=5, 
                           min_units=2, max_units=10):
    num_levels = np.random.randint(min_levels, max_levels + 1)
    levels = [np.random.randint(min_units, max_units + 1) 
              for _ in range(num_levels)]
    return levels

# Evolution can initialize population with diverse structures
population = [DHPCTIndividual(env_name, create_random_structure()) 
              for _ in range(pop_size)]
```

### 12. Fixed Weights/Nodes/Levels

**Decision**: Use boolean masks or sets to mark elements as immutable during evolution

**Rationale**:
- Enables transfer learning (fix pre-trained layers, evolve new ones)
- Supports ablation studies (fix some weights, vary others)
- Can be specified at different granularities (weight-level, layer-level, level-level)

**Implementation Approach**:
```python
class DHPCTIndividual:
    def __init__(self, ..., fixed_weights=None, fixed_levels=None):
        self.fixed_weights = fixed_weights or set()  # set of layer names
        self.fixed_levels = fixed_levels or set()    # set of level indices
    
    def mutate(self, prob=0.1):
        for layer_name, weights in self.get_weights().items():
            # Skip if layer is marked as fixed
            if layer_name in self.fixed_weights:
                continue
            
            level = extract_level_from_name(layer_name)
            if level in self.fixed_levels:
                continue
            
            # Apply mutation
            mutated = mutate_weights(weights, self.weight_types[layer_name], prob)
            self.set_layer_weights(layer_name, mutated)
```

## Best Practices Summary

1. **Keras Model Building**: Use Functional API with explicit layer naming for traceability
2. **Evolution**: Leverage DEAP toolbox pattern for flexibility, use multiprocessing for parallelization
3. **Optimization**: Use Optuna TPE sampler as default, enable pruning for efficiency
4. **Environment Integration**: Always handle both `terminated` and `truncated` flags from Gymnasium
5. **Visualization**: Build NetworkX graphs from model structure, use Matplotlib for rendering
6. **Configuration**: Default to JSON for human readability, offer pickle for Python convenience
7. **Testing**: Use fixed random seeds for deterministic tests, test each weight type separately
8. **Nbdev Development**: Follow three-notebook pattern, use `#export` directives carefully
9. **Parallelization**: Wrap pool creation in `if __name__ == '__main__'` for Windows compatibility
10. **Optional Dependencies**: Gracefully degrade when optional packages (comet_ml, networkx) unavailable

## Unknowns Resolved

All technical unknowns from the specification have been resolved:
- ✅ Keras Functional API approach for hierarchical models
- ✅ DEAP configuration for evolution
- ✅ Optuna setup for hyperparameter optimization
- ✅ Gymnasium integration patterns
- ✅ Parallelization strategy
- ✅ Visualization implementation approach
- ✅ Configuration serialization formats
- ✅ Legacy format conversion strategy
- ✅ Random structure initialization
- ✅ Fixed weights implementation
- ✅ Weight type constraints
- ✅ Comet.ml integration approach

## Next Steps

Proceed to Phase 1:
1. Generate `data-model.md` with entity definitions
2. Generate `contracts/` with API specifications
3. Generate `quickstart.md` with usage examples
4. Update agent context with technologies from this research
