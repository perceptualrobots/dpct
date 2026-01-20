# Quick Start Guide: DPCT Core Library

**Purpose**: Get started with Deep Perceptual Control Theory library in 5 minutes

## Installation

```bash
pip install dpct
```

Or for development:
```bash
git clone https://github.com/yourusername/dpct.git
cd dpct
pip install -e ".[dev]"
```

## Basic Concepts

The DPCT library provides three main components:

1. **DHPCTIndividual**: A hierarchical control system that interacts with environments
2. **DHPCTEvolver**: Evolutionary optimization of populations of individuals
3. **DHPCTOptimizer**: Hyperparameter tuning for evolution settings

## Example 1: Create and Run a Hierarchy

```python
from dpct.core import DHPCTIndividual

# Create a 3-level hierarchy for CartPole environment
individual = DHPCTIndividual(
    env_name="CartPole-v1",
    levels=[4, 3, 2],  # 4 units at bottom, 3 in middle, 2 at top
    activation_funcs="linear",
    weight_types="float",
    random_seed=42
)

# Compile the Keras model
individual.compile()

# Run in environment for 500 steps
fitness = individual.run(steps=500, early_termination=True)
print(f"Fitness: {fitness}")

# Save configuration
individual.save_config("my_individual.json")
```

**Expected Output**: Fitness around 20-200 (random weights, not optimized)

## Example 2: Save and Load Configurations

```python
from dpct.core import DHPCTIndividual

# Load from saved configuration
loaded_individual = DHPCTIndividual.from_config(
    DHPCTIndividual.load_config("my_individual.json")
)

# Run and verify same behavior (with same seed)
fitness = loaded_individual.run(steps=500)
print(f"Loaded individual fitness: {fitness}")
```

## Example 3: Manual Evolution Operations

```python
from dpct.core import DHPCTIndividual

# Create two parent individuals
parent1 = DHPCTIndividual("CartPole-v1", [4, 3, 2], random_seed=1)
parent1.compile()

parent2 = DHPCTIndividual("CartPole-v1", [4, 3, 2], random_seed=2)
parent2.compile()

# Create offspring via crossover
child1, child2 = parent1.mate(parent2)

# Mutate offspring
child1.mutate(weight_prob=0.2)

# Evaluate
fitness1 = child1.evaluate(nevals=3, aggregate="mean")
print(f"Child 1 fitness (3 runs, mean): {fitness1}")
```

## Example 4: Evolve a Population

```python
from dpct.core import DHPCTIndividual
from dpct.evolver import DHPCTEvolver

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

# Create evolver
evolver = DHPCTEvolver(
    pop_size=30,
    generations=20,
    cxpb=0.6,
    mutpb=0.2,
    parallel=True  # Use multiprocessing
)

# Setup and run evolution
evolver.setup_evolution(template, fitness_function, minimize=False)
best, stats = evolver.run_evolution(verbose=True)

print(f"\nBest individual fitness: {best.fitness}")

# Save results
evolver.save_results("results/cartpole_evolution")
```

**Expected Output**: Fitness improving over generations, best around 200-500 after 20 generations

## Example 5: Optimize Hyperparameters

```python
from dpct.core import DHPCTIndividual
from dpct.optimizer import DHPCTOptimizer
import optuna

# Create template
template = DHPCTIndividual("CartPole-v1", [4, 3, 2], random_seed=42)
template.compile()

def fitness_function(individual):
    return individual.evaluate(nevals=2, aggregate="mean")

# Define parameter search space
parameters = {
    "pop_size": {"type": "int", "low": 20, "high": 50, "fixed": False},
    "generations": {"type": "int", "value": 15, "fixed": True},
    "cxpb": {"type": "float", "low": 0.4, "high": 0.9, "fixed": False},
    "mutpb": {"type": "float", "low": 0.1, "high": 0.4, "fixed": False},
}

# Create optimizer
optimizer = DHPCTOptimizer(
    parameters=parameters,
    n_trials=10,
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Define objective with budget
budget = {"generations": 15, "nevals": 2}
optimizer.define_objective(template, fitness_function, budget)

# Run optimization
study = optimizer.run_optimization(verbose=True)

# Get best parameters
best_params = optimizer.get_best_params()
print(f"\nBest parameters: {best_params}")
print(f"Best fitness: {optimizer.get_best_value()}")
```

## Example 6: Execution History and Visualization

```python
from dpct.core import DHPCTIndividual
from dpct.visualization import visualize_execution_history, visualize_hierarchy_layers

# Create and run individual with history recording
individual = DHPCTIndividual("CartPole-v1", [4, 3, 2])
individual.compile()

fitness = individual.run(steps=500, record_history=True)

# Visualize execution history
visualize_execution_history(
    individual.history,
    metrics=["observations", "actions", "errors"]
)

# Visualize network structure
visualize_hierarchy_layers(individual, layout="hierarchical")
```

## Example 7: Random Structure Evolution

```python
from dpct.core import DHPCTIndividual
from dpct.evolver import DHPCTEvolver

# Template with minimal configuration
template = DHPCTIndividual("CartPole-v1", [4], random_seed=42)
template.compile()

# Evolver with random structure initialization
evolver = DHPCTEvolver(
    pop_size=30,
    generations=25,
    random_structure={
        'min_levels': 2,
        'max_levels': 4,
        'min_units': 2,
        'max_units': 8
    }
)

def fitness_fn(ind):
    return ind.run(steps=500)

evolver.setup_evolution(template, fitness_fn)
best, stats = evolver.run_evolution()

print(f"Best structure: {best.levels}")
print(f"Best fitness: {best.fitness}")
```

## Example 8: Fixed Weights for Transfer Learning

```python
from dpct.core import DHPCTIndividual

# Load pre-trained individual
pretrained = DHPCTIndividual.from_config("pretrained.json")

# Create new individual with first level fixed
individual = DHPCTIndividual(
    env_name="CartPole-v1",
    levels=[4, 3, 2],
    fixed_levels={0}  # Level 0 won't be mutated
)
individual.compile()

# Copy weights from pretrained to first level
individual.set_level_weights(0, pretrained.get_level_weights(0))

# Now evolve only higher levels
evolver = DHPCTEvolver(pop_size=20, generations=15)
evolver.setup_evolution(individual, fitness_fn)
best, stats = evolver.run_evolution()
```

## Example 9: Comet.ml Integration

```python
from dpct.evolver import DHPCTEvolver

evolver = DHPCTEvolver(
    pop_size=30,
    generations=50,
    use_comet=True,
    comet_config={
        'project_name': 'dpct-experiments',
        'workspace': 'your-workspace',
        'api_key': 'your-api-key'
    }
)

evolver.setup_evolution(template, fitness_function)
best, stats = evolver.run_evolution()

# Evolution metrics automatically logged to Comet.ml
```

## Example 10: LunarLanderContinuous Environment

```python
from dpct.core import DHPCTIndividual
from dpct.evolver import DHPCTEvolver

# LunarLander has 8 observations and 2 continuous actions
template = DHPCTIndividual(
    env_name="LunarLanderContinuous-v2",
    levels=[8, 6, 4],  # Larger hierarchy for complex environment
    random_seed=42
)
template.compile()

def fitness_fn(individual):
    return individual.run(steps=1000, early_termination=True)

evolver = DHPCTEvolver(
    pop_size=50,
    generations=100,
    parallel=True,
    save_arch_best=True
)

evolver.setup_evolution(template, fitness_fn)
best, stats = evolver.run_evolution()

print(f"Best fitness: {best.fitness}")
best.save_config("lunar_lander_best.json")
```

**Expected Outcome**: Solving LunarLander (fitness > 200) typically requires 100+ generations

## Common Patterns

### Pattern 1: Quick Experiment

```python
# Single script for quick testing
individual = DHPCTIndividual("CartPole-v1", [4, 3])
individual.compile()
print(individual.run(steps=500))
```

### Pattern 2: Evolution with Best Individual Tracking

```python
evolver = DHPCTEvolver(
    pop_size=30,
    generations=50,
    save_arch_best=True,  # Save best each generation
    run_best=True         # Display best performance
)
evolver.setup_evolution(template, fitness_fn)
best, stats = evolver.run_evolution()
```

### Pattern 3: Hyperparameter Optimization → Final Evolution

```python
# Step 1: Find best parameters (fast, small generations)
optimizer = DHPCTOptimizer(parameters, n_trials=20)
optimizer.define_objective(template, fitness_fn, {"generations": 15})
study = optimizer.run_optimization()
best_params = optimizer.get_best_params()

# Step 2: Final evolution with best parameters (slow, many generations)
final_evolver = DHPCTEvolver(**best_params, generations=200)
final_evolver.setup_evolution(template, fitness_fn)
champion, stats = final_evolver.run_evolution()
```

## Troubleshooting

### Issue: "Model not compiled"
**Solution**: Call `individual.compile()` before `run()`, `mate()`, or `mutate()`

### Issue: Low fitness / Not learning
**Solutions**:
- Increase population size and generations
- Try different weight types ("float", "boolean", "ternary")
- Adjust mutation/crossover probabilities
- Use hyperparameter optimization to find good settings

### Issue: Evolution is slow
**Solutions**:
- Enable parallelization: `evolver = DHPCTEvolver(parallel=True)`
- Reduce population size or generations for testing
- Use shorter evaluation runs (fewer steps)

### Issue: "Environment not found"
**Solution**: Install gymnasium environments: `pip install gymnasium[all]`

## Next Steps

- Read full API documentation in `contracts/` directory
- Explore example notebooks in `nbs/` directory
- Check `PROJECT_REQUIREMENTS.md` for implementation details
- Review `data-model.md` for entity relationships
- See `research.md` for technical design decisions

## Quick Reference

```python
# Create individual
ind = DHPCTIndividual(env, levels, seed=42)
ind.compile()

# Run
fitness = ind.run(steps=500)

# Save/load
ind.save_config("file.json")
ind2 = DHPCTIndividual.from_config(config_dict)

# Evolution
evolver = DHPCTEvolver(pop_size=30, generations=20)
evolver.setup_evolution(template, fitness_fn)
best, stats = evolver.run_evolution()

# Optimization
optimizer = DHPCTOptimizer(params, n_trials=10)
optimizer.define_objective(template, fitness_fn, budget)
study = optimizer.run_optimization()
best_params = optimizer.get_best_params()
```

Happy evolving! 🧬🤖
