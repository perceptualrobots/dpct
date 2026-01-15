# Enhanced Project Requirements Document for DPCT Library

## 1. Project Overview

**Deep Perceptual Control Theory (DPCT)** is a Python library designed to:
- Evolve and train hierarchical control systems based on Perceptual Control Theory (PCT)
- Represent these control systems using TensorFlow/Keras models
- Optimize these systems through evolutionary algorithms using DEAP

This library merges concepts from Deep Learning and PCT to create adaptable control hierarchies that can interface with various environments.

## 2. Development Environment & Structure

### 2.1 Development Framework
- All code must be written using the **nbdev** framework (https://nbdev.fast.ai)
- Development happens in Jupyter notebooks located in the `nbs/` directory
- Code will be automatically exported to Python modules via nbdev directives

### 2.2 File Naming Conventions
- Notebook filenames follow pattern: `NN_packagename.ipynb` where NN is a two-digit number
- For each package, create three notebooks:
  1. `NN_packagename.ipynb` - Main implementation code
  2. `NN_packagename_usage.ipynb` - Usage examples with `#gui` directive in each code cell
  3. `NN_packagename_unittests.ipynb` - Unit tests for the package

### 2.3 Required Dependencies
```
pip install nbdev tensorflow gymnasium deap numpy matplotlib optuna
```

### 2.4 Development Workflow
1. Create notebook files following naming convention
2. Develop code using nbdev directives (`#export`, `#hide`, etc.)
3. Run `nbdev_prepare` to validate and export code
4. Fix any errors reported by nbdev_prepare
5. Use `nbdev_build_lib` to build the library
6. Use `nbdev_build_docs` to generate documentation

## 3. Core Components

An individual represents a PCT control hierarchy and its environment.

It should include the ability to evolve by mating and mutation.

It should also be able to compute a fitness value when run in its environment.

### 3.1 DHPCTIndividual Class

#### 3.1.1 Overview
`DHPCTIndividual` encapsulates:
- An environment (e.g., OpenAI Gym (Gymnasium)) that provides sensory observations and receives actions
- A Keras model representing a PCT hierarchy with multiple control levels

#### 3.1.2 Class Structure
```python
class DHPCTIndividual:    def __init__(self, env_name, env_props=None, levels=None, activation_funcs=None, weight_types=None, input_references=None):
        """
        Initialize a new individual with environment and hierarchy specifications.
        
        Parameters:
        - env_name: String identifier for the environment (e.g. 'CartPole-v1')
        - env_props: Additional environment properties
        - levels: Array of column sizes for each level in the hierarchy
        - activation_funcs: Dict mapping levels to activation functions
        - weight_types: Dict specifying weight variable types
        - input_references: a list of values for the fixed reference input layer
        """
        pass
        
    @classmethod
    def from_config(cls, config_dict):
        """Create an individual from a configuration dictionary"""
        pass
        
    def compile(self):
        """Build the environment and Keras model"""
        pass
        
    def config(self):
        """
        Return a dictionary of the individual's properties
        
        Returns:
        - dict: Complete configuration dictionary with environment, hierarchy, and weight data
        """
        pass
        
    def save_config(self, filepath):
        """
        Save the individual's configuration to a JSON file
        
        Parameters:
        - filepath: Path where the configuration will be saved
        
        Returns:
        - bool: True if save was successful, False otherwise
        """
        pass
        
    def run(self, steps, train=False, early_termination=False):
        """
        Run the individual in its environment
        
        Parameters:
        - steps: Number of timesteps to run
        - train: Whether to enable online learning during execution
        - early_termination: Whether to terminate early based on environment signals
        """
        pass
        
    # Evolution methods
    def mate(self, other):
        """Create two new individuals by crossing this one with another"""
        pass
        
    def mutate(self, struct_prob=0.1, weight_prob=0.1):
        """Mutate this individual's structure and/or weights"""
        pass
        
    def evaluate(self, nevals=1):
        """Evaluate this individual's fitness"""
        pass
```

#### 3.1.3 Hierarchical Structure

A hierarchy consists of a number of control units. Each unit comprises a reference, perception, comparator and output function. A comparator function is the perception subtracted from the reference function. An output function is the weighted value of the comparator value. Reference and perception functions are weighted sums (by default, though could also exponential smoothed weighted sums) of their inputs , which come from outside the unit.

Each level in the hierarchy can have one or more control units. At the lowest level the perception function takes its inputs from the environment observations. At each subsequent level the perception function takes its inputs from the values of all the perception functions of the lower level.

At the highest level the reference values are from the external reference inputs. At each lower level the inputs to a reference function are all the valves of the output functions from the level above.

At each level the comparator value for each unit is the reference minus the perception value for that unit.

At each level the output function is a weighted value of the comparator value.

The outputs from the lowest level are the action values applied to the environment.

A hierarchy includes:
- **Perceptual Inputs**: Environment observations fed into the first level
- **Reference Inputs**: External inputs to higher levels
- **Multiple Levels**: Each with perception, reference, comparator, and output layers
- **Actions**: Outputs from the lowest level affecting the environment
- **Errors**: Collection of all comparator values across levels

A keras layer is the equivalent of a PCT function. So a layer could be a weighted sum , a smoothed weighted sum , a weighted value or a subtraction.

Create functionality to create the different types of layers. 

Test them against the functions from the pct Python library for Weighted Sum, SmoothWeightedSum and Subtract.


#### 3.1.4 Model Creation Specification
The Keras model must follow this structure:
- Level 0 (lowest): Perception layer = weighted sum of inputs
- Higher levels: Perception layer = weighted sum of perception layer below
- Top level: Comparator = reference inputs - perception layer
- Other levels: Comparator = reference layer - perception layer
- Reference layers: Weighted sum of output layer from above
- Output layers: Element-wise multiplication of weights and comparator values
- Model outputs: Actions (from level 0) and errors (all comparators)


Create functionality for creating keras model hierarchies with an environment and arbitrary levels and units.

Define a JSON configuration for the model.

Define a networkx model of the nodes and layers of the model.

Create a model from the pct hierarchy format.

Tests- test the following for a DHPCT Individual and an equivalent pct hierarchy and compare

- a system with a single level and single unit
- a system with a single level and two units
- a system with a two levels and two units
- a system created from a pct configuration

Confirm that the new individual will run environments such as CartPole and Lunar Lander.


### 3.2 DHPCTEvolver Class

#### 3.2.1 Overview
`DHPCTEvolver` configures and runs evolutionary optimization of `DHPCTIndividual` populations using DEAP.

#### 3.2.2 Class Structure
```python
class DHPCTEvolver:
    def __init__(self, 
                 pop_size=50, 
                 generations=100,
                 evolve_termination=None,
                 evolve_static_termination=None,
                 unchanged_generations=5,
                 run_best=True,
                 save_arch_best=True,
                 save_arch_all=False):
        """
        Initialize the evolver with configuration parameters.
        """
        pass
        
    def setup_evolution(self, template_individual, fitness_function, minimize=True):
        """Configure DEAP toolbox with evolutionary operators"""
        pass
        
    def run_evolution(self, verbose=True):
        """Run the evolutionary process"""
        pass
        
    def save_results(self, path):
        """Save evolution results and configurations"""
        pass
```

#### 3.2.3 Evolution Features
- Generation statistics: min/mean/max fitness, mutation percentages, time
- Option to evaluate and display best individual each generation
- Save configurations (best or all individuals) each generation
- Early termination based on fitness target or stagnation
- Detailed logging and visualization of evolutionary progress

### 3.3 DHPCTOptimizer Class

#### 3.3.1 Overview
`DHPCTOptimizer` uses Optuna to perform hyperparameter optimization on the evolution process, allowing for efficient search of optimal evolutionary parameters.

#### 3.3.2 Class Structure
```python
class DHPCTOptimizer:
    def __init__(self, 
                 evolution_params, 
                 n_trials=100, 
                 timeout=None, 
                 pruner=None, 
                 sampler=None,
                 study_name=None,
                 storage=None):
        """
        Initialize the optimizer with configuration parameters.
        
        Parameters:
        - evolution_params: Dictionary of parameters for evolution, each with a 'fixed' flag
                           If fixed=True, the parameter is not optimized and the provided value is used
                           If fixed=False, the parameter is optimized within the specified range
        - n_trials: Number of optimization trials to run
        - timeout: Maximum time for optimization in seconds
        - pruner: Optuna pruner instance
        - sampler: Optuna sampler instance
        - study_name: Name for the Optuna study
        - storage: Optuna storage URL
        """
        pass
        
    def define_objective(self, template_individual, fitness_function, evaluation_budget=None):
        """Define the objective function for Optuna to optimize"""
        pass
        
    def run_optimization(self, verbose=True):
        """Run the hyperparameter optimization process"""
        pass
        
    def get_best_params(self):
        """Get the best parameters from the optimization"""
        pass
        
    def visualize_results(self):
        """Visualize the optimization results"""
        pass
        
    def save_results(self, path):
        """Save optimization results"""
        pass
```

#### 3.3.3 Optimization Features
- Flexible parameter specification with fixed/variable flags
- Support for various parameter types: continuous, discrete, categorical
- Parameter constraints and conditional parameters
- Automatic pruning of unpromising trials
- Visualization of parameter importance and optimization history
- Integration with TPE, CMA-ES, and other Optuna samplers

### 3.4 Online Learning

#### 3.4.1 Overview
The DPCT library supports online learning capabilities, allowing evolved individuals to be further trained using Keras while interacting with their environments.

#### 3.4.2 Implementation Details
- Extended `DHPCTIndividual.run()` method with online learning capabilities
- Comparator values used as training outputs with target values of zero
- Implementation of custom Keras callbacks for online learning
- Configurable learning rate and training frequency

#### 3.4.3 Online Learning Configuration
```python
online_learning_config = {
    "enabled": True,
    "learning_rate": 0.01,
    "optimizer": "adam",
    "batch_size": 32,
    "train_every_n_steps": 10,
    "error_weight_coefficients": {
        "level_0": 1.0,
        "level_1": 0.5,
        "level_2": 0.1
    }
}
```

## 4. Implementation Details

### 4.1 DHPCTIndividual Implementation Notes
- Use Keras Functional API for model creation
- Default activation function: linear
- Default weight type: float - other types are boolean and ternary
- Mate operation: Use DEAP crossover for structure, blend weights
- Mutation operation: Structure mutation + Weight mutation 
- Evaluation: Run multiple times and choose average or maximum fitness

#### 4.1.1 Keras Model structure

- The default activations are linear unless specified in self.activation_funcs
- The obs_input input layer takes the shape of obs_space
- The ref_input input layer takes the shape of the input_references or the shape of the top level
- For each level the structure is as follows:
-- A perception layer is the weighted sum of perception layer below, except for level 0 where the perception layer is the weighted sum of the observations input layer
-- A reference layer is the weighted sum of output layer from above, except for the top layer where the reference layer is reference input layer
-- A comparator layer = reference layer - perception layer
-- An output layer is the element-wise multiplication of weights and comparator layer
- The actions layer is the weighted sum of the level 0 output layer

#### 4.1.2 Model Naming Conventions

The naming of the model layers should follow this convention:
- The letter P, R, C or O if the layer is a perception, reference, comparator or output
- The letter L for level
- Two digits for the level, left padded with zero
- The InputLayer should just be called Observations
- The action outputs should just be called Actions
- The output array of all comparator values should be called Errors



### 4.2 Example Configuration Dictionary
```python
config = {
    "env": {
        "env_name": "CartPole-v1",
        "properties": {"render_mode": "rgb_array"}
    },
    "hierarchy": {
        "levels": [4, 3, 2],  # Column sizes for each level
        "activation_funcs": {
            0: "linear",
            1: "relu", 
            2: "tanh"
        },
        "weight_types": {
            "all": "float",
            "level_1_perception": "boolean"
        }
    },
    "weights": {
        # Nested dictionary of weight values
    }
}
```

### 4.3 Configuration Usage Example
```python
# Create an individual from a configuration file
with open('best_individual.json', 'r') as f:
    config_dict = json.load(f)
individual = DHPCTIndividual.from_config(config_dict)

# Run the individual
fitness = individual.run(steps=1000)

# Modify the individual
individual.mutate(weight_prob=0.2)

# Get the updated configuration
new_config = individual.config()

# Save the configuration for later use
individual.save_config('modified_individual.json')
```

## 5. Verification and Testing

### 5.1 Development Validation
- Run `nbdev_prepare` after creating or modifying each notebook
- Fix any reported errors before proceeding
- Ensure all exported functions have docstrings

### 5.2 Unit Testing
- Create comprehensive tests in the `*_unittests.ipynb` notebooks
- Test individual components and integration
- Verify deterministic behavior with fixed random seeds

### 5.3 Usage Examples
- Provide clear examples in `*_usage.ipynb` notebooks
- Include visualization of hierarchy structure
- Demonstrate evolution process with simple environments

## 6. Appendix: Implementation Timeline

1. Core functionality: Basic DHPCTIndividual class
2. Environment integration with OpenAI Gym (Gymnasium)
3. Keras model creation and structure
4. Evolution operators (mate, mutate)
5. DHPCTEvolver implementation
6. Documentation and examples
7. Performance optimization
