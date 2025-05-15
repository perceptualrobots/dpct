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
pip install nbdev tensorflow gym deap numpy matplotlib
```

### 2.4 Development Workflow
1. Create notebook files following naming convention
2. Develop code using nbdev directives (`#export`, `#hide`, etc.)
3. Run `nbdev_prepare` to validate and export code
4. Fix any errors reported by nbdev_prepare
5. Use `nbdev_build_lib` to build the library
6. Use `nbdev_build_docs` to generate documentation

## 3. Core Components

### 3.1 DHPCTIndividual Class

#### 3.1.1 Overview
`DHPCTIndividual` encapsulates:
- An environment (e.g., OpenAI Gym) that provides sensory observations and receives actions
- A Keras model representing a PCT hierarchy with multiple control levels

#### 3.1.2 Class Structure
```python
class DHPCTIndividual:
    def __init__(self, env_name, gym_name, env_props=None, levels=None, activation_funcs=None, weight_types=None):
        """
        Initialize a new individual with environment and hierarchy specifications.
        
        Parameters:
        - env_name: String identifier for the environment
        - gym_name: Name of the OpenAI Gym environment
        - env_props: Additional environment properties
        - levels: Array of column sizes for each level in the hierarchy
        - activation_funcs: Dict mapping levels to activation functions
        - weight_types: Dict specifying weight variable types
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
        """Return a dictionary of the individual's properties"""
        pass
        
    def run(self, steps, train=False, early_termination=False):
        """Run the individual in its environment"""
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
The PCT hierarchy consists of:
- **Perceptual Inputs**: Environment observations fed into the first level
- **Reference Inputs**: External inputs to higher levels
- **Multiple Levels**: Each with perception, reference, comparator, and output layers
- **Actions**: Outputs from the lowest level affecting the environment
- **Errors**: Collection of all comparator values across levels

#### 3.1.4 Model Creation Specification
The Keras model must follow this structure:
- Level 0 (lowest): Perception layer = weighted sum of inputs
- Higher levels: Perception layer = weighted sum of perception layer below
- Top level: Comparator = reference inputs - perception layer
- Other levels: Comparator = reference layer - perception layer
- Reference layers: Weighted sum of output layer from above
- Output layers: Element-wise multiplication of weights and comparator values
- Model outputs: Actions (from level 0) and errors (all comparators)

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
                 unchanged_generations=10,
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

## 4. Implementation Details

### 4.1 DHPCTIndividual Implementation Notes
- Use Keras Functional API for model creation
- Default activation function: linear
- Default weight type: float
- Mate operation: Use DEAP crossover for structure, blend weights
- Mutation operation: Structure mutation (10%) + Weight mutation (100%)
- Evaluation: Run multiple times and average fitness

### 4.2 Example Configuration Dictionary
```python
config = {
    "env": {
        "name": "CartPole-v1",
        "gym_name": "CartPole-v1",
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
2. Environment integration with OpenAI Gym
3. Keras model creation and structure
4. Evolution operators (mate, mutate)
5. DHPCTEvolver implementation
6. Documentation and examples
7. Performance optimization
