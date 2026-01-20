# API Contract: DHPCTIndividual

**Module**: `dpct.core`  
**Description**: Represents a hierarchical Perceptual Control Theory system

## Constructor

```python
DHPCTIndividual(
    env_name: str,
    levels: list[int],
    activation_funcs: str | list[str] = "linear",
    weight_types: str | list[str] = "float",
    fixed_weights: set[str] | None = None,
    fixed_levels: set[int] | None = None,
    obs_connection_level: int = 0,
    random_seed: int | None = None
)
```

**Parameters**:
- `env_name`: Gymnasium environment identifier (e.g., "CartPole-v1")
- `levels`: Number of units per level, bottom to top (e.g., [4, 3, 2])
- `activation_funcs`: Activation function(s) - single string or list per level
- `weight_types`: Weight type(s) - "float", "boolean", or "ternary"
- `fixed_weights`: Set of layer names that should not mutate
- `fixed_levels`: Set of level indices that should not mutate  
- `obs_connection_level`: Which level receives environment observations
- `random_seed`: Seed for reproducible weight initialization

**Returns**: New DHPCTIndividual instance (uncompiled)

**Raises**:
- `ValueError`: If levels is empty, activation_funcs/weight_types length mismatch
- `EnvironmentError`: If environment cannot be created

---

## Class Methods

### from_config

```python
@classmethod
def from_config(cls, config: dict) -> DHPCTIndividual
```

**Description**: Create individual from configuration dictionary

**Parameters**:
- `config`: HierarchyConfiguration dictionary (see data-model.md)

**Returns**: New compiled DHPCTIndividual with weights loaded

**Raises**:
- `ValueError`: If configuration is invalid
- `KeyError`: If required keys missing from config

---

### from_legacy_config

```python
@classmethod
def from_legacy_config(cls, legacy_config: dict) -> DHPCTIndividual
```

**Description**: Create individual from legacy configuration format

**Parameters**:
- `legacy_config`: Configuration in legacy format specification

**Returns**: New DHPCTIndividual instance

**Raises**:
- `ValueError`: If legacy config cannot be converted

---

## Instance Methods

### compile

```python
def compile(self) -> None
```

**Description**: Build Keras Functional API model from hierarchy specification

**Side Effects**:
- Creates `self.model` as Keras Model
- Initializes weights according to weight_types
- Sets up layer connections following PCT principles

**Raises**:
- `RuntimeError`: If already compiled
- `ValueError`: If hierarchy specification is invalid

**Post-conditions**:
- `self.model` is not None
- Model has inputs: "Observations", "ReferencesInput"
- Model has outputs: "Actions", "Errors"
- All layers follow naming convention (PL##, RL##, CL##, OL##)

---

### run

```python
def run(
    self,
    steps: int = 500,
    train: bool = False,
    early_termination: bool = True,
    record_history: bool = False,
    train_every_n_steps: int = 1,
    learning_rate: float = 0.01,
    optimizer: str = "adam",
    error_weight_coefficients: list[float] | None = None,
    render: bool = False
) -> float
```

**Description**: Execute individual in environment and return fitness

**Parameters**:
- `steps`: Maximum number of environment steps
- `train`: Enable online learning during execution
- `early_termination`: Stop when environment returns done=True
- `record_history`: Record ExecutionHistory
- `train_every_n_steps`: Frequency of weight updates during training
- `learning_rate`: Learning rate for online learning
- `optimizer`: Optimizer name for training ("adam", "sgd", etc.)
- `error_weight_coefficients`: Weights for different level errors in training
- `render`: Render environment during execution

**Returns**: Fitness score (typically cumulative reward)

**Side Effects**:
- Creates environment instance
- Updates `self.fitness`
- If `record_history=True`, stores ExecutionHistory in `self.history`
- If `train=True`, modifies `self.weights`

**Raises**:
- `RuntimeError`: If not compiled
- `EnvironmentError`: If environment interaction fails

---

### evaluate

```python
def evaluate(self, nevals: int = 1, aggregate: str = "mean") -> float
```

**Description**: Run multiple trials and aggregate fitness

**Parameters**:
- `nevals`: Number of evaluation runs
- `aggregate`: Aggregation method ("mean", "max", "min", "median")

**Returns**: Aggregated fitness across runs

**Raises**:
- `ValueError`: If aggregate method unknown

---

### mate

```python
def mate(self, other: DHPCTIndividual) -> tuple[DHPCTIndividual, DHPCTIndividual]
```

**Description**: Create two offspring via crossover with another individual

**Parameters**:
- `other`: Parent individual to mate with

**Returns**: Tuple of two new offspring individuals

**Raises**:
- `ValueError`: If individuals have incompatible structures
- `RuntimeError`: If either parent not compiled

**Crossover Strategy**:
- Uniform crossover for float weights: blend parent values
- Uniform crossover for boolean/ternary: randomly select parent value
- Offspring inherit structure from parents
- Fixed weights/levels preserved in offspring

---

### mutate

```python
def mutate(
    self,
    weight_prob: float = 0.1,
    struct_prob: float = 0.0,
    weight_mutation_strength: float = 1.0
) -> None
```

**Description**: Modify weights and/or structure

**Parameters**:
- `weight_prob`: Probability of mutating each weight
- `struct_prob`: Probability of structural mutation (reserved for future)
- `weight_mutation_strength`: Scale factor for weight mutations

**Side Effects**:
- Modifies `self.weights` for non-fixed layers/levels
- Float: Gaussian noise added
- Boolean: Bit flip
- Ternary: Random reassignment to {-1, 0, 1}

**Raises**:
- `RuntimeError`: If not compiled
- `ValueError`: If probabilities out of [0, 1] range

---

### config

```python
def config(self) -> dict
```

**Description**: Return complete configuration dictionary

**Returns**: HierarchyConfiguration dict (see data-model.md)

**Raises**:
- `RuntimeError`: If not compiled

---

### save_config

```python
def save_config(self, filepath: str, format: str = "json") -> None
```

**Description**: Save configuration to file

**Parameters**:
- `filepath`: Path to save configuration
- `format`: File format ("json" or "pickle")

**Side Effects**:
- Creates file at filepath
- Creates parent directories if needed

**Raises**:
- `IOError`: If file cannot be written
- `ValueError`: If format unknown

---

### to_legacy_config

```python
def to_legacy_config(self) -> dict
```

**Description**: Convert configuration to legacy format

**Returns**: Configuration in legacy format specification

**Raises**:
- `RuntimeError`: If not compiled

---

## Properties

### is_compiled

```python
@property
def is_compiled(self) -> bool
```

**Returns**: True if model has been compiled

---

### num_levels

```python
@property
def num_levels(self) -> int
```

**Returns**: Number of hierarchy levels

---

### total_parameters

```python
@property
def total_parameters(self) -> int
```

**Returns**: Total number of trainable parameters in model

**Raises**:
- `RuntimeError`: If not compiled

---

### layer_names

```python
@property
def layer_names(self) -> list[str]
```

**Returns**: List of all layer names in model

**Raises**:
- `RuntimeError`: If not compiled

---

## Usage Example

```python
# Create individual
individual = DHPCTIndividual(
    env_name="CartPole-v1",
    levels=[4, 3, 2],
    activation_funcs="linear",
    weight_types="float",
    random_seed=42
)

# Compile model
individual.compile()

# Run in environment
fitness = individual.run(steps=500, early_termination=True)
print(f"Fitness: {fitness}")

# Save configuration
individual.save_config("individual.json")

# Load configuration
loaded = DHPCTIndividual.from_config(individual.config())
```

## Related Requirements

- FR-001: Support creating with env, levels, activations, weight types
- FR-002: Support from_config() class method
- FR-003: compile() creates Keras model
- FR-004: run() executes in environment
- FR-005, FR-005a: config() returns pickleable dict
- FR-006: save_config() persists to JSON
- FR-007: mate() creates offspring
- FR-008: mutate() modifies structure/weights
- FR-009: evaluate() runs multiple trials
- FR-009a: to/from_legacy_config() methods
- FR-009b: run() supports history recording
- FR-042-046: Online learning features
