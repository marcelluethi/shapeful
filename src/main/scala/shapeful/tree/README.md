# TreeMap Implementation - JAX-like Tree Operations

This implementation provides a JAX-like `tree_map` system for Scala, allowing you to apply functions to all tensors in nested data structures while preserving the structure.

## Core Concepts

### TreeMap Typeclass
```scala
trait TreeMap[T]:
  def map(value: T, f: Tensor[?] => Tensor[?]): T
  def fold[A](value: T, init: A)(f: (A, Tensor[?]) => A): A
```

### Built-in Instances
- **Tensors**: Leaf nodes where functions are applied
- **Scalars**: Pass-through types (Int, Float, Double, String, Boolean)

## Usage Examples

### 1. Basic Parameter Structure
```scala
case class LinearParams(
  weight: Tensor2["input", "output"], 
  bias: Tensor1["output"]
)

// Manual TreeMap instance
given linearTreeMap: TreeMap[LinearParams] with
  def map(params: LinearParams, f: Tensor[?] => Tensor[?]): LinearParams =
    LinearParams(
      weight = f(params.weight).asInstanceOf[Tensor2["input", "output"]],
      bias = f(params.bias).asInstanceOf[Tensor1["output"]]
    )
  
  def fold[A](params: LinearParams, init: A)(f: (A, Tensor[?]) => A): A =
    val afterWeight = f(init, params.weight)
    f(afterWeight, params.bias)
```

### 2. Tree Operations
```scala
import shapeful.tree.{treeMap, treeFold, tensorCount, allTensors}

val params = LinearParams(
  weight = Tensor2["input", "output"](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f))),
  bias = Tensor1["output"](Seq(0.1f, 0.2f))
)

// Apply function to all tensors
val doubled = params.treeMap(tensor => tensor * Tensor0(2.0f))

// Count tensors in structure
val count = params.tensorCount  // Returns 2

// Extract all tensors
val tensors = params.allTensors  // Returns List[Tensor[?]]

// Fold over all tensors
val totalSum = params.treeFold(0.0f)((acc, tensor) => acc + tensor.sum.toFloat)
```

### 3. Nested Structures
```scala
case class MLPParams(
  layer1: LinearParams,
  layer2: LinearParams,
  learningRate: Float  // Non-tensor field
)

given mlpTreeMap: TreeMap[MLPParams] with
  def map(params: MLPParams, f: Tensor[?] => Tensor[?]): MLPParams =
    MLPParams(
      layer1 = summon[TreeMap[LinearParams]].map(params.layer1, f),
      layer2 = summon[TreeMap[LinearParams]].map(params.layer2, f),
      learningRate = params.learningRate  // Unchanged
    )
  
  def fold[A](params: MLPParams, init: A)(f: (A, Tensor[?]) => A): A =
    val afterLayer1 = summon[TreeMap[LinearParams]].fold(params.layer1, init)(f)
    summon[TreeMap[LinearParams]].fold(params.layer2, afterLayer1)(f)
```

### 4. Optimizer-Style Operations
```scala
// SGD step simulation
def updateParams(params: MLPParams, gradients: MLPParams, lr: Float): MLPParams =
  params.treeMap(param => {
    // In real usage, you'd lookup corresponding gradient
    param - (param * Tensor0(lr * 0.01f))  // Mock gradient step
  })

// Apply L2 regularization
val regularized = params.treeMap(tensor => tensor * Tensor0(0.99f))

// Zero gradients
val zeros = params.treeMap(_ => Tensor.zeros(tensor.shape))
```

## API Reference

### Extension Methods
```scala
extension [T: TreeMap](tree: T)
  def treeMap(f: Tensor[?] => Tensor[?]): T
  def treeFold[A](init: A)(f: (A, Tensor[?]) => A): A
  def tensorCount: Int
  def allTensors: List[Tensor[?]]
```

### Static Methods
```scala
object TreeOps:
  def treeMap[T: TreeMap](tree: T, f: Tensor[?] => Tensor[?]): T
  def treeFold[T: TreeMap, A](tree: T, init: A)(f: (A, Tensor[?]) => A): A
  def tensorCount[T: TreeMap](tree: T): Int
  def allTensors[T: TreeMap](tree: T): List[Tensor[?]]
```

## Benefits

### 1. **Type Safety**
- Preserves exact types and shapes
- Compile-time structure validation
- No runtime type errors

### 2. **JAX Compatibility**
- Similar API to JAX's `tree_map`
- Natural translation from Python JAX code
- Functional programming style

### 3. **Flexibility**
- Works with arbitrarily nested structures
- Handles mixed tensor/non-tensor fields
- Extensible to new parameter types

### 4. **Performance**
- No runtime reflection
- Inlined operations where possible
- Minimal allocation overhead

## Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Manual TreeMap** | Full type safety, explicit control | Boilerplate for each type |
| **Reflection-based** | No boilerplate | Runtime overhead, type erasure |
| **Macro derivation** | Automatic, type-safe | Complex implementation |
| **Generic programming** | Flexible | Less type information |

## Future Enhancements

1. **Automatic Derivation**: Macro-based generation of TreeMap instances
2. **Path Information**: Track field paths during traversal
3. **Conditional Mapping**: Apply different functions based on tensor properties
4. **Parallel Operations**: Concurrent tensor processing
5. **Integration**: Seamless integration with autodiff and optimizers

This TreeMap implementation provides a solid foundation for JAX-like tree operations in Scala while maintaining strong type safety and performance characteristics.
