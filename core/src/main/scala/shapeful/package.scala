package object shapeful:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  // Export tensor and related types
  export shapeful.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3}
  export shapeful.tensor.{FloatTensor, FloatTensor0, FloatTensor1, FloatTensor2, FloatTensor3}
  export shapeful.tensor.{IntTensor, IntTensor0, IntTensor1, IntTensor2, IntTensor3}
  export shapeful.tensor.{BooleanTensor, BooleanTensor0, BooleanTensor1, BooleanTensor2, BooleanTensor3}
  export shapeful.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export shapeful.tensor.{DType, Device}
  export shapeful.tensor.{Label, Labels, Axis, AxisIndex, AxisIndices, Dim, TensorValue, ScalarValue}

  // Export type helpers
  export shapeful.tensor.Axis.UnwrapAxes
  export shapeful.tensor.TupleHelpers.*
  export shapeful.tensor.Broadcast
  export StringLabelMath.`*`

  // Export operations
  export shapeful.tensor.TensorOps.*

  // Export automatic differentiation
  export shapeful.autodiff.{Autodiff, TensorTree, ToPyTree}

  // Export Just-in-Time compilation
  export shapeful.jax.Jit.{jit, jit2}

  // Note: Implicit conversions were removed in the Value refactoring
