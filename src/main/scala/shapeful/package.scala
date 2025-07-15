import scala.language.experimental.namedTypeArguments

package object shapeful:

  import tensor.*

  type Label = Singleton

  export tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export tensor.Shape.{Shape0, Shape1, Shape2, Shape3}
  export tensor.Tensor
  export tensor.{Tensor0, Tensor1, Tensor2, Tensor3}
  export tensor.Tensor.{Tensor0, Tensor1, Tensor2, Tensor3}
  export tensor.DType
  export linalg.Linalg

  // Export all TensorOps extensions
  export tensor.TensorOps.*
