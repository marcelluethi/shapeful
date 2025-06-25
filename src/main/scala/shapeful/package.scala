import scala.language.experimental.namedTypeArguments

package object shapeful:

  import jax.Jax
  import tensor.*

  type Label = Singleton

  export jax.Jax.PyAny
  export jax.Jax

  export tensor.Tensor
  export tensor.{Tensor0, Tensor1, Tensor2, Tensor3}
  export tensor.Tensor.{Tensor0, Tensor1, Tensor2, Tensor3}

  // Export all TensorOps extensions
  export tensor.TensorOps.*
