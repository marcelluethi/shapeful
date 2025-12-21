package object shapeful:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  trait Prime[T]
  object Prime:
    given[L](using label: Label[L]): Label[Prime[L]] with
      val name: String = s"${label.name}'"

    type RemovePrimes[T <: Tuple] <: Tuple = T match
      case EmptyTuple => EmptyTuple
      case Prime[l] *: tail => l *: RemovePrimes[tail]
      case h *: tail => h *: RemovePrimes[tail]

    extension[T <: Tuple : Labels](tensor: Tensor[T])
      def dropPrimes: Tensor[RemovePrimes[T]] =
        given newLabels: Labels[RemovePrimes[T]] with
          val names: List[String] = 
            val oldLabels = summon[Labels[T]]
            oldLabels.names.toList.map(_.replace("'", ""))
        Tensor.fromPy(tensor.jaxValue)

  object LabelMath:
    case class Combined[A, B]()
    infix type *[A, B] = Combined[A, B]

  // Export tensor and related types
  export shapeful.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3}
  export shapeful.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export shapeful.tensor.{DType, Device}
  export shapeful.tensor.{Label, Labels, Axis, AxisIndex, AxisIndices, Dim}
  
  // Export type helpers
  export shapeful.tensor.Axis.UnwrapAxes
  export shapeful.tensor.TupleHelpers.*
  export shapeful.tensor.Broadcast
  export LabelMath.{Combined, `*`}
  export Prime.*
  
  // Export operations
  export shapeful.tensor.TensorOps.*
  
  // Export automatic differentiation
  export shapeful.autodiff.{Autodiff, TensorTree, ToPyTree}

  // Export Just-in-Time compilation
  export shapeful.jax.Jit.{jit, jit2}

  // Export implicit conversions
  object Conversions:
    export shapeful.tensor.Tensor0.{given_Conversion_Int_Tensor0, given_Conversion_Float_Tensor0}

