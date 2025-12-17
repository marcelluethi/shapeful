package shapeful.tensor

import shapeful.jax.{Jax, Einops}
import scala.annotation.targetName
import scala.annotation.implicitNotFound
import shapeful.tensor.TupleHelpers.{Subset, Remover, RemoverAll, Replacer}
import shapeful.tensor.{Label, Labels}
import shapeful.tensor.Axis.UnwrapAxes
import scala.util.NotGiven
import scala.collection.View.Empty
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

trait Broadcast[T1 <: Tuple, T2 <: Tuple, Out <: Tuple]:
  def broadcast(t1: Tensor[T1], t2: Tensor[T2]): (Tensor[Out], Tensor[Out])

object Broadcast:
  
  import shapeful.tensor.TensorOps.Structural.lift

  given identity[T <: Tuple]: Broadcast[T, T, T] with
    def broadcast(t1: Tensor[T], t2: Tensor[T]): (Tensor[T], Tensor[T]) = (t1, t2)
  
  given broadcastToLeft[T1 <: Tuple : Labels, T2 <: Tuple : Labels](using
    ev: Subset[T1, T2]
  ): Broadcast[T1, T2, T1] with
    def broadcast(t1: Tensor[T1], t2: Tensor[T2]): (Tensor[T1], Tensor[T1]) =
      val liftedT2 = t2.lift[T1](t1.shape)
      (t1, liftedT2)

  given broadcastToRight[T1 <: Tuple : Labels, T2 <: Tuple : Labels](using
    ev: Subset[T2, T1],
  ): Broadcast[T1, T2, T2] with
    def broadcast(t1: Tensor[T1], t2: Tensor[T2]): (Tensor[T2], Tensor[T2]) =
      val liftedT1 = t1.lift[T2](t2.shape)
      (liftedT1, t2)

object TensorOps:

  // -----------------------------------------------------------
  // 1. Elementwise Operations (The Field)
  // Preserves Shape: T -> T
  // -----------------------------------------------------------
  object Elementwise:

    def maximum[T <: Tuple : Labels](t1: Tensor[T], t2: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))
    def minimum[T <: Tuple : Labels](t1: Tensor[T], t2: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple : Labels](t: Tensor[T])
      
      def +(other: Tensor[T]): Tensor[T] = t.add(other)
      def :+[O <: Tuple](other: Tensor[O])(using broadcaster: Broadcast[T, O, T]): Tensor[T] = broadcaster.broadcast(t, other) match { case (l, r) => l.add(r) }
      def +:[O <: Tuple : Labels](other: Tensor[O])(using broadcaster: Broadcast[O, T, O]): Tensor[O] = broadcaster.broadcast(other, t) match { case (r, l) => l.add(r) }
      private def add(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))
      
      def unary_- : Tensor[T] = Tensor(Jax.jnp.negative(t.jaxValue))
      def -(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))
      def :-[O <: Tuple](other: Tensor[O])(using broadcaster: Broadcast[T, O, T]): Tensor[T] = broadcaster.broadcast(t, other) match { case (l, r) => l.subtract(r) }
      def -:[O <: Tuple : Labels](other: Tensor[O])(using broadcaster: Broadcast[O, T, O]): Tensor[O] = broadcaster.broadcast(other, t) match { case (r, l) => l.subtract(r) }
      private def subtract(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))

      def *(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      def scale(other: Tensor0) = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      def :*[O <: Tuple](other: Tensor[O])(using broadcaster: Broadcast[T, O, T]): Tensor[T] = broadcaster.broadcast(t, other) match { case (l, r) => l.multiply(r) }
      def *:[O <: Tuple : Labels](other: Tensor[O])(using broadcaster: Broadcast[O, T, O]): Tensor[O] = broadcaster.broadcast(other, t) match { case (r, l) => l.multiply(r) }
      private def multiply(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      
      def /(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))
      def :/[O <: Tuple](other: Tensor[O])(using broadcaster: Broadcast[T, O, T]): Tensor[T] = broadcaster.broadcast(t, other) match { case (l, r) => l.divide(r) }
      def /:[O <: Tuple : Labels](other: Tensor[O])(using broadcaster: Broadcast[O, T, O]): Tensor[O] = broadcaster.broadcast(other, t) match { case (r, l) => l.divide(r) }
      private def divide(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))

      // --- Unary Math ---
      def abs: Tensor[T] = Tensor(Jax.jnp.abs(t.jaxValue))
      def sign: Tensor[T] = Tensor(Jax.jnp.sign(t.jaxValue))
      def pow(n: Tensor0): Tensor[T] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))
      def sqrt: Tensor[T] = Tensor(Jax.jnp.sqrt(t.jaxValue))
      def exp: Tensor[T] = Tensor(Jax.jnp.exp(t.jaxValue))
      def log: Tensor[T] = Tensor(Jax.jnp.log(t.jaxValue))
      def sin: Tensor[T] = Tensor(Jax.jnp.sin(t.jaxValue))
      def cos: Tensor[T] = Tensor(Jax.jnp.cos(t.jaxValue))
      def tanh: Tensor[T] = Tensor(Jax.jnp.tanh(t.jaxValue))

      // --- Clipping ---
      def clip(min: Float, max: Float): Tensor[T] = Tensor(Jax.jnp.clip(t.jaxValue, min, max))
      def clip(min: Tensor0, max: Tensor0): Tensor[T] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))

      // --- Comparison ---
      def <(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.less(t.jaxValue, other.jaxValue))
      def <=(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.less_equal(t.jaxValue, other.jaxValue))
      def >(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.greater(t.jaxValue, other.jaxValue))
      def >=(other: Tensor[T]): Tensor[T] = Tensor(Jax.jnp.greater_equal(t.jaxValue, other.jaxValue))
      
      def elementEquals(other: Tensor[T]): Tensor[T] =
        require(t.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${t.shape.dimensions} vs ${other.shape.dimensions}")
        Tensor(jaxValue = Jax.jnp.equal(t.jaxValue, other.jaxValue))

      def all: Boolean = Tensor0(Jax.jnp.all(t.jaxValue)).toBool
      def any: Boolean = Tensor0(Jax.jnp.any(t.jaxValue)).toBool

      def approxEquals(other: Tensor[T], tolerance: Float = 1e-6f): Boolean = approxElementEquals(other, tolerance).all
      def approxElementEquals(other: Tensor[T], tolerance: Float = 1e-6f): Tensor[T] =
        Tensor(Jax.jnp.allclose(
          t.jaxValue,
          other.jaxValue,
          atol = tolerance,
          rtol = tolerance
        ))
  
  end Elementwise

  // -----------------------------------------------------------
  // 2. Reduction Operations (The Monoid)
  // Reduces Rank: T -> T - {Axis}
  // -----------------------------------------------------------
  object Reduction:

    extension [T <: Tuple : Labels](t: Tensor[T])

      def sum: Tensor0 = Tensor0(Jax.jnp.sum(t.jaxValue))
      def sum[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = t.sum(axes = Tuple1(axis))
      def sum[Inputs <: Tuple](axes: Inputs)(using remover: RemoverAll[T, UnwrapAxes[Inputs]], namesOf: Labels[UnwrapAxes[Inputs]], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[remover.Out] = Tensor(Jax.jnp.sum(t.jaxValue, axesIndices.values.toPythonProxy))
      
      def mean: Tensor0 = Tensor0(Jax.jnp.mean(t.jaxValue))
      def mean[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = t.mean(axes = Tuple1(axis))
      def mean[Inputs <: Tuple](axes: Inputs)(using remover: RemoverAll[T, UnwrapAxes[Inputs]], namesOf: Labels[UnwrapAxes[Inputs]], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[remover.Out] = Tensor(Jax.jnp.mean(t.jaxValue, axesIndices.values.toPythonProxy))
      
      def std: Tensor0 = Tensor0(Jax.jnp.std(t.jaxValue))
      def std[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = t.std(axes = Tuple1(axis))
      def std[Inputs <: Tuple](axes: Inputs)(using remover: RemoverAll[T, UnwrapAxes[Inputs]], namesOf: Labels[UnwrapAxes[Inputs]], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[remover.Out] = Tensor(Jax.jnp.std(t.jaxValue, axesIndices.values.toPythonProxy))
      
      def max: Tensor0 = Tensor0(Jax.jnp.max(t.jaxValue))
      def max[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = t.max(axes = Tuple1(axis))
      def max[Inputs <: Tuple](axes: Inputs)(using remover: RemoverAll[T, UnwrapAxes[Inputs]], namesOf: Labels[UnwrapAxes[Inputs]], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[remover.Out] = Tensor(Jax.jnp.max(t.jaxValue, axesIndices.values.toPythonProxy))
      
      def min: Tensor0 = Tensor0(Jax.jnp.min(t.jaxValue))
      def min[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = t.min(axes = Tuple1(axis))
      def min[Inputs <: Tuple](axes: Inputs)(using remover: RemoverAll[T, UnwrapAxes[Inputs]], namesOf: Labels[UnwrapAxes[Inputs]], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[remover.Out] = Tensor(Jax.jnp.min(t.jaxValue, axesIndices.values.toPythonProxy))
      
      def argmax: Tensor0 = Tensor0(Jax.jnp.argmax(t.jaxValue))
      def argmax[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = axisIndex.value))
      
      def argmin: Tensor0 = Tensor0(Jax.jnp.argmin(t.jaxValue))
      def argmin[L : Label](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover[T, L]): Tensor[remover.Out] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = axisIndex.value))
      
  end Reduction

  object Contraction:

    extension [T <: Tuple : Labels](tensor: Tensor[T])
      
      def outerProduct[OtherShape <: Tuple : Labels](other: Tensor[OtherShape]): Tensor[Tuple.Concat[T, OtherShape]] =
        import Labels.ForConcat.given
        Tensor(
          // Jax outer product flattens, reshape required
          Jax.jnp.reshape(
            Jax.jnp.outer(tensor.jaxValue, other.jaxValue), 
            (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
          )
        )

      def contract[
          ContractAxis : Label,
          OtherContractAxis : Label,
          OtherShape <: Tuple : Labels,
      ]
      (axes: (Axis[ContractAxis], Axis[OtherContractAxis]))
      (other: Tensor[OtherShape])(using
        replacer: Replacer[T, ContractAxis, OtherContractAxis],
        remover: Remover[replacer.Out, OtherContractAxis],
        otherRemover: Remover[OtherShape, OtherContractAxis],
        axisIndex: AxisIndex[replacer.Out, OtherContractAxis],
        otherAxisIndex: AxisIndex[OtherShape, OtherContractAxis],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] = 
        val (thisAxes, otherAxis) = axes
        val tensorRenamed = tensor.relabel(thisAxes -> otherAxis)
        tensorRenamed.contract[OtherContractAxis, OtherShape](otherAxis)(other)

      def contract[
        ContractAxis : Label,
        OtherShape <: Tuple : Labels,
      ]
      (axis: Axis[ContractAxis])
      (other: Tensor[OtherShape])(using
        remover: Remover[T, ContractAxis],
        otherRemover: Remover[OtherShape, ContractAxis],
        axisIndex: AxisIndex[T, ContractAxis],
        otherAxisIndex: AxisIndex[OtherShape, ContractAxis],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import Labels.ForConcat.given

        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(axisIndex.value).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

        Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))
  
  end Contraction

  object LinearAlgebra:
    
    extension [T <: Tuple : Labels](t: Tensor[T])
      def norm: Tensor0 = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))
      def inv: Tensor[T] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))
      
      def det[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2])(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out] = 
        // JAX det only works on the last two axes (-2, -1). We must move the user's selected axes to the end.
        val moved = Jax.jnp.moveaxis(
          t.jaxValue, 
          source = Seq(idx1.value, idx2.value).toPythonProxy, 
          destination = Seq(-2, -1).toPythonProxy
        )
        Tensor(Jax.jnp.linalg.det(moved))
      
      def trace[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int=0)(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = idx1.value, axis2 = idx2.value))

      def diagonal[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int=0)(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out *: L1 *: EmptyTuple] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = idx1.value, axis2 = idx2.value))

    extension [L1 : Label, L2 : Label](t: Tensor2[L1, L2])
      def det: Tensor0 = Tensor0(Jax.jnp.linalg.det(t.jaxValue))
      def trace: Tensor0 = t.trace(0)
      def trace(offset: Int): Tensor0 = Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))
      def diagonal: Tensor1[L1] = t.diagonal(0)
      def diagonal(offset: Int): Tensor1[L1] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

  end LinearAlgebra

  // -----------------------------------------------------------
  // 4. Structural Operations (Isomorphisms)
  // Permutations and Views: T1 -> T2 (Size(T1) == Size(T2))
  // -----------------------------------------------------------
  object Structural:
    
    private object Util:
      
      type InsertBefore[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail => B *: A *: tail
        case h *: tail => h *: InsertBefore[tail, A, B]

      type InsertAfter[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail => A *: B *: tail
        case h *: tail => h *: InsertAfter[tail, A, B]

      type SliceIndex = Int | List[Int] | Range
      type ExtractLabel[X] = X match
          case (Axis[l], SliceIndex) => l
      type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

      trait SliceLabelExtractor[Inputs <: Tuple, Out <: Tuple]

      object SliceLabelExtractor:

        given empty: SliceLabelExtractor[EmptyTuple, EmptyTuple] = 
          new SliceLabelExtractor[EmptyTuple, EmptyTuple] {}

        given consInt[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] = 
          new SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] {}

        given consSeq[L, SeqT <: Seq[Int], Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] = 
          new SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] {}

      type Swap[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => EmptyTuple
        case A *: tail  => B *: Swap[tail, A, B]
        case B *: tail  => A *: Swap[tail, A, B]
        case h *: tail  => h *: Swap[tail, A, B]

      type TupleReduce[T <: Tuple, Op[_ <: String, _ <: String]] = T match
        case EmptyTuple => ""
        case h *: EmptyTuple => h
        case h *: t => Op[h, TupleReduce[t, Op]]

      type JoinNames[T <: Tuple] = TupleReduce[T, shapeful.StringLabelMath.*]

      trait DimExtractor[T]:
        def extract(t: T): Map[String, Int]

      object DimExtractor:
        given DimExtractor[EmptyTuple] with
          def extract(t: EmptyTuple) = Map.empty

        given [L, Tail <: Tuple](using
          labelValue: ValueOf[L],
          tailExtractor: DimExtractor[Tail]
        ): DimExtractor[(Axis[L], Int) *: Tail] with
          def extract(t: (Axis[L], Int) *: Tail) =
            val (_, size) = t.head
            Map(labelValue.value.toString -> size) ++ tailExtractor.extract(t.tail)
        
    import Util.*

    object TensorWhere:
      def where[T <: Tuple : Labels](
        condition: Tensor[T],
        x: Tensor[T],
        y: Tensor[T]
      ): Tensor[T] =
        Tensor(Jax.jnp.where(condition.jaxValue, x.jaxValue, y.jaxValue))
    
    export TensorWhere.where

    def stack[L : Label, T <: Tuple : Labels](
      tensors: Seq[Tensor[T]], 
      newAxis: Axis[L],
    ): Tensor[L *: T] = 
      require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = 0)
      Tensor(stackedJaxValue)

    def stack[NewL, L, T <: Tuple : Labels](
      tensors: Seq[Tensor[T]], 
      newAxis: Axis[NewL],
      afterAxis: Axis[L],
    )(using 
      newLabel: Label[NewL],
      axisIndex: AxisIndex[T, L],
    ): Tensor[InsertAfter[T, L, NewL]] =
      require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
      val axisIdx = axisIndex.value + 1 // we are inserting after the given axis, so shift by 1
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = axisIdx)
      val names = summon[Labels[T]].names
      val newNames = names.take(axisIdx) ++ Seq(newLabel.name) ++ names.drop(axisIdx)
      given Labels[InsertAfter[T, L, NewL]] with
        val names = newNames.toSeq
      Tensor(stackedJaxValue)

    def concatenate[L : Label, T <: Tuple : Labels](
      tensors: Seq[Tensor[T]], 
      concatAxis: Axis[L],
    )(
      using axisIndex: AxisIndex[T, L],
    ): Tensor[T] =
      require(tensors.nonEmpty, "Cannot concatenate an empty sequence of tensors")
      val axisIdx = axisIndex.value
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
      Tensor(concatenatedJaxValue)

    extension [T <: Tuple : Labels](tensor: Tensor[T])

      private def calcPyIndices[Inputs <: Tuple](
          inputs: Inputs,
          axesIndices: AxisIndices[T, ExtractLabels[Inputs]]
      ) = 

        val PySlice = py.Dynamic.global.slice
        val Colon = PySlice(py.None)
        val rank = tensor.shape.rank
        val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

        val inputList = inputs.toList.asInstanceOf[List[(Any, Any)]]
        val targetDims: List[Int] = axesIndices.values

        targetDims.zip(inputList).foreach { 
          case (dimIndex, (_, sliceIndex)) =>
            val dimSize = tensor.shape.dimensions(dimIndex)
            sliceIndex match {
              case sliceSeq: List[Int] @unchecked => 
                indicesBuffer(dimIndex) = sliceSeq.map(py.Any.from).toPythonProxy
              case range: Range @unchecked => 
                indicesBuffer(dimIndex) = PySlice(range.head, range.last+1, range.step)
              case idx: Int =>
                indicesBuffer(dimIndex) = py.Any.from(idx)
            }
        }
        
        Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)

      def split[newL, splitL](newAxis: Axis[newL], splitAxis: Axis[splitL], interval: Int)(using 
        newLabel: Label[newL],
        axisIndex: AxisIndex[T, splitL],
      ): Tensor[InsertBefore[T, splitL, newL]] = 
        val splitIdx = axisIndex.value
        val names = summon[Labels[T]].names
        val newNames = names.take(splitIdx) ++ Seq(newLabel.name) ++ names.drop(splitIdx)
        given Labels[InsertBefore[T, splitL, newL]] with
          val names = newNames.toSeq
        val (before, after) = tensor.shape.dimensions.splitAt(splitIdx)
        val newShape = before ++ Seq(interval, after.head / interval) ++ after.drop(1)
        println(newShape)
        Tensor(
          Jax.jnp.reshape(
            tensor.jaxValue,
            Jax.Dynamic.global.tuple(
              newShape.map(py.Any.from).toPythonProxy
            )
          )
        )

      def chunk[splitL : Label](splitAxis: Axis[splitL], interval: Int)(using 
        axisIndex: AxisIndex[T, splitL],
      ): Seq[Tensor[T]] = 
        val res = Jax.jnp.split(tensor.jaxValue, interval, axis = axisIndex.value).as[Seq[Jax.PyDynamic]]
        res.map(x => Tensor[T](x))

      def tile = ???
      def repeat = ???

      def slice[Inputs <: Tuple, LabelsToRemove <: Tuple](
        inputs: Inputs,
      )(using 
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        remover: RemoverAll[T, LabelsToRemove],
        axesIndices: AxisIndices[T, ExtractLabels[Inputs]],
        namesOf: Labels[LabelsToRemove],
      ): Tensor[remover.Out] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices)
        Tensor(tensor.jaxValue.bracketAccess(pyIndices))

      def slice[L, I, LabelsToRemove <: Tuple](
        axisWithSliceIndex: (Axis[L], I)
      )(using 
        sliceExtractor: SliceLabelExtractor[Tuple1[(Axis[L], I)], LabelsToRemove],
        remover: RemoverAll[T, LabelsToRemove],
        axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]],
        namesOf: Labels[LabelsToRemove],
      ): Tensor[remover.Out] = slice(Tuple1(axisWithSliceIndex))

      def set[Inputs <: Tuple, LabelsToRemove <: Tuple, O <: Tuple](
        inputs: Inputs
      )(using 
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        remover: RemoverAll[T, LabelsToRemove]{ type Out = O },
        axesIndices: AxisIndices[T, ExtractLabels[Inputs]],
        namesOf: Labels[LabelsToRemove]
      )(value: Tensor[remover.Out]): Tensor[T] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices)
        val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value.jaxValue)
        Tensor(result)

      def set[L, I, LabelsToRemove <: Tuple](
        axisWithSliceIndex: (Axis[L], I)
      )(using 
        sliceExtractor: SliceLabelExtractor[Tuple1[(Axis[L], I)], LabelsToRemove],
        remover: RemoverAll[T, LabelsToRemove],
        axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]],
        namesOf: Labels[LabelsToRemove]
      )(value: Tensor[remover.Out]): Tensor[T] = set(Tuple1(axisWithSliceIndex))(value)

      def rearrange[Axes <: Tuple](newOrder: Axes)(using Labels[UnwrapAxes[Axes]]): Tensor[UnwrapAxes[Axes]] = 
        rearrange[Axes, EmptyTuple](newOrder, EmptyTuple)

      def rearrange[Axes <: Tuple, Dims <: Tuple](
          newOrder: Axes,
          dims: Dims,
      )(using 
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[Dims],
      ): Tensor[UnwrapAxes[Axes]] =
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          def cleanPattern(pattern: String): String =
            // to replace all a*b*c in pattern with (a b c), example:
            // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
            val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(pattern, m => {
              val group = m.group(1)
              val replaced = group.split("\\*").mkString("(", " ", ")")
              replaced
            })
          s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
        val fromPattern = tensor.shape.labels.mkString(" ")
        val toPattern = newLabels.names.mkString(" ")
        val pattern = createEinopsPattern(fromPattern, toPattern)
        val dimSizesMap = extractor.extract(dims)
        Tensor(
          Einops.rearrange(
            tensor.jaxValue,
            pattern,
            kwargsMap = dimSizesMap
          )
        )

      def lift[O <: Tuple : Labels](newShape: Shape[O])(
        using ev: Subset[O, T] // Ensures T's axes are all present in O
      ): Tensor[O] =
        val t = tensor
        
        val currentNames = summon[Labels[T]].names
        val targetNames = summon[Labels[O]].names
        
        val targetOrder = targetNames.filter(currentNames.contains)
        val permutation = targetOrder.map(n => currentNames.indexOf(n))
        
        val alignedJax = if (permutation != currentNames.indices.toList) {
          Jax.jnp.transpose(t.jaxValue, permutation.toPythonProxy)
        } else {
          t.jaxValue
        }

        val currentShapeMap = currentNames.zip(t.shape.dimensions).toMap
        
        val intermediateShape = targetNames.map { name =>
          currentShapeMap.getOrElse(name, 1)
        }
        
        val reshapedJax = Jax.jnp.reshape(alignedJax, intermediateShape.toPythonProxy)
        Tensor(Jax.jnp.broadcast_to(reshapedJax, newShape.dimensions.toPythonProxy))

      def relabel[OldLabel : Label, NewLabel : Label](
        axis: (Axis[OldLabel], Axis[NewLabel]),
      )(
        using replacer: Replacer[T, OldLabel, NewLabel],
      ): Tensor[replacer.Out] = Tensor(tensor.jaxValue)

      def retag[newT <: Tuple](using newLabels: Labels[newT]): Tensor[newT] = 
        Tensor(tensor.jaxValue)(using newLabels)

      // TODO rename this as as[] is defined in Any and is taken in Scala
      def as[newT <: Tuple](using 
        newLabels: Labels[UnwrapAxes[newT]],
        @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
        evSameSize: Tuple.Size[newT] =:= Tuple.Size[T],
      ): Tensor[UnwrapAxes[newT]] = Tensor[UnwrapAxes[newT]](tensor.jaxValue)
  
      def swap[L1 : Label, L2 : Label](
        axis1: Axis[L1],
        axis2: Axis[L2],
      )(using
        axisIndex1: AxisIndex[T, L1],
        axisIndex2: AxisIndex[T, L2],
      ): Tensor[Swap[T, L1, L2]] =
        given Labels[Swap[T, L1, L2]] with
          def names = 
            val originalNames = summon[Labels[T]].names
            val ax1Name = summon[Label[L1]].name
            val ax2Name = summon[Label[L2]].name
            originalNames.map {
              case n if n == ax1Name => ax2Name
              case n if n == ax2Name => ax1Name
              case n => n
            }
        Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.value, axisIndex2.value))

      def ravel: Tensor1[JoinNames[T]] = 
        given Labels[Tuple1[JoinNames[T]]] with
          def names = List(summon[Labels[T]].names.mkString("*"))
        Tensor(Jax.jnp.ravel(tensor.jaxValue))

      def appendAxis[L : Label](axis: Axis[L]): Tensor[Tuple.Concat[T, Tuple1[L]]] =
        import Labels.ForConcat.given
        val newShape = tensor.shape.dimensions :+ 1
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def prependAxis[L : Label](axis: Axis[L]): Tensor[Tuple.Concat[Tuple1[L], T]] =
        import Labels.ForConcat.given
        val newShape = 1 +: tensor.shape.dimensions
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def squeeze[L : Label](axis: Axis[L])(using 
        remover: Remover[T, L],
        axisIndex: AxisIndex[T, L],
      ): Tensor[remover.Out] =
        require(
          tensor.shape.dimensions(axisIndex.value) == 1, 
          s"Cannot squeeze axis ${axis} of size ${tensor.shape.dimensions(axisIndex.value)}"
        )
        Tensor(Jax.jnp.squeeze(tensor.jaxValue, axis = axisIndex.value))

  end Structural

  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------
  object Functional:

    object ZipVmap:

      type TensorsOf[Shapes <: Tuple] <: Tuple = Shapes match
        case EmptyTuple => EmptyTuple
        case head *: tail => head match
          case Tuple => Tensor[head] *: TensorsOf[tail]

      type ExtractShape[T] = T match
        case Tensor[s] => s

      type ShapesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractShape]

      trait Zipper[Shapes <: Tuple, L]:
        type SlicedShapes <: Tuple
        def dimSize(tensors: TensorsOf[Shapes], axis: Axis[L]): Int
        def sliceAll(tensors: TensorsOf[Shapes], axis: Axis[L], idx: Int): TensorsOf[SlicedShapes]

      object Zipper:
        type Aux[Shapes <: Tuple, L, O <: Tuple] = Zipper[Shapes, L] { type SlicedShapes = O }

        given empty[L]: Zipper.Aux[EmptyTuple, L, EmptyTuple] = new Zipper[EmptyTuple, L]:
          type SlicedShapes = EmptyTuple
          def dimSize(t: EmptyTuple, axis: Axis[L]) = 0
          def sliceAll(t: EmptyTuple, axis: Axis[L], idx: Int) = EmptyTuple

        given cons[HeadShape <: Tuple : Labels, TailShapes <: Tuple, L : Label, TailSliced <: Tuple](
          using
          remover: Remover[HeadShape, L],
          axisIndex: AxisIndex[HeadShape, L],
          tailZipper: Zipper.Aux[TailShapes, L, TailSliced],
        ): Zipper.Aux[HeadShape *: TailShapes, L, remover.Out *: TailSliced] = 
          new Zipper[HeadShape *: TailShapes, L]:
            type SlicedShapes = remover.Out *: TailSliced

            def dimSize(tensors: TensorsOf[HeadShape *: TailShapes], axis: Axis[L]): Int =
              val head = tensors.asInstanceOf[Tensor[HeadShape] *: Tuple].head
              head.shape.dimensions(axisIndex.value)

            def sliceAll(tensors: TensorsOf[HeadShape *: TailShapes], axis: Axis[L], idx: Int): TensorsOf[SlicedShapes] =
              val tuple = tensors.asInstanceOf[Tensor[HeadShape] *: TensorsOf[TailShapes]]
              val slicedHead = tuple.head.slice(axis -> idx)
              val slicedTail = tailZipper.sliceAll(tuple.tail, axis, idx)
              (slicedHead *: slicedTail).asInstanceOf[TensorsOf[SlicedShapes]]

      case class ZipResult[L : Label, Shapes <: Tuple](
        axis: Axis[L],
        tensors: TensorsOf[Shapes]
      ):
        def vmap[OutShape <: Tuple : Labels](using
          zipper: Zipper[Shapes, L]
        )(
          f: TensorsOf[zipper.SlicedShapes] => Tensor[OutShape]
        ): Tensor[L *: OutShape] =

          val size = zipper.dimSize(tensors, axis)

          val results = (0 until size).map { i =>
            val slicedTuple = zipper.sliceAll(tensors, axis, i)
            f(slicedTuple)
          }

          Structural.stack(results, axis)

      def zip[L : Label, Inputs <: Tuple](
        axis: Axis[L]
      )(
        tensors: Inputs
      ): ZipResult[L, ShapesOf[Inputs]] = 
        ZipResult(axis, tensors.asInstanceOf[TensorsOf[ShapesOf[Inputs]]])

      def zipvmap[
          L : Label, 
          Inputs <: Tuple, 
          OutShape <: Tuple : Labels, 
      ](
          axis: Axis[L]
      )(
          tensors: Inputs
      )(using 
          zipper: Zipper[ShapesOf[Inputs], L]
      )(
          f: TensorsOf[zipper.SlicedShapes] => Tensor[OutShape]
      ): Tensor[L *: OutShape] = 
          zip(axis)(tensors).vmap(f)
    
    export ZipVmap.zipvmap

    extension [T <: Tuple : Labels](t: Tensor[T])

      def vmap[VmapAxis : Label, OuterShape <: Tuple : Labels](
        axis: Axis[VmapAxis]
      )(using
        remover: Remover[T, VmapAxis],
        vmapAxisIndex: AxisIndex[T, VmapAxis],
      )(
          f: Tensor[remover.Out] => Tensor[OuterShape]
      ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =
        val fpy = (jxpr: Jax.PyDynamic) =>
            val innerTensor = Tensor[remover.Out](jxpr)
            val result = f(innerTensor)
            result.jaxValue

        import Labels.ForConcat.given
        Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))

      def vapply[L : Label, OutAxis : Label](
        axis: Axis[L]
      )(using
        axisIndex: AxisIndex[T, L],
        replacer: Replacer[T, L, OutAxis],
      )(
        f: Tensor[Tuple1[L]] => Tensor[Tuple1[OutAxis]]
      ): Tensor[replacer.Out] = 
        val fpy = (jxpr: Jax.PyDynamic) =>
          val inputTensor = Tensor[Tuple1[L]](jxpr)
          val result = f(inputTensor)
          result.jaxValue

        Tensor(Jax.jnp.apply_along_axis(
          fpy, 
          axisIndex.value, 
          t.jaxValue
        ))

      def vreduce[L : Label](
        axis: Axis[L]
      )(
        f: Tensor[Tuple1[L]] => Tensor0
      )(using
        axisIndex: AxisIndex[T, L],
        remover: Remover[T, L],
      ): Tensor[remover.Out] = 
        val fpy = (jxpr: Jax.PyDynamic) =>
          val inputTensor = Tensor[Tuple1[L]](jxpr)
          val result = f(inputTensor)
          result.jaxValue

        Tensor(Jax.jnp.apply_along_axis(
          fpy, 
          axisIndex.value, 
          t.jaxValue
        ))

  end Functional

  export Elementwise.*
  export Reduction.*
  export Contraction.*
  export LinearAlgebra.*
  export Structural.*
  export Functional.*

  // -----------------------------------------------------------
  // Common specialized operation names
  // -----------------------------------------------------------
  object ScalarOps:
    extension (t: Tensor0)
      def toInt: Int = t.jaxValue.item().as[Int]
      def toFloat: Float = t.jaxValue.item().as[Float]
      def toBool: Boolean = t.jaxValue.item().as[Boolean]

      @targetName("tensor0Pow")
      def pow(exponent: Tensor0): Tensor0 = Tensor0(Jax.jnp.pow(t.jaxValue, exponent.jaxValue))

  object VectorOps:
    extension [L : Label](t: Tensor1[L])
      def dot(other: Tensor1[L]): Tensor0 = t.innerDot(other)
      def innerDot(other: Tensor1[L]): Tensor0 = t.contract(Axis[L])(other)
      def outerDot[OtherLabel : Label](other: Tensor1[OtherLabel]): Tensor2[L, OtherLabel] = 
        t.outerProduct(other)

  object MatrixOps:
    extension [L1 : Label, L2 : Label](t: Tensor2[L1, L2])
      def transpose: Tensor2[L2, L1] = t.rearrange((Axis[L2], Axis[L1]))

      @targetName("tensor2MatmulTensor2")
      def matmul[L3 : Label](other: Tensor2[L2, L3])(
        using 
        remover: Remover[(L1, L2), L2],
        otherRemover: Remover[(L2, L3), L2],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import Labels.ForConcat.given
        t.contract(Axis[L2])(other)

      @targetName("tensor2MatmulTensor1")
      def matmul(other: Tensor1[L2])(
        using 
        remover: Remover[(L1, L2), L2],
        otherRemover: Remover[Tuple1[L2], L2],
      ): Tensor[Tuple.Concat[remover.Out, otherRemover.Out]] =
        import Labels.ForConcat.given
        t.contract(Axis[L2])(other)

  export ScalarOps.*
  export VectorOps.*
  export MatrixOps.*

end TensorOps
