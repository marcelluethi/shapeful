package shapeful.jax

import shapeful.tensor.{Tensor, Shape, TupleHelpers, DType}
import shapeful.jax.{Jax, JaxDType}
import shapeful.autodiff.ToPyTree

/** JIT compilation utilities for high-performance tensor operations.
  *
  * This object provides methods to JIT (Just-In-Time) compile functions that work with tensors and PyTrees, enabling
  * significant performance improvements (typically 10-50x faster) for repeated operations.
  *
  * @example
  *   {{{
  *   // Basic usage - compile any PyTree function
  *   val jitted = Jit[Tensor[Shape], OutShape](t => t.relu.softmax)
  *   val result = jitted(tensor)
  *
  *   // Convenience method for single tensor
  *   val jitted = Jit.function[Shape, OutShape](t => t.relu.sum)
  *
  *   // Compile functions with model parameters
  *   val jitted = Jit.withParams[Params, InShape, OutShape]((params, input) => model(params, input))
  *   }}}
  */
object Jit:

  /** JIT compile a function that takes any PyTree input and returns a Tensor. This is the most general form - all other
    * methods are convenience wrappers around this.
    *
    * @tparam PyTree
    *   The input type, must have a ToPyTree instance (Tensor, tuples, case classes, etc.)
    * @tparam OutT
    *   The output tensor shape as a tuple type
    * @param f
    *   The function to compile
    * @return
    *   A compiled version of the function
    *
    * @example
    *   {{{
    *   // Single tensor
    *   val jitted = Jit[Tensor[("Batch", "Feature")], ("Batch",)](t => t.sum(axis = 1))
    *
    *   // Tuple of tensors
    *   val jitted = Jit[(Tensor[S1], Tensor[S2]), OutShape] { case (t1, t2) => t1.add(t2) }
    *
    *   // Parameters and tensors
    *   val jitted = Jit[(ModelParams, Tensor[InShape], Tensor[LabelShape]), EmptyTuple] {
    *     case (params, inputs, labels) => loss(params, inputs, labels)
    *   }
    *   }}}
    */
  def apply[PyTree: ToPyTree, OutT <: Tuple](
      f: PyTree => Tensor[OutT]
  ): PyTree => Tensor[OutT] =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      // Reconstruct Scala object from pytree
      val pyTree = ToPyTree[PyTree].fromPyTree(pyTreePy)

      // Apply the function
      val result = f(pyTree)
      result.jaxValue

    val jitted = Jax.jax_helper.jit_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: PyTree) =>
      val pyTreePy = ToPyTree[PyTree].toPyTree(pyTree)
      val resultJax = jitted(pyTreePy)

      val resultDims = resultJax.shape.as[Seq[Int]]
      val resultTuple = TupleHelpers.createTupleFromSeq[OutT](resultDims)
      val resultShape = Shape[OutT](resultTuple)

      // Get dtype from result
      val dtype = JaxDType.fromJaxDtype(resultJax.dtype)
      new Tensor(resultShape, resultJax, dtype)

  /** Convenience method to JIT compile a function that takes a single tensor. This is equivalent to
    * `Jit[Tensor[InT], OutT](f)` but with a simpler signature.
    *
    * @example
    *   {{{
    *   val jitted = Jit.function[("Batch", "Feature"), ("Batch",)](t => t.sum(axis = 1))
    *   val result = jitted(tensor)
    *   }}}
    */
  def function[InT <: Tuple, OutT <: Tuple](
      f: Tensor[InT] => Tensor[OutT]
  ): Tensor[InT] => Tensor[OutT] =
    apply[Tensor[InT], OutT](f)

  /** Convenience method to JIT compile a function that takes a single tensor. Alias for `function` method for
    * consistency with function2.
    *
    * @example
    *   {{{
    *   val jitted = Jit.function1[("Batch", "Feature"), ("Batch",)](t => t.sum(axis = 1))
    *   val result = jitted(tensor)
    *   }}}
    */
  def function1[InT <: Tuple, OutT <: Tuple](
      f: Tensor[InT] => Tensor[OutT]
  ): Tensor[InT] => Tensor[OutT] =
    apply[Tensor[InT], OutT](f)

  /** Convenience method to JIT compile a function that takes two tensors. This is useful for operations like computing
    * accuracy, losses, or other metrics that compare two tensors (e.g., predictions vs targets).
    *
    * @example
    *   {{{
    *   val jittedAccuracy = Jit.function2[(Sample, Output), (Sample, Output), EmptyTuple](
    *     (predictions, targets) => computeAccuracy(predictions, targets)
    *   )
    *   val accuracy = jittedAccuracy(preds, targets)
    *   }}}
    */
  def function2[In1T <: Tuple, In2T <: Tuple, OutT <: Tuple](
      f: (Tensor[In1T], Tensor[In2T]) => Tensor[OutT]
  ): (Tensor[In1T], Tensor[In2T]) => Tensor[OutT] =
    val jittedTuple = apply[(Tensor[In1T], Tensor[In2T]), OutT] { case (t1, t2) =>
      f(t1, t2)
    }
    (input1: Tensor[In1T], input2: Tensor[In2T]) => jittedTuple((input1, input2))

  /** Convenience method to JIT compile a function that takes parameters and a tensor. This is useful for neural network
    * forward passes and similar operations.
    *
    * The function is compiled to accept params and input as separate arguments, but internally they're treated as a
    * tuple PyTree.
    *
    * @example
    *   {{{
    *   val jitted = Jit.withParams[MLPParams, InputShape, OutputShape](
    *     (params, input) => model.forward(params, input)
    *   )
    *   val output = jitted(myParams, myInput)
    *   }}}
    */
  def withParams[Params: ToPyTree, InT <: Tuple, OutT <: Tuple](
      f: (Params, Tensor[InT]) => Tensor[OutT]
  ): (Params, Tensor[InT]) => Tensor[OutT] =
    val jittedTuple = apply[(Params, Tensor[InT]), OutT] { case (params, tensor) =>
      f(params, tensor)
    }
    (params: Params, input: Tensor[InT]) => jittedTuple((params, input))

  /** JIT compile a gradient step function that returns updated parameters. This is optimized for training loops where
    * you want to compile the entire forward + backward + parameter update in one JIT-compiled function.
    *
    * Unlike other JIT methods, this returns Params instead of a Tensor.
    *
    * @example
    *   {{{
    *   val jittedStep = Jit.gradientStep[MLPParams, BatchInput, BatchLabels](
    *     (params, inputs, labels) => {
    *       val lossFn = (p: MLPParams) => computeLoss(p, inputs, labels)
    *       val grads = Autodiff.grad(lossFn)(params)
    *       optimizer.applyGradients(params, grads)
    *     }
    *   )
    *
    *   // In training loop - typically 10-50x faster!
    *   newParams = jittedStep(oldParams, batchInputs, batchLabels)
    *   }}}
    */
  def gradientStep[Params: ToPyTree, In1T <: Tuple, In2T <: Tuple](
      f: (Params, Tensor[In1T], Tensor[In2T]) => Params
  ): (Params, Tensor[In1T], Tensor[In2T]) => Params =

    var cachedShape1: Option[Shape[In1T]] = None
    var cachedShape2: Option[Shape[In2T]] = None
    var cachedDtype: Option[DType] = None

    val fpy = (paramsPy: Jax.PyDynamic, tensor1Jax: Jax.PyDynamic, tensor2Jax: Jax.PyDynamic) =>
      val shape1 = cachedShape1.getOrElse(throw new IllegalStateException("Shape1 not initialized"))
      val shape2 = cachedShape2.getOrElse(throw new IllegalStateException("Shape2 not initialized"))
      val dt = cachedDtype.getOrElse(DType.Float32)

      val params = ToPyTree[Params].fromPyTree(paramsPy)
      val tensor1 = new Tensor[In1T](shape1, tensor1Jax, dt)
      val tensor2 = new Tensor[In2T](shape2, tensor2Jax, dt)

      val result = f(params, tensor1, tensor2)
      ToPyTree[Params].toPyTree(result) // Return pytree, not tensor!

    val jitted = Jax.jax_helper.jit_fn(fpy)

    (params: Params, input1: Tensor[In1T], input2: Tensor[In2T]) =>
      if cachedShape1.isEmpty then
        cachedShape1 = Some(input1.shape)
        cachedShape2 = Some(input2.shape)
        cachedDtype = Some(input1.dtype)

      val paramsPy = ToPyTree[Params].toPyTree(params)
      val resultPy = jitted(paramsPy, input1.jaxValue, input2.jaxValue)

      ToPyTree[Params].fromPyTree(resultPy)
