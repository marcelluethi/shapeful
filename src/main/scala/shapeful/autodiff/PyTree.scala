package shapeful.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import shapeful.inference.AffineFlow

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import shapeful.tensor.TupleHelpers
import shapeful.jax.JaxDType
import scala.deriving.*
import scala.compiletime.*

trait ToPyTree[P]:
  def toPyTree(p: P): Jax.PyAny
  def fromPyTree(p: Jax.PyAny): P

object ToPyTree:

  def apply[P](using pt: ToPyTree[P]): ToPyTree[P] = pt

  // Keep the tensor instance
  given [T <: Tuple]: ToPyTree[Tensor[T]] with
    def toPyTree(t: Tensor[T]): Jax.PyAny = t.jaxValue
    def fromPyTree(p: Jax.PyAny): Tensor[T] =
      val shapeJax = p.as[Jax.PyDynamic].shape.as[Seq[Int]]
      val shapeTuple = TupleHelpers.createTupleFromSeq[T](shapeJax)
      val shape = Shape[T](shapeTuple)
      val dtype = JaxDType.fromJaxDtype(p.as[Jax.PyDynamic].dtype)
      new Tensor[T](shape, p.as[Jax.PyDynamic], dtype)

  // HIGH PRIORITY: Explicit instance for AffineFlowParams
  given affineFlowParams[From <: shapeful.Label, To <: shapeful.Label]: ToPyTree[AffineFlow.AffineFlowParams[From, To]]
  with
    def toPyTree(p: AffineFlow.AffineFlowParams[From, To]): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(p.weight.jaxValue, p.bias.jaxValue).toPythonProxy)

    def fromPyTree(pyTree: Jax.PyAny): AffineFlow.AffineFlowParams[From, To] =
      val pyTuple = pyTree.as[py.Dynamic]
      val weightPyTree = pyTuple.bracketAccess(0)
      val biasPyTree = pyTuple.bracketAccess(1)

      // Reconstruct tensors with proper shapes
      val weightShapeJax = weightPyTree.shape.as[Seq[Int]]
      val weightDtype = JaxDType.fromJaxDtype(weightPyTree.dtype)
      val weightShapeTuple = TupleHelpers.createTupleFromSeq(weightShapeJax)
      val weight = new Tensor(Shape(weightShapeTuple), weightPyTree, weightDtype)

      val biasShapeJax = biasPyTree.shape.as[Seq[Int]]
      val biasDtype = JaxDType.fromJaxDtype(biasPyTree.dtype)
      val biasShapeTuple = TupleHelpers.createTupleFromSeq(biasShapeJax)
      val bias = new Tensor(Shape(biasShapeTuple), biasPyTree, biasDtype)

      AffineFlow.AffineFlowParams(weight.asInstanceOf[Tensor2[From, To]], bias.asInstanceOf[Tensor1[To]])

  // Fallback instances for basic types
  given ToPyTree[String] with
    def toPyTree(s: String): Jax.PyAny = py.Dynamic.global.str(s)
    def fromPyTree(p: Jax.PyAny): String = p.as[String]

  given ToPyTree[Int] with
    def toPyTree(i: Int): Jax.PyAny = py.Dynamic.global.int(i)
    def fromPyTree(p: Jax.PyAny): Int = p.as[Int]

  given ToPyTree[Float] with
    def toPyTree(f: Float): Jax.PyAny = py.Dynamic.global.float(f)
    def fromPyTree(p: Jax.PyAny): Float = p.as[Float]

  given ToPyTree[Double] with
    def toPyTree(d: Double): Jax.PyAny = py.Dynamic.global.float(d)
    def fromPyTree(p: Jax.PyAny): Double = p.as[Double]

  // Tuple instances - these should have lower priority than specific case classes
  given tupleInstance[A, B](using ta: ToPyTree[A], tb: ToPyTree[B]): ToPyTree[(A, B)] with
    def toPyTree(t: (A, B)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(ta.toPyTree(t._1), tb.toPyTree(t._2)).toPythonProxy)

    def fromPyTree(p: Jax.PyAny): (A, B) =
      val pyTuple = p.as[py.Dynamic]
      val a = ta.fromPyTree(pyTuple.bracketAccess(0))
      val b = tb.fromPyTree(pyTuple.bracketAccess(1))
      (a, b)

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): ToPyTree[P] =
    new ToPyTree[P]:
      def toPyTree(p: P): Jax.PyAny =
        val productElems = p.productIterator.toList
        val pyTreeElems = productElems.map(convertElementToPyTree)
        py.Dynamic.global.tuple(pyTreeElems.toPythonProxy)

      def fromPyTree(pyTree: Jax.PyAny): P =
        val pyTuple = pyTree.as[py.Dynamic]
        val reconstructedTuple = reconstructFields[m.MirroredElemTypes](pyTuple, 0)
        m.fromProduct(reconstructedTuple)

  // Compile-time reconstruction using field types
  inline def reconstructFields[Types <: Tuple](pyTuple: py.Dynamic, index: Int): Tuple =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        EmptyTuple
      case _: (head *: tail) =>
        val elem = reconstructField[head](pyTuple.bracketAccess(index))
        val rest = reconstructFields[tail](pyTuple, index + 1)
        elem *: rest

  inline def reconstructField[T](pyElem: py.Dynamic): T =
    inline erasedValue[T] match
      case _: Tensor[?] =>
        // Reconstruct tensor directly
        val shapeJax = pyElem.shape.as[Seq[Int]]
        val dtype = JaxDType.fromJaxDtype(pyElem.dtype)
        val shapeTuple = TupleHelpers.createTupleFromSeq(shapeJax)
        new Tensor(Shape(shapeTuple), pyElem, dtype).asInstanceOf[T]
      case _: String =>
        pyElem.as[String].asInstanceOf[T]
      case _: Int =>
        pyElem.as[Int].asInstanceOf[T]
      case _: Float =>
        pyElem.as[Float].asInstanceOf[T]
      case _: Double =>
        pyElem.as[Double].asInstanceOf[T]
      case _ =>
        // For complex types (case classes), try to find ToPyTree instance
        compiletime.summonInline[ToPyTree[T]].fromPyTree(pyElem)

  // Convert any element to PyTree recursively - delegates to proper instances
  private def convertElementToPyTree(elem: Any): Jax.PyAny =
    elem match
      case tensor: Tensor[?] =>
        tensor.jaxValue
      case product: Product =>
        // Delegate to proper ToPyTree instance using reflection
        convertProductToPyTree(product)
      case str: String =>
        py.Dynamic.global.str(str)
      case int: Int =>
        py.Dynamic.global.int(int)
      case float: Float =>
        py.Dynamic.global.float(float)
      case double: Double =>
        py.Dynamic.global.float(double)
      case other =>
        throw new IllegalArgumentException(s"Cannot convert $other to PyTree")

  // Helper to convert Product types generically
  private def convertProductToPyTree(product: Product): Jax.PyAny =
    // Generic product conversion - always recursive
    val nestedElems = product.productIterator.toList
    val nestedPyElems = nestedElems.map(convertElementToPyTree)
    py.Dynamic.global.tuple(nestedPyElems.toPythonProxy)
