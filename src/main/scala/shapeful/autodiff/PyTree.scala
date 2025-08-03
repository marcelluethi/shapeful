package shapeful.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

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

  // Tuple instances - these should have lower priority than specific case classes
  given tupleInstance[A, B](using ta: ToPyTree[A], tb: ToPyTree[B]): ToPyTree[(A, B)] with
    def toPyTree(t: (A, B)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(ta.toPyTree(t._1), tb.toPyTree(t._2)).toPythonProxy)

    def fromPyTree(p: Jax.PyAny): (A, B) =
      val pyTuple = p.as[py.Dynamic]
      val a = ta.fromPyTree(pyTuple.bracketAccess(0))
      val b = tb.fromPyTree(pyTuple.bracketAccess(1))
      (a, b)


 // Add this instance for Seq support
  given seqInstance[T](using elemInstance: ToPyTree[T]): ToPyTree[Seq[T]] with
    def toPyTree(seq: Seq[T]): Jax.PyAny =
      val pyElements = seq.map(elemInstance.toPyTree)
      py.Dynamic.global.tuple(pyElements.toPythonProxy)

    def fromPyTree(p: Jax.PyAny): Seq[T] =
      val pyTuple = p.as[py.Dynamic]
      val length = pyTuple.as[py.Dynamic].__len__().as[Int]
      (0 until length).map { i =>
        elemInstance.fromPyTree(pyTuple.bracketAccess(i))
      }.toSeq

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): ToPyTree[P] =
    new ToPyTree[P]:
      def toPyTree(p: P): Jax.PyAny =
        val productElems = Tuple.fromProductTyped(p)
        val pyTreeElems = convertFieldsAtCompileTime[m.MirroredElemTypes](productElems)
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

  // Compile-time field conversion
  inline def convertFieldsAtCompileTime[Types <: Tuple](fields: Types): List[Jax.PyAny] =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        Nil
      case _: (head *: tail) =>
        val headElem = fields.asInstanceOf[head *: tail].head
        val tailElems = fields.asInstanceOf[head *: tail].tail
        val headPy = convertSingleField[head](headElem)
        val tailPy = convertFieldsAtCompileTime[tail](tailElems)
        headPy :: tailPy

  inline def convertSingleField[T](elem: T): Jax.PyAny =
    inline erasedValue[T] match
      case _: Tensor[?] =>
        elem.asInstanceOf[Tensor[?]].jaxValue
      case _: String =>
        py.Dynamic.global.str(elem.asInstanceOf[String])
      case _: Int =>
        py.Dynamic.global.int(elem.asInstanceOf[Int])
      case _: Float =>
        py.Dynamic.global.float(elem.asInstanceOf[Float])
      case _: Double =>
        py.Dynamic.global.float(elem.asInstanceOf[Double])
      case _ =>
        // Use compile-time instance lookup
        compiletime.summonInline[ToPyTree[T]].toPyTree(elem)
