package shapeful.autodiff

import shapeful.*
import scala.deriving.*
import scala.compiletime.*

trait TensorTree[P]:
  def map(p: P, f: [T <: Tuple] => Tensor[T] => Tensor[T]): P
  def zipMap(p1: P, p2: P, f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): P

object TensorTree:
  def apply[P](using pt: TensorTree[P]): TensorTree[P] = pt

  given [Q <: Tuple]: TensorTree[Tensor[Q]] = new TensorTree[Tensor[Q]]:
    def map(t: Tensor[Q], f: [T <: Tuple] => Tensor[T] => Tensor[T]): Tensor[Q] =
      f(t)

    def zipMap(p1: Tensor[Q], p2: Tensor[Q], f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): Tensor[Q] =
      f(p1, p2)

  // Derivation for product types (case classes)
  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorTree[P] =
    new TensorTree[P]:
      def map(p: P, f: [T <: Tuple] => Tensor[T] => Tensor[T]): P =
        val productElems = p.productIterator.toList
        val mappedElems = mapTuple(productElems, f)
        m.fromProduct(Tuple.fromArray(mappedElems.toArray))

      def zipMap(p1: P, p2: P, f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): P =
        val elems1 = p1.productIterator.toList
        val elems2 = p2.productIterator.toList
        val zippedElems = zipMapTuple(elems1.zip(elems2), f)
        m.fromProduct(Tuple.fromArray(zippedElems.toArray))

  private def mapTuple(elems: List[Any], f: [T <: Tuple] => Tensor[T] => Tensor[T]): List[Any] =
    elems.map {
      case tensor: Tensor[?] => f(tensor.asInstanceOf[Tensor[Tuple]])
      case product: Product  =>
        // Recursively handle nested Product types by processing their elements directly
        mapProductDirectly(product, f)
      case other => other
    }

  private def zipMapTuple(pairs: List[(Any, Any)], f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): List[Any] =
    pairs.map {
      case (t1: Tensor[?], t2: Tensor[?]) =>
        f(t1.asInstanceOf[Tensor[Tuple]], t2.asInstanceOf[Tensor[Tuple]])
      case (p1: Product, p2: Product) =>
        // Recursively handle nested Product types by processing their elements directly
        zipMapProductDirectly(p1, p2, f)
      case (other1, other2) =>
        // For non-tensor, non-product fields, return the first one
        other1
    }

  // Helper to recursively map over product elements without type class summoning
  private def mapProductDirectly(product: Product, f: [T <: Tuple] => Tensor[T] => Tensor[T]): Any =
    val productElems = product.productIterator.toList
    val mappedElems = mapTuple(productElems, f)

    // Use reflection to reconstruct the product
    val clazz = product.getClass
    val constructors = clazz.getConstructors

    // Try to find the right constructor and handle inner class case
    val constructor = constructors.find(_.getParameterCount == mappedElems.length).getOrElse {
      // If no constructor matches the mapped elements count, it might be an inner class
      // Inner classes have an implicit outer instance parameter
      constructors.find(_.getParameterCount == mappedElems.length + 1).getOrElse(constructors.head)
    }

    try
      if constructor.getParameterCount == mappedElems.length then
        // Regular case class
        constructor.newInstance(mappedElems.map(_.asInstanceOf[Object])*)
      else
        // Inner class - need to get the outer instance
        val outerInstance = product.getClass.getDeclaredFields
          .find(_.getName.contains("$outer"))
          .map { field =>
            field.setAccessible(true)
            field.get(product)
          }
          .getOrElse(throw new IllegalArgumentException("Cannot find outer instance for inner class"))

        constructor.newInstance((outerInstance +: mappedElems).map(_.asInstanceOf[Object])*)
    catch
      case e: IllegalArgumentException =>
        println(s"Constructor expects ${constructor.getParameterCount} args, got ${mappedElems.length}")
        println(s"Constructor parameter types: ${constructor.getParameterTypes.mkString(", ")}")
        println(s"Provided arg types: ${mappedElems.map(_.getClass.getName).mkString(", ")}")
        throw e

  // Helper to recursively zipMap over product elements without type class summoning
  private def zipMapProductDirectly(
      p1: Product,
      p2: Product,
      f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]
  ): Any =
    if p1.getClass != p2.getClass then
      throw new IllegalArgumentException("Products must be of the same type for zipMap")

    val elems1 = p1.productIterator.toList
    val elems2 = p2.productIterator.toList

    if elems1.length != elems2.length then
      throw new IllegalArgumentException("Products must have the same arity for zipMap")

    val zippedElems = zipMapTuple(elems1.zip(elems2), f)

    // Use reflection to reconstruct the product with inner class handling
    val clazz = p1.getClass
    val constructors = clazz.getConstructors

    val constructor = constructors.find(_.getParameterCount == zippedElems.length).getOrElse {
      constructors.find(_.getParameterCount == zippedElems.length + 1).getOrElse(constructors.head)
    }

    try
      if constructor.getParameterCount == zippedElems.length then
        // Regular case class
        constructor.newInstance(zippedElems.map(_.asInstanceOf[Object])*)
      else
        // Inner class - need to get the outer instance
        val outerInstance = p1.getClass.getDeclaredFields
          .find(_.getName.contains("$outer"))
          .map { field =>
            field.setAccessible(true)
            field.get(p1)
          }
          .getOrElse(throw new IllegalArgumentException("Cannot find outer instance for inner class"))

        constructor.newInstance((outerInstance +: zippedElems).map(_.asInstanceOf[Object])*)
    catch
      case e: IllegalArgumentException =>
        println(s"Constructor expects ${constructor.getParameterCount} args, got ${zippedElems.length}")
        println(s"Constructor parameter types: ${constructor.getParameterTypes.mkString(", ")}")
        println(s"Provided arg types: ${zippedElems.map(_.getClass.getName).mkString(", ")}")
        throw e

// extension method for convenience
extension [P](p: P)(using pt: TensorTree[P])
  def map(f: [T <: Tuple] => Tensor[T] => Tensor[T]): P = pt.map(p, f)
  def zipMap(p2: P, f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): P = pt.zipMap(p, p2, f)
