package shapeful.tree

import scala.language.experimental.namedTypeArguments
import shapeful.*

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

// extension method for convenience
extension [P](p: P)(using pt: TensorTree[P])
  def map(f: [T <: Tuple] => Tensor[T] => Tensor[T]): P = pt.map(p, f)
  def zipMap(p2: P, f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): P = pt.zipMap(p, p2, f)
