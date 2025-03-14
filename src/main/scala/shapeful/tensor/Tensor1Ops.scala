package shapeful.tensor

import shapeful.tensor.Tensor.Tensor1

object Tensor1Ops {
    extension[A] (tensor: Tensor1[A]) 

        def dot[B](other : Tensor[Tuple1[B]]) : Tensor[Tuple1[A]] =
            val newt = tensor.stensor `@` other.stensor
            val newShape = Shape[A](newt.shape.head)
            new Tensor[Tuple1[A]](newShape, newt)

        def cov : Tensor[(A, A)] =
            val n = tensor.dim[A]
            val newt = tensor.stensor.matmul(tensor.stensor.t) * (1.0f / n)
            val newShape = Shape[A, A](n, n)
            new Tensor(newShape, newt)

}
