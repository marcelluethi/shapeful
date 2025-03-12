package shapeful.tensor

import shapeful.tensor.Tensor.Tensor1

object Tensor1Ops {
    extension[A] (tensor: Tensor1[A]) 

        def dot[B](other : Tensor[Tuple1[B]]) : Tensor[Tuple1[A]] =
            val newt = tensor.stensor `@` other.stensor
            new Tensor[(Tuple1[A])](newt, List(newt.shape.head))

        def cov : Tensor[(A, A)] =
            val n = tensor.shape.head
            val newt = tensor.stensor.matmul(tensor.stensor.t) * (1.0f / n)
            new Tensor[(A, A)](newt, List(newt.shape.head))

}
