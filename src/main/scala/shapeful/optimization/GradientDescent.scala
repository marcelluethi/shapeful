package shapeful.optimization

import shapeful.tensor.Tensor.Tensor0

import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor
import shapeful.tensor.multScalar
import shapeful.tensor.add

// trait Tensor1Conv[A <: Tuple] {
//     def toTensor1(f : A => Tensor0) : Tensor1[Float] => Tensor0
//     def fromTensor1(t : Tensor1[Float] => Tensor0) : 
// }

// class GradientDescent(lr_ : Float, steps : Int) :
    
//     val lr = Tensor(lr_, requiresGrad = false)

//     def optimize[A <: Tuple, B <: Tuple](f :  DifferentiableFunction[(A, B)], init : (Tensor[A], Tensor[B])) : (Tensor[A], Tensor[B]) =
//         val (w, b) = (0 until steps).foldLeft(init) { case ((w, b), i) =>
//             if i % 100 == 0 then
//                 println(s"Iteration $i: $w $b")
            


//             val (wg, wb) = f.deriv(w, b)
        
//             val neww = w.add(wg.multScalar(lr)).copy(requiresGrad = true)
//             val newb = b.add(wb.multScalar(lr)).copy(requiresGrad = true)
        
//             (neww, newb)
//         }
//         (w, b)

  // optimize
//   val lr : Tensor0 = Tensor(0.01f, requiresGrad = false)
//   val w0 : Tensor1[Space] = Tensor(0.0, requiresGrad = true)
//   val b0 : Tensor0 = Tensor(2.0, requiresGrad = true)

//   val (w, b) = (0 until 1000).foldLeft((w0, b0)) { case ((w, b), i) =>
//     if i % 100 == 0 then
//         println(s"Iteration $i: $w $b")
  
//     val d = deriv(f)
//     val (wg, wb) = d(w, b)
  
//     val neww = w.add(wg.multScalar(lr)).copy(requiresGrad = true)
//     val newb = b.add(wb.multScalar(lr)).copy(requiresGrad = true)
  
//     (neww, newb)
//   }
