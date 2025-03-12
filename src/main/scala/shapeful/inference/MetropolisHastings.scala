package shapeful.inference

import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.IsTensorTuple
import shapeful.tensor.Tensor0Ops.*
import shapeful.tensor.TensorOps.*


object MetropolisHastings:

    def sample[Tensors <: Tuple : IsTensorTuple](targetLogDensity: Tensors => Tensor0, proposal: Tensors => Tensors, initial: Tensors) : Iterator[Tensors] =
        Iterator.iterate(initial) { current =>
            val proposed = proposal(current)
            val logAcceptance = targetLogDensity(proposed).sub(targetLogDensity(current)).item
            val acceptance = Math.min(1, Math.exp(logAcceptance))
            if (Math.random() < acceptance) proposed else current
        }
