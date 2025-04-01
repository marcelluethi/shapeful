package shapeful.inference

import shapeful.tensor.Tensor0
import torch.Float32
import shapeful.autodiff.Params

import shapeful.tensor.TensorOps.sub

object MetropolisHastings:

    def sample(targetLogDensity: Params => Tensor0[Float32], proposal: Params => Params, initial: Params) : Iterator[Params] =
        Iterator.iterate(initial) { current =>
            val proposed = proposal(current)
            val logAcceptance = targetLogDensity(proposed).sub(targetLogDensity(current)).item
            val acceptance = Math.min(1, Math.exp(logAcceptance))
            if (Math.random() < acceptance) proposed else current
        }
