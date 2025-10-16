package examples.nn.losses

import shapeful.*

object BinaryCrossEntropy:

  def apply[Output <: Label](
      logits: Tensor1[Output],
      targets: Tensor1[Output]
  ): Tensor0 =
    val logProbs = logits - logits.exp.sum.log
    (targets * logProbs * Tensor0(-1f)).sum
