package shapeful.inference.flows

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.inference.flows.Flow
import shapeful.autodiff.TensorTree
import shapeful.autodiff.ToPyTree

class IdentityFlow[Dim <: Label] extends Flow[Dim, Dim, IdentityFlow.Params]:
  def forwardSample(x: Tensor1[Dim])(params: IdentityFlow.Params): Tensor1[Dim] = x

object IdentityFlow:
  case class Params() derives TensorTree, ToPyTree

  def initialParams = Params()
