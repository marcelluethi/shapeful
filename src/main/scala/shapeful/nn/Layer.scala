package shapeful.nn

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.autodiff.TensorTree
import shapeful.autodiff.ToPyTree
import java.awt.image.ComponentSampleModel

trait Layer[In <: Tuple, Out <: Tuple, Param] extends Function1[Param, Tensor[In] => Tensor[Out]]:

  override def apply(params: Param): Tensor[In] => Tensor[Out] =
    forward(params)

  def forward(params: Param): Tensor[In] => Tensor[Out]

// Convenience trait for common 1D -> 1D layers
trait Layer1D[InDim <: Label, OutDim <: Label, Param] extends Layer[Tuple1[InDim], Tuple1[OutDim], Param]
