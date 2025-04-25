package shapeful.inference

import shapeful.autodiff.Autodiff
import shapeful.autodiff.Params
import shapeful.distributions.MVNormal
import shapeful.distributions.Normal
import shapeful.nn.AffineTransformation
import shapeful.nn.Transformation2
import shapeful.tensor.->>
import shapeful.tensor.FromRepr
import shapeful.tensor.Shape
import shapeful.tensor.Shape3
import shapeful.tensor.Tensor0
import shapeful.tensor.Tensor1
import shapeful.tensor.Tensor1Ops.mean
import shapeful.tensor.Tensor2
import shapeful.tensor.Tensor2Ops.*
import shapeful.tensor.Tensor2Ops.addTensor1
import shapeful.tensor.Tensor2Ops.map
import shapeful.tensor.Tensor2Ops.matmul
import shapeful.tensor.Tensor3
import shapeful.tensor.Tensor3Ops.{reduce, map}
import shapeful.tensor.Tensor2Ops.reduce
import shapeful.tensor.TensorOps.add
import shapeful.tensor.TensorOps.log
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.Variable1
import shapeful.tensor.Variable2
import torch.DType.float32
import torch.Float32
import shapeful.tensor.TensorOps.abs
import shapeful.Label

class AffineFlow[From <: Label, To <: Label](w : Tensor2[From, To, Float32], b : Tensor1[To, Float32]):

    def jac[Sample <: Label](x : Tensor2[Sample, From, Float32]) : Tensor3[Sample, From, To, Float32] = 
        val j = Tensor3(
            new Shape3[Sample, From, To](x.shape.dim[Sample], w.shape.dim[From], w.shape.dim[To]),
            0f
        )
        j.map[Sample](row => 
            row.add(w)
        )


    def forward[Sample <: Label](x : Tensor2[Sample, From, Float32]) : Tensor2[Sample, To, Float32] = 
        x.matmul(w).map[Sample](row => row.add(b))
        
    

def normalizingFlow(
    baseDist : MVNormal["latent"],
    baseSamples : Tensor2["samples", "latent", Float32],
    posteriorLogProb : Tensor1["parameters", Float32] => Tensor0[Float32]
)(using fromRepr : FromRepr[Float32, Tensor3["samples", "latent", "parameters", Float32]], 
        fromRepr1 : FromRepr[Float32, Tensor1["samples", Float32]]): Params => Tensor0[Float32 ]= 
    params =>
   
        val flow = AffineFlow(
            params.get[Variable2["latent", "parameters"]]("w1"),
            params.get[Variable1["parameters"]]("b1") 
        )

        val transformedSamples = flow.forward(baseSamples)
        //println("trasnsformed" +transformedSamples)
        val jac = flow.jac(baseSamples)
        val logdet : Tensor1["samples", Float32] = jac.reduce["samples"](t => t.det.abs).log
        val baseLogProb = baseSamples.reduce["samples"](baseDist.logpdf)
        val logProdTransformed = transformedSamples.reduce["samples"](posteriorLogProb)
        baseLogProb.sub(logdet).sub(logProdTransformed).mean
        
        

 
    
    