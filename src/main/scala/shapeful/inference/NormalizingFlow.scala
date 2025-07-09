package shapeful.inference
import scala.language.experimental.namedTypeArguments
import shapeful.Label
import shapeful.*
import shapeful.distributions.MVNormal

class AffineFlow[From <: Label, To <: Label](w : Tensor2[From, To], b : Tensor1[To]):

    def jac[Sample <: Label](x : Tensor2[Sample, From]) : Tensor3[Sample, From, To] = 
        val j = Tensor.zeros(
            Shape3[Sample, From, To](x.shape.dim[Sample], w.shape.dim[From], w.shape.dim[To]),
            0f
        )
        j.vmap[VmapAxis=Sample](row => row + w
        )


    def forward[Sample <: Label](x : Tensor2[Sample, From]) : Tensor2[Sample, To] = 
        x.vmap[VmapAxis=Sample](sample
             => sample.matmul(w) + b
        )
        
    
type FlowParams[Latent <: Label, Output <: Label] = (Tensor2[Latent, Output], Tensor1[Output])

def normalizingFlow[Samples <: Label, Latent <: Label, Output <: Label](
    baseDist : MVNormal[Latent],
    numSamples : Int,
    posteriorLogProb : Tensor1[Output] => Tensor0
): FlowParams[Latent, Output] => Tensor0 = 
    params =>
        val (w, b) = params
        
        val flow = AffineFlow(w, b)

        val baseSamples = baseDist.sample[Samples](numSamples)
        val transformedSamples = flow.forward(baseSamples)
        val jac = flow.jac(baseSamples)
        val logdet  = jac.vmap[VmapAxis = Samples](t => t.det.abs.log)
        val baseLogProb = baseSamples.vmap[VmapAxis = Samples](baseDist.logpdf)
        val logProdTransformed = transformedSamples.vmap[VmapAxis = Samples](posteriorLogProb)
        (baseLogProb -  logdet - logProdTransformed).mean
        
        
