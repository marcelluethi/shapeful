// package src.main.scala.distributions

// object DistributionPlayground extends App:
//   import shapeful.*
//   import shapeful.Conversions.given
//   import shapeful.stats.*
//   import shapeful.random.Random
//   import shapeful.tensor.Shape

//   val key = Random.Key.fromTime()

//   type Samples = "samples"
//   type Features = "features"

//   // create training data for a linear regresssion
//   // y = 3.0 * x + noise
//   val trueSlope = Tensor1(Axis[Features], Array(3.0f, 3.0f))
//   val bias = Tensor0(3f)
//   val numPoints = 100
//   val xData = Random.uniform(key, Shape(Axis[Samples] -> numPoints))
//     .vmap(Axis[Samples])(s => Random.normal(Random.Key.fromTime(), Shape(Axis[Features] -> 2)))

//   val noise = IndependentNormal(
//     mean = Tensor.zeros(Shape(Axis[Samples] -> numPoints)),
//     stddev = Tensor.ones(Shape(Axis[Samples] -> numPoints)) :* Tensor0(2f),
//   ).sample(key)

//   val yData = (xData.contract(Axis[Features])(trueSlope) :+ bias) + noise

//   class Model(val weight : Tensor1[Features], val bias: Tensor0):

//     def predict(x: Tensor2[Samples, Features]): Tensor1[Samples] =
//       x.contract(Axis[Features])(weight) :+ bias

//     def likelihood(x: Tensor2[Samples, Features], y: Tensor1[Samples]): Tensor0 =
//       val preds = predict(x)
//       val dist = IndependentNormal(
//         mean = preds,
//         stddev = Tensor.ones(Shape(Axis[Samples] -> numPoints)) :* Tensor0(2f),
//       )
//       dist.logProb(y)

//   val w0 = Tensor1(Axis[Features], Array(0.0f, 0.0f))
//   val b0 = Tensor0(0f)
//   case class Params(weight: Tensor1[Features], bias: Tensor0) derives ToPyTree

//   def optFun(params: Params): Tensor0 =
//     val model = Model(params.weight, params.bias)
//     -model.likelihood(xData, yData) / Tensor0(numPoints.toFloat)
