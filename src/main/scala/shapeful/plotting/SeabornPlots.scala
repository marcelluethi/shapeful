package shapeful.plotting

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import shapeful.*

object SeabornPlots:

  // Initialize Python modules
  private val plt = py.module("matplotlib.pyplot")
  private val sns = py.module("seaborn")
  private val np = py.module("numpy")

  // Configure matplotlib for X11 display
  def setupX11Display(): Unit =
    try
      plt.switch_backend("TkAgg") // Try TkAgg first (most common for X11)
      println("Matplotlib backend set to TkAgg for X11 display")
    catch
      case _: Exception =>
        try
          plt.switch_backend("Qt5Agg") // Fallback to Qt5Agg
          println("Matplotlib backend set to Qt5Agg for X11 display")
        catch
          case _: Exception =>
            plt.switch_backend("Agg") // Fallback to file-only
            println("X11 backends not available, using Agg (file-only)")

  // Set seaborn style
  sns.set_style("whitegrid")

  case class PlotConfig(
      figsize: (Int, Int) = (10, 6),
      title: Option[String] = None,
      xlabel: Option[String] = None,
      ylabel: Option[String] = None,
      color: Option[String] = None,
      alpha: Double = 1.0
  )

  /** Convert a Tensor0 to a Python float
    */
  def tensor0ToPython(tensor: Tensor0): py.Any =
    py.Dynamic.global.float(tensor.toFloat)

  /** Convert a Tensor1 to a Python numpy array
    */
  def tensor1ToPython[A <: Label](tensor: Tensor1[A]): py.Any =
    val values = (0 until tensor.shape.size).map(i => tensor.at(Tuple1(i)).get.toFloat).toSeq.toPythonProxy
    np.array(values)

  /** Create a scatter plot of two 1D tensors
    */
  def scatterPlot[A <: Label, B <: Label](
      x: Tensor1[A],
      y: Tensor1[B],
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val xData = tensor1ToPython(x)
    val yData = tensor1ToPython(y)

    // Simple plot call without keyword unpacking
    config.color match
      case Some(c) => sns.scatterplot(x = xData, y = yData, color = c, alpha = config.alpha)
      case None    => sns.scatterplot(x = xData, y = yData, alpha = config.alpha)

    config.title.foreach(plt.title(_))
    config.xlabel.foreach(plt.xlabel(_))
    config.ylabel.foreach(plt.ylabel(_))

    plt.tight_layout()
    plt.show()

  /** Create a line plot of two 1D tensors
    */
  def linePlot[A <: Label, B <: Label](
      x: Tensor1[A],
      y: Tensor1[B],
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val xData = tensor1ToPython(x)
    val yData = tensor1ToPython(y)

    config.color match
      case Some(c) => sns.lineplot(x = xData, y = yData, color = c, alpha = config.alpha)
      case None    => sns.lineplot(x = xData, y = yData, alpha = config.alpha)

    config.title.foreach(plt.title(_))
    config.xlabel.foreach(plt.xlabel(_))
    config.ylabel.foreach(plt.ylabel(_))

    plt.tight_layout()
    plt.show()

  /** Create a regression plot with confidence interval
    */
  def regressionPlot[A <: Label, B <: Label](
      x: Tensor1[A],
      y: Tensor1[B],
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val xData = tensor1ToPython(x)
    val yData = tensor1ToPython(y)

    config.color match
      case Some(c) => sns.regplot(x = xData, y = yData, color = c)
      case None    => sns.regplot(x = xData, y = yData)

    config.title.foreach(plt.title(_))
    config.xlabel.foreach(plt.xlabel(_))
    config.ylabel.foreach(plt.ylabel(_))

    plt.tight_layout()
    plt.show()

  /** Plot optimization trajectory (parameter values over iterations)
    */
  def optimizationTrajectory(
      iterations: Seq[Int],
      parameterValues: Seq[Float],
      parameterName: String,
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val iterData = np.array(iterations.toPythonProxy)
    val paramData = np.array(parameterValues.toPythonProxy)

    config.color match
      case Some(c) => sns.lineplot(x = iterData, y = paramData, color = c, alpha = config.alpha)
      case None    => sns.lineplot(x = iterData, y = paramData, alpha = config.alpha)

    plt.title(config.title.getOrElse(s"$parameterName Over Iterations"))
    plt.xlabel("Iteration")
    plt.ylabel(parameterName)

    plt.tight_layout()
    plt.show()

  /** Plot multiple parameters on the same plot
    */
  def multiParameterTrajectory(
      iterations: Seq[Int],
      parameters: Map[String, Seq[Float]],
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val iterData = np.array(iterations.toPythonProxy)

    parameters.foreach { case (name, values) =>
      val paramData = np.array(values.toPythonProxy)
      sns.lineplot(x = iterData, y = paramData, label = name, alpha = config.alpha)
    }

    config.title.foreach(plt.title(_))
    plt.xlabel(config.xlabel.getOrElse("Iteration"))
    plt.ylabel(config.ylabel.getOrElse("Parameter Value"))
    plt.legend()

    plt.tight_layout()
    plt.show()

  /** Create a histogram of a 1D tensor
    */
  def histogram[A <: Label](
      data: Tensor1[A],
      bins: Int = 30,
      config: PlotConfig = PlotConfig()
  ): Unit =
    val figsize = py.Dynamic.global.tuple(Seq(config.figsize._1, config.figsize._2).toPythonProxy)
    plt.figure(figsize = figsize)

    val values = tensor1ToPython(data)

    config.color match
      case Some(c) => sns.histplot(values, bins = bins, color = c, alpha = config.alpha)
      case None    => sns.histplot(values, bins = bins, alpha = config.alpha)

    config.title.foreach(plt.title(_))
    config.xlabel.foreach(plt.xlabel(_))
    config.ylabel.foreach(plt.ylabel(_))

    plt.tight_layout()
    plt.show()

  /** Save the current plot to a file
    */
  def savePlot(filename: String, dpi: Int = 300): Unit =
    plt.savefig(filename, dpi = dpi, bbox_inches = "tight")

  /** Clear the current plot
    */
  def clearPlot(): Unit =
    plt.clf()
