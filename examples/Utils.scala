package examples

object Utils:
  def timeIt[T](block: => T): (T, Long) =
    val start = System.currentTimeMillis()
    val result = block
    val end = System.currentTimeMillis()
    (result, end - start)
