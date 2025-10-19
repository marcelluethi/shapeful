package shapeful.tensor

import shapeful.Label

/** Axis companion that enables low-ceremony axis creation */
object Axis:
  /** Create an Axis value from a label type */
  inline def apply[A <: Label]: Axis[A] =
    new Axis[A] {}

  extension [A <: Label](axis: Axis[A])
    /** Get the runtime name of the axis from its literal type */
    def name(using v: ValueOf[A]): String = v.value.toString

    /** String representation showing the axis label */
    def asString(using v: ValueOf[A]): String = s"Axis[${v.value}]"

/** An Axis is both a type-level label and has a runtime value. This allows for type-safe axis operations with good type
  * inference.
  */
sealed trait Axis[A <: Label]
