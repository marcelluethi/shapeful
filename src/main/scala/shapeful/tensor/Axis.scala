package shapeful.tensor

import shapeful.Label

/** Axis companion that enables low-ceremony axis creation */
object Axis:
  /** Create an Axis value from a label type */
  inline def apply[A <: Label]: Axis[A] =
    new Axis[A] {}

/** An Axis is both a type-level label and has a runtime value. This allows for type-safe axis operations with good type
  * inference.
  */
sealed trait Axis[A <: Label]
