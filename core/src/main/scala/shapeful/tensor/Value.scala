package shapeful.tensor

// Typeclass linking DType singleton types to runtime DType values
trait Value[V]:
  def dtype: DType

object Value:
  // Helper to summon instances
  inline def summon[V](using v: Value[V]): Value[V] = v
  
  // Simple two-line pattern for semantic types:
  //   trait Y
  //   object Y extends Value.As[Y, Float32]
  abstract class As[V, BaseType](using base: Value[BaseType]) extends Value[V]:
    def dtype: DType = base.dtype
    given Value[V] = this
  
  // Given instances for all supported DType enum cases
  given Value[DType.Float32.type] with
    def dtype: DType = DType.Float32

  given Value[DType.Float64.type] with
    def dtype: DType = DType.Float64

  given Value[DType.Int32.type] with
    def dtype: DType = DType.Int32

  given Value[DType.Int64.type] with
    def dtype: DType = DType.Int64

  given Value[DType.Int16.type] with
    def dtype: DType = DType.Int16

  given Value[DType.Int8.type] with
    def dtype: DType = DType.Int8

  given Value[DType.UInt32.type] with
    def dtype: DType = DType.UInt32

  given Value[DType.UInt16.type] with
    def dtype: DType = DType.UInt16

  given Value[DType.UInt8.type] with
    def dtype: DType = DType.UInt8

  given Value[DType.Bool.type] with
    def dtype: DType = DType.Bool
