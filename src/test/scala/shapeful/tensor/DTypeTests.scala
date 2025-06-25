package shapeful.tensor

import munit.FunSuite

class DTypeTests extends FunSuite:

  test("DType should have correct name and size for all types") {
    assertEquals(DType.Float32.name, "float32")
    assertEquals(DType.Float32.size, 4)

    assertEquals(DType.Float64.name, "float64")
    assertEquals(DType.Float64.size, 8)

    assertEquals(DType.Int32.name, "int32")
    assertEquals(DType.Int32.size, 4)

    assertEquals(DType.Int64.name, "int64")
    assertEquals(DType.Int64.size, 8)

    assertEquals(DType.Int16.name, "int16")
    assertEquals(DType.Int16.size, 2)

    assertEquals(DType.Int8.name, "int8")
    assertEquals(DType.Int8.size, 1)

    assertEquals(DType.UInt32.name, "uint32")
    assertEquals(DType.UInt32.size, 4)

    assertEquals(DType.UInt16.name, "uint16")
    assertEquals(DType.UInt16.size, 2)

    assertEquals(DType.UInt8.name, "uint8")
    assertEquals(DType.UInt8.size, 1)

    assertEquals(DType.Bool.name, "bool")
    assertEquals(DType.Bool.size, 1)

    assertEquals(DType.Complex64.name, "complex64")
    assertEquals(DType.Complex64.size, 8)

    assertEquals(DType.Complex128.name, "complex128")
    assertEquals(DType.Complex128.size, 16)
  }

  test("DType should provide correct zero values") {
    assertEquals(DType.Float32.zero, 0.0f)
    assertEquals(DType.Float64.zero, 0.0)
    assertEquals(DType.Int32.zero, 0)
    assertEquals(DType.Int64.zero, 0L)
    assertEquals(DType.Int16.zero, 0.toShort)
    assertEquals(DType.Int8.zero, 0.toByte)
    assertEquals(DType.UInt32.zero, 0)
    assertEquals(DType.UInt16.zero, 0.toShort)
    assertEquals(DType.UInt8.zero, 0.toByte)
    assertEquals(DType.Bool.zero, false)
    assertEquals(DType.Complex64.zero, 0.0f)
    assertEquals(DType.Complex128.zero, 0.0)
  }

  test("DType should correctly identify floating point types") {
    assert(DType.Float32.isFloating)
    assert(DType.Float64.isFloating)
    assert(!DType.Int32.isFloating)
    assert(!DType.Int64.isFloating)
    assert(!DType.Bool.isFloating)
    assert(!DType.Complex64.isFloating)
    assert(!DType.Complex128.isFloating)
  }

  test("DType should correctly identify integer types") {
    assert(DType.Int32.isInteger)
    assert(DType.Int64.isInteger)
    assert(DType.Int16.isInteger)
    assert(DType.Int8.isInteger)
    assert(DType.UInt32.isInteger)
    assert(DType.UInt16.isInteger)
    assert(DType.UInt8.isInteger)
    assert(!DType.Float32.isInteger)
    assert(!DType.Float64.isInteger)
    assert(!DType.Bool.isInteger)
    assert(!DType.Complex64.isInteger)
    assert(!DType.Complex128.isInteger)
  }

  test("DType should correctly identify complex types") {
    assert(DType.Complex64.isComplex)
    assert(DType.Complex128.isComplex)
    assert(!DType.Float32.isComplex)
    assert(!DType.Float64.isComplex)
    assert(!DType.Int32.isComplex)
    assert(!DType.Bool.isComplex)
  }

  test("DType should correctly identify unsigned types") {
    assert(DType.UInt8.isUnsigned)
    assert(DType.UInt16.isUnsigned)
    assert(DType.UInt32.isUnsigned)
    assert(DType.Bool.isUnsigned)
    assert(!DType.Int8.isUnsigned)
    assert(!DType.Int16.isUnsigned)
    assert(!DType.Int32.isUnsigned)
    assert(!DType.Int64.isUnsigned)
    assert(!DType.Float32.isUnsigned)
    assert(!DType.Float64.isUnsigned)
  }

  test("DType should correctly identify signed types") {
    assert(DType.Int8.isSigned)
    assert(DType.Int16.isSigned)
    assert(DType.Int32.isSigned)
    assert(DType.Int64.isSigned)
    assert(DType.Float32.isSigned)
    assert(DType.Float64.isSigned)
    assert(DType.Complex64.isSigned)
    assert(DType.Complex128.isSigned)
    assert(!DType.UInt8.isSigned)
    assert(!DType.UInt16.isSigned)
    assert(!DType.UInt32.isSigned)
    assert(!DType.Bool.isSigned)
  }

  test("DType should provide correct element types") {
    assertEquals(DType.Float32.elementType, "float")
    assertEquals(DType.Float64.elementType, "float")
    assertEquals(DType.Int32.elementType, "int")
    assertEquals(DType.Int64.elementType, "int")
    assertEquals(DType.Int16.elementType, "int")
    assertEquals(DType.Int8.elementType, "int")
    assertEquals(DType.UInt32.elementType, "uint")
    assertEquals(DType.UInt16.elementType, "uint")
    assertEquals(DType.UInt8.elementType, "uint")
    assertEquals(DType.Bool.elementType, "bool")
    assertEquals(DType.Complex64.elementType, "complex")
    assertEquals(DType.Complex128.elementType, "complex")
  }

  test("DType should provide correct unsigned conversions") {
    assertEquals(DType.Int8.toUnsigned, Some(DType.UInt8))
    assertEquals(DType.Int16.toUnsigned, Some(DType.UInt16))
    assertEquals(DType.Int32.toUnsigned, Some(DType.UInt32))
    assertEquals(DType.Int64.toUnsigned, None)
    assertEquals(DType.Float32.toUnsigned, None)
    assertEquals(DType.Bool.toUnsigned, None)
  }

  test("DType should provide correct signed conversions") {
    assertEquals(DType.UInt8.toSigned, Some(DType.Int8))
    assertEquals(DType.UInt16.toSigned, Some(DType.Int16))
    assertEquals(DType.UInt32.toSigned, Some(DType.Int32))
    assertEquals(DType.Int32.toSigned, None)
    assertEquals(DType.Float32.toSigned, None)
    assertEquals(DType.Bool.toSigned, None)
  }

  test("DType should calculate correct bit widths") {
    assertEquals(DType.Int8.bitWidth, 8)
    assertEquals(DType.Int16.bitWidth, 16)
    assertEquals(DType.Int32.bitWidth, 32)
    assertEquals(DType.Int64.bitWidth, 64)
    assertEquals(DType.Float32.bitWidth, 32)
    assertEquals(DType.Float64.bitWidth, 64)
    assertEquals(DType.Bool.bitWidth, 8)
    assertEquals(DType.Complex64.bitWidth, 64)
    assertEquals(DType.Complex128.bitWidth, 128)
  }

  test("DType.promoteTypes should handle same types correctly") {
    assertEquals(DType.promoteTypes(DType.Int32, DType.Int32), DType.Int32)
    assertEquals(DType.promoteTypes(DType.Float64, DType.Float64), DType.Float64)
  }

  test("DType.promoteTypes should promote bool to any other type") {
    assertEquals(DType.promoteTypes(DType.Bool, DType.Int32), DType.Int32)
    assertEquals(DType.promoteTypes(DType.Float32, DType.Bool), DType.Float32)
    assertEquals(DType.promoteTypes(DType.Bool, DType.Complex64), DType.Complex64)
  }

  test("DType.promoteTypes should promote to complex types correctly") {
    assertEquals(DType.promoteTypes(DType.Complex128, DType.Float64), DType.Complex128)
    assertEquals(DType.promoteTypes(DType.Int32, DType.Complex64), DType.Complex64)
    assertEquals(DType.promoteTypes(DType.Complex64, DType.Complex128), DType.Complex128)
  }

  test("DType.promoteTypes should promote to floating point types correctly") {
    assertEquals(DType.promoteTypes(DType.Float64, DType.Int32), DType.Float64)
    assertEquals(DType.promoteTypes(DType.Float32, DType.Float64), DType.Float64)
    assertEquals(DType.promoteTypes(DType.Int64, DType.Float32), DType.Float32)
  }

  test("DType.promoteTypes should promote integer types correctly") {
    assertEquals(DType.promoteTypes(DType.Int32, DType.Int64), DType.Int64)
    assertEquals(DType.promoteTypes(DType.Int16, DType.Int32), DType.Int32)
    assertEquals(DType.promoteTypes(DType.UInt32, DType.Int32), DType.Int64)
    assertEquals(DType.promoteTypes(DType.Int8, DType.UInt16), DType.Int32)
  }

  test("DType.canPromoteSafely should handle safe promotions correctly") {
    assert(DType.canPromoteSafely(DType.Int32, DType.Int32))
    assert(DType.canPromoteSafely(DType.Bool, DType.Int32))
    assert(DType.canPromoteSafely(DType.Int8, DType.Int16))
    assert(DType.canPromoteSafely(DType.Int32, DType.Int64))
    assert(DType.canPromoteSafely(DType.Float32, DType.Float64))
    assert(DType.canPromoteSafely(DType.Float32, DType.Complex64))
    assert(DType.canPromoteSafely(DType.UInt8, DType.Int16))
  }

  test("DType.canPromoteSafely should handle unsafe promotions correctly") {
    assert(!DType.canPromoteSafely(DType.Int64, DType.Int32))
    assert(!DType.canPromoteSafely(DType.Float64, DType.Float32))
    assert(!DType.canPromoteSafely(DType.Complex128, DType.Float64))
    assert(!DType.canPromoteSafely(DType.Int32, DType.UInt32))
  }

  test("DType.fromString should parse dtype names correctly") {
    assertEquals(DType.fromString("float32"), Some(DType.Float32))
    assertEquals(DType.fromString("f32"), Some(DType.Float32))
    assertEquals(DType.fromString("float64"), Some(DType.Float64))
    assertEquals(DType.fromString("double"), Some(DType.Float64))
    assertEquals(DType.fromString("int32"), Some(DType.Int32))
    assertEquals(DType.fromString("int"), Some(DType.Int32))
    assertEquals(DType.fromString("long"), Some(DType.Int64))
    assertEquals(DType.fromString("bool"), Some(DType.Bool))
    assertEquals(DType.fromString("boolean"), Some(DType.Bool))
    assertEquals(DType.fromString("complex64"), Some(DType.Complex64))
    assertEquals(DType.fromString("c128"), Some(DType.Complex128))
  }

  test("DType.fromString should handle case insensitive parsing") {
    assertEquals(DType.fromString("FLOAT32"), Some(DType.Float32))
    assertEquals(DType.fromString("Int64"), Some(DType.Int64))
    assertEquals(DType.fromString("BOOL"), Some(DType.Bool))
  }

  test("DType.fromString should return None for unknown strings") {
    assertEquals(DType.fromString("unknown"), None)
    assertEquals(DType.fromString("float128"), None)
    assertEquals(DType.fromString(""), None)
  }

  test("DType given conversions should work correctly") {
    import DType.given

    assertEquals((0.0f: DType), DType.Float32)
    assertEquals((0.0: DType), DType.Float64)
    assertEquals((42: DType), DType.Int32)
    assertEquals((42L: DType), DType.Int64)
    assertEquals((42.toShort: DType), DType.Int16)
    assertEquals((42.toByte: DType), DType.Int8)
    assertEquals((true: DType), DType.Bool)
  }
