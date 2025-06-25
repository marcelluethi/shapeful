package shapeful.tensor

import munit.FunSuite
import shapeful.tensor.TupleHelpers.*

class TupleHelpersTests extends FunSuite:

  test("ToIntTuple converts empty tuple") {
    // Test that ToIntTuple works with EmptyTuple
    val empty: ToIntTuple[EmptyTuple] = EmptyTuple
    assertEquals(empty, EmptyTuple)
  }

  test("ToIntTuple converts single element tuple") {
    // Test ToIntTuple with single element
    val single: ToIntTuple[String *: EmptyTuple] = Tuple1(42)
    assertEquals(single, Tuple1(42))
  }

  test("ToIntTuple converts multiple element tuple") {
    // Test ToIntTuple with multiple elements
    val triple: ToIntTuple[String *: Boolean *: Double *: EmptyTuple] =
      (1, 2, 3)
    assertEquals(triple, (1, 2, 3))
  }

  test("Remove removes element from tuple") {
    // Test Remove type - we can't directly test types, but we can test compilation
    val original: String *: Int *: Boolean *: EmptyTuple = ("hello", 42, true)

    // These should compile if Remove works correctly
    val removedString: Remove[String, String *: Int *: Boolean *: EmptyTuple] =
      (42, true)
    val removedInt: Remove[Int, String *: Int *: Boolean *: EmptyTuple] =
      ("hello", true)
    val removedBoolean: Remove[Boolean, String *: Int *: Boolean *: EmptyTuple] =
      ("hello", 42)

    assertEquals(removedString, (42, true))
    assertEquals(removedInt, ("hello", true))
    assertEquals(removedBoolean, ("hello", 42))
  }

  test("Remove removes first occurrence only") {
    // Test that Remove only removes the first occurrence
    val withDuplicates: String *: Int *: String *: EmptyTuple =
      ("first", 42, "second")
    val removed: Remove[String, String *: Int *: String *: EmptyTuple] =
      (42, "second")
    assertEquals(removed, (42, "second"))
  }

  test("RemoveAll removes first occurrence of each specified element") {
    // Test RemoveAll type - removing String and Boolean from (String, Int, Boolean, String)
    // RemoveAll removes first occurrence of String, then first occurrence of Boolean
    // Original: (String, Int, Boolean, String) -> remove String -> (Int, Boolean, String) -> remove Boolean -> (Int, String)
    val original: String *: Int *: Boolean *: String *: EmptyTuple =
      ("hello", 42, true, "world")
    val removed: RemoveAll[
      String *: Boolean *: EmptyTuple,
      String *: Int *: Boolean *: String *: EmptyTuple
    ] = (42, "world")
    assertEquals(removed, (42, "world"))
  }

  test("createTupleFromSeq creates correct tuples") {
    // Test createTupleFromSeq for various sizes
    val empty = createTupleFromSeq[EmptyTuple](Seq.empty)
    assertEquals(empty, EmptyTuple)

    val single = createTupleFromSeq[Int *: EmptyTuple](Seq(42))
    assertEquals(single, Tuple1(42))

    val pair = createTupleFromSeq[Int *: Int *: EmptyTuple](Seq(1, 2))
    assertEquals(pair, (1, 2))

    val triple =
      createTupleFromSeq[Int *: Int *: Int *: EmptyTuple](Seq(1, 2, 3))
    assertEquals(triple, (1, 2, 3))

    val quad = createTupleFromSeq[Int *: Int *: Int *: Int *: EmptyTuple](
      Seq(1, 2, 3, 4)
    )
    assertEquals(quad, (1, 2, 3, 4))

    val five =
      createTupleFromSeq[Int *: Int *: Int *: Int *: Int *: EmptyTuple](
        Seq(1, 2, 3, 4, 5)
      )
    assertEquals(five, (1, 2, 3, 4, 5))

    val six =
      createTupleFromSeq[Int *: Int *: Int *: Int *: Int *: Int *: EmptyTuple](
        Seq(1, 2, 3, 4, 5, 6)
      )
    assertEquals(six, (1, 2, 3, 4, 5, 6))
  }

  test("indexOf finds correct indices") {
    // Test indexOf inline function
    assertEquals(indexOf[String, String *: Int *: Boolean *: EmptyTuple], 0)
    assertEquals(indexOf[Int, String *: Int *: Boolean *: EmptyTuple], 1)
    assertEquals(indexOf[Boolean, String *: Int *: Boolean *: EmptyTuple], 2)
  }

  test("indicesOf finds indices of multiple elements") {
    // Test indicesOf inline function
    val indices = indicesOf[
      String *: Boolean *: EmptyTuple,
      String *: Int *: Boolean *: EmptyTuple
    ]
    assertEquals(indices, (0, 2))

    val singleIndex =
      indicesOf[Int *: EmptyTuple, String *: Int *: Boolean *: EmptyTuple]
    assertEquals(singleIndex, Tuple1(1))

    val emptyIndices =
      indicesOf[EmptyTuple, String *: Int *: Boolean *: EmptyTuple]
    assertEquals(emptyIndices, EmptyTuple)
  }

  test("indexOf with non-existent element should not compile") {
    // This test ensures that indexOf gives a compile-time error for non-existent elements
    // We can't test this directly in a unit test, but we can document the expected behavior

    // The following should cause a compile error:
    // indexOf[Double, String *: Int *: Boolean *: EmptyTuple]

    // For testing purposes, we'll just verify that valid cases work
    assertEquals(indexOf[String, String *: EmptyTuple], 0)
  }

  test("createTupleFromSeq with too many elements should throw") {
    // Test that createTupleFromSeq handles oversized sequences
    intercept[IllegalArgumentException] {
      createTupleFromSeq[Int *: EmptyTuple](
        Seq(1, 2, 3, 4, 5, 6, 7, 8)
      ) // Too many elements
    }
  }
