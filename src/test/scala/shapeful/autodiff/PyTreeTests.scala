package shapeful.autodiff

import scala.language.experimental.namedTypeArguments

import munit.FunSuite

import shapeful.*
import shapeful.jax.Jax
import shapeful.tensor.DType
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.SeqConverters

class PyTreeTests extends FunSuite:

  override def beforeAll(): Unit =
    super.beforeAll()

  test("derives ToPyTree for Tensor0") {
    val tensor = Tensor0(1f)
    val pyTree = summon[ToPyTree[Tensor0]].toPyTree(tensor)
    assert(pyTree.as[Float] == 1f, "Expected PyTree to be a simple float value")
    val reconstructed = ToPyTree[Tensor0].fromPyTree(pyTree)
    assert(reconstructed == tensor, "Expected reconstruction to yield the original Tensor0")
  }

  test("derives ToPyTree for Tensor1") {
    val tensor = Tensor1["Label"](Seq(1f, 2f, 3f))
    val pyTree = summon[ToPyTree[Tensor1["Label"]]].toPyTree(tensor)
    assert(pyTree.as[Seq[Float]] == Seq(1f, 2f, 3f), "Expected PyTree to be a sequence of floats")
    val reconstructed = ToPyTree[Tensor1["Label"]].fromPyTree(pyTree)
    assert(reconstructed == tensor, "Expected reconstruction to yield the original Tensor1")
  }

  test("derives ToPyTree for Tensor2") {
    val tensor = Tensor2["Label1", "Label2"](Seq(Seq(1f, 2f), Seq(3f, 4f)))
    val pyTree = summon[ToPyTree[Tensor2["Label1", "Label2"]]].toPyTree(tensor)
    assert(
      pyTree.as[Seq[Seq[Float]]] == Seq(Seq(1f, 2f), Seq(3f, 4f)),
      "Expected PyTree to be a sequence of sequences of floats"
    )
    val reconstructed = ToPyTree[Tensor2["Label1", "Label2"]].fromPyTree(pyTree)
    assert(reconstructed == tensor, "Expected reconstruction to yield the original Tensor2")
  }

  test("derives ToPyTree for Tuple of different tensors") {
    val tensor1 = Tensor1["Label1"](Seq(1f, 2f))
    val tensor2 = Tensor2["Label2", "Label3"](Seq(Seq(3f, 4f), Seq(5f, 6f)))
    val tuple = (tensor1, tensor2)
    val pyTree = summon[ToPyTree[(Tensor1["Label1"], Tensor2["Label2", "Label3"])]].toPyTree(tuple)
    val reconstructed = ToPyTree[(Tensor1["Label1"], Tensor2["Label2", "Label3"])].fromPyTree(pyTree)
    assert(reconstructed == tuple, "Expected reconstruction to yield the original tuple")
  }

  test("derives ToPyTree for case class with tensors") {
    case class SomeParams(weight: Tensor2["Label1", "Label2"], bias: Tensor1["Label2"])
    val params = SomeParams(Tensor2["Label1", "Label2"](Seq(Seq(1f, 2f), Seq(3f, 4f))), Tensor1["Label2"](Seq(5f, 6f)))
    val pyTree = summon[ToPyTree[SomeParams]].toPyTree(params)
    val reconstructed = ToPyTree[SomeParams].fromPyTree(pyTree)
    assert(reconstructed == params, "Expected reconstruction to yield the original case class")
  }

  test("derives ToPyTree for nested case classes with tensors") {
    case class InnerParams(weight: Tensor2["Label1", "Label2"], bias: Tensor1["Label2"]) derives ToPyTree
    case class OuterParams(inner: InnerParams, other: Tensor1["Label2"]) derives ToPyTree

    val innerParams =
      InnerParams(Tensor2["Label1", "Label2"](Seq(Seq(1f, 2f), Seq(3f, 4f))), Tensor1["Label2"](Seq(5f, 6f)))
    val outerParams = OuterParams(innerParams, Tensor1["Label2"](Seq(7f, 8f)))
    val pyTree = summon[ToPyTree[OuterParams]].toPyTree(outerParams)
    val reconstructed = ToPyTree[OuterParams].fromPyTree(pyTree)
    assert(reconstructed == outerParams, "Expected reconstruction to yield the original nested case class")
  }

  test("derives ToPyTree for Seq with single element") {
    val singleElementSeq = Seq(Tensor0(42f))
    val pyTree = summon[ToPyTree[Seq[Tensor0]]].toPyTree(singleElementSeq)
    val reconstructed = ToPyTree[Seq[Tensor0]].fromPyTree(pyTree)
    assert(reconstructed.length == 1, "Expected reconstructed sequence to have length 1")
    assert(reconstructed.head == Tensor0(42f), "Expected reconstructed element to match original")
    assert(reconstructed == singleElementSeq, "Expected reconstruction to yield the original single-element sequence")
  }

  test("derives ToPyTree for Seq with multiple elements") {
    val multiElementSeq = Seq(Tensor0(1f), Tensor0(2f), Tensor0(3f))
    val pyTree = summon[ToPyTree[Seq[Tensor0]]].toPyTree(multiElementSeq)
    val reconstructed = ToPyTree[Seq[Tensor0]].fromPyTree(pyTree)
    assert(reconstructed.length == 3, "Expected reconstructed sequence to have length 3")
    assert(reconstructed == multiElementSeq, "Expected reconstruction to yield the original multi-element sequence")
  }

  test("derives ToPyTree for empty Seq") {
    val emptySeq = Seq.empty[Tensor0]
    val pyTree = summon[ToPyTree[Seq[Tensor0]]].toPyTree(emptySeq)
    val reconstructed = ToPyTree[Seq[Tensor0]].fromPyTree(pyTree)
    assert(reconstructed.isEmpty, "Expected reconstructed sequence to be empty")
    assert(reconstructed == emptySeq, "Expected reconstruction to yield the original empty sequence")
  }

  test("derives ToPyTree for case class with Seq field") {
    case class ParamsWithSeq(weights: Seq[Tensor0], bias: Tensor0) derives ToPyTree

    // Test with single element in Seq
    val paramsWithSingle = ParamsWithSeq(Seq(Tensor0(1f)), Tensor0(2f))
    val pyTreeSingle = summon[ToPyTree[ParamsWithSeq]].toPyTree(paramsWithSingle)
    val reconstructedSingle = ToPyTree[ParamsWithSeq].fromPyTree(pyTreeSingle)
    assert(
      reconstructedSingle == paramsWithSingle,
      "Expected reconstruction to yield the original case class with single-element Seq"
    )

    // Test with multiple elements in Seq
    val paramsWithMultiple = ParamsWithSeq(Seq(Tensor0(1f), Tensor0(2f), Tensor0(3f)), Tensor0(4f))
    val pyTreeMultiple = summon[ToPyTree[ParamsWithSeq]].toPyTree(paramsWithMultiple)
    val reconstructedMultiple = ToPyTree[ParamsWithSeq].fromPyTree(pyTreeMultiple)
    assert(
      reconstructedMultiple == paramsWithMultiple,
      "Expected reconstruction to yield the original case class with multi-element Seq"
    )
  }
