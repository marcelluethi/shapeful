package shapeful.tensor

import scala.util.NotGiven

object TupleHelpers:

  trait Subset[T <: Tuple, SubsetT <: Tuple]:
    type Out <: Tuple

  object Subset:

    given empty[T <: Tuple]: Subset[T, EmptyTuple] with
      type Out = EmptyTuple

    given chain[T <: Tuple, K1, K2, Rest <: Tuple, Inter <: Tuple, O <: Tuple](using
      s1: Subset[T, K1 *: EmptyTuple] { type Out = Inter },
      s2: Subset[Inter, K2 *: Rest] { type Out = O }
    ): Subset[T, K1 *: K2 *: Rest] with
      type Out = s2.Out

    given singleFound[K, Tail <: Tuple]: Subset[K *: Tail, K *: EmptyTuple] with
      type Out = Tail

    given singleSearch[H, Tail <: Tuple, K, TailOut <: Tuple](using
      next: Subset[Tail, K *: EmptyTuple] { type Out = TailOut }
    ): Subset[H *: Tail, K *: EmptyTuple] with
      type Out = H *: TailOut

  type Remover[T <: Tuple, ToRemoveElement] = RemoverAll[T, ToRemoveElement *: EmptyTuple]

  object Remover:
    type Aux[T <: Tuple, ToRemoveElement, O <: Tuple] = RemoverAll.Aux[T, ToRemoveElement *: EmptyTuple, O]

  trait RemoverAll[T <: Tuple, ToRemove <: Tuple]:
    type Out <: Tuple

  object RemoverAll extends LowPriorityRemoverAll:
    
    // 0. The Aux type alias forces the compiler to resolve 'O' explicitly
    type Aux[T <: Tuple, ToRemove <: Tuple, O <: Tuple] = 
      RemoverAll[T, ToRemove] { type Out = O }

    // 1. Base Case: Empty keys -> Return input as is
    given emptyKeys[T <: Tuple]: Aux[T, EmptyTuple, T] = 
      new RemoverAll[T, EmptyTuple] { type Out = T }

    // 2. Chain Case: Process K1, then K2...
    // We use Aux to capture 'Inter' and 'O' explicitly
    given chain[T <: Tuple, K1, K2, Rest <: Tuple, Inter <: Tuple, O <: Tuple](using
      r1: Aux[T, K1 *: EmptyTuple, Inter],
      r2: Aux[Inter, K2 *: Rest, O]
    ): Aux[T, K1 *: K2 *: Rest, O] = 
      new RemoverAll[T, K1 *: K2 *: Rest] { type Out = O }

    // 3. Found Case: H is a subtype of K
    // We explicitly return 'Tail' as the output
    given singleFound[H, Tail <: Tuple, K](using H <:< K): Aux[H *: Tail, K *: EmptyTuple, Tail] = 
      new RemoverAll[H *: Tail, K *: EmptyTuple] { type Out = Tail }

  trait LowPriorityRemoverAll:
    // 4. Search Case: Recurse
    // We capture 'TailOut' as a type parameter to ensure it is fully resolved
    given singleSearch[H, Tail <: Tuple, K, TailOut <: Tuple](using
      next: RemoverAll.Aux[Tail, K *: EmptyTuple, TailOut]
    ): RemoverAll.Aux[H *: Tail, K *: EmptyTuple, H *: TailOut] = 
      new RemoverAll[H *: Tail, K *: EmptyTuple] { type Out = H *: TailOut }
    
  trait Replacer[T <: Tuple, Target, Replacement]:
    type Out <: Tuple

  object Replacer extends ReplacerLowPriority:

    given found[Target, Tail <: Tuple, Replacement]: Replacer[Target *: Tail, Target, Replacement] with
      type Out = Replacement *: Tail

  trait ReplacerLowPriority:
    given recurse[Head, Tail <: Tuple, Target, Replacement, TailOut <: Tuple](using
      next: Replacer[Tail, Target, Replacement] { type Out = TailOut }
    ): Replacer[Head *: Tail, Target, Replacement] with
      type Out = Head *: TailOut