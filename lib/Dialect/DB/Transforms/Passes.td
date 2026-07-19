#ifndef LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DBToCiphertextSemantic : Pass<"db-to-ciphertext-semantic"> {
  let summary = "Lower DB dialect ops to FHE ciphertext-semantic ops";
  let description = [{
    Lowers DB dialect ops via the rank kernel of Mazzone et al., "Efficient
    Ranking, Order Statistics, and Sorting under CKKS" (USENIX Security 2025)
    to slot-wise `arith` ops, `tensor_ext.rotate`, and `tensor` ops with
    public indices only, at a constant comparison depth of 2.

    The shared kernel packs a length-N vector into an N*N-slot tensor,
    compares all pairs with ONE evaluation of an f/g composite sign
    polynomial (Cheon et al.), sums the comparison matrix into per-element
    ranks, and applies a tie-correction offset so the ranks are always an
    exact permutation of 1..N, duplicates included.

    - `db.max` / `db.min`  -> indicator around rank N (resp. 1) selects the
                              extreme via an encrypted one-hot mask and an
                              inner product with the values.
    - `db.sort`            -> ranks replicated over rows, shifted by per-row
                              targets, ONE zero-indicator extracts all N order
                              statistics simultaneously (descending attr flips
                              the target constants).

    Client contract: inputs normalized into a public bound before encryption
    (see kSortInputAbsBound); power-of-two lengths (clients pad).

    `db.search_similar` (dot-product scores -> rank -> masked row delivery)
    lowers on the same kernel in a later milestone.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_
