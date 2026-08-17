#ifndef LIB_DIALECT_DB_IR_DBOPS_TD_
#define LIB_DIALECT_DB_IR_DBOPS_TD_

include "DBDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class DB_Op<string mnemonic, list<Trait> traits = []> :
    Op<DB_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::db";
}

//===----------------------------------------------------------------------===//
// db.search_similar
//===----------------------------------------------------------------------===//

def DB_SearchSimilarOp : DB_Op<"search_similar", []> {
  let summary = "Return the k database rows most similar to a query vector.";

  let description = [{
    `db.search_similar` computes the dot-product similarity between a query
    vector and every row of a database matrix, then returns the `k` most
    similar (highest inner-product) rows -- their contents, concatenated,
    NOT their indices (the selection stays oblivious under FHE).

    The optional `k` attribute (default 1) selects the top-k rows.  The
    `query` operand is a 1-D ranked tensor of length `dim`; the `database`
    operand is a 2-D ranked tensor `numRows x dim`.  The result is a 1-D
    tensor of length `k * dim`: the best match in the first `dim` slots, the
    second best in the next `dim`, and so on.

    Example:
    ```mlir
    // top-2 of 1024 rows, each 128-D: result is the 2 rows concatenated
    %rows = db.search_similar %query, %database {k = 2 : i64}
                : tensor<128xf32>, tensor<1024x128xf32> -> tensor<256xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$database,
    DefaultValuedAttr<I64Attr, "1">:$k
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $query `,` $database attr-dict `:` type($query) `,` type($database) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// db.max
//===----------------------------------------------------------------------===//

def DB_MaxOp : DB_Op<"max", []> {
  let summary = "Compute the maximum value of a ranked tensor.";

  let description = [{
    `db.max` returns the maximum element of the input ranked tensor.  The
    result type must match the element type of the input tensor.

    The operand is the `n x n` rank-kernel hall flattened row-major, not a
    bare length-`n` vector: the client places its `n` values in slots
    `0..n-1` and zeros the remaining `n*n - n` slots before encrypting.
    Packing on the client keeps the lowering from growing a secret tensor,
    which would emit a `secret.conceal` that CKKS lowering rejects.  `n` must
    be a power of two.

    Example:
    ```mlir
    // 4 values live in slots 0..3 of a 4x4 hall; slots 4..15 are zero
    %m = db.max %packed : tensor<16xf32> -> f32
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// db.min
//===----------------------------------------------------------------------===//

def DB_MinOp : DB_Op<"min", []> {
  let summary = "Compute the minimum value of a ranked tensor.";

  let description = [{
    `db.min` returns the minimum element of the input ranked tensor.  The
    result type must match the element type of the input tensor.

    This is the dual of `db.max`, useful for oblivious minimum selection over
    encrypted data.  It takes the same client-packed `n x n` hall as `db.max`:
    values in slots `0..n-1`, the rest zero, `n` a power of two.

    Example:
    ```mlir
    // 4 values live in slots 0..3 of a 4x4 hall; slots 4..15 are zero
    %m = db.min %packed : tensor<16xf32> -> f32
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// db.sort
//===----------------------------------------------------------------------===//

def DB_SortOp : DB_Op<"sort", []> {
  let summary = "Sort the elements of a ranked tensor.";

  let description = [{
    `db.sort` returns a new tensor with the same shape as the input, containing
    the elements of the input sorted in ascending order.

    The optional `descending` boolean attribute (default: false) switches the
    sort order to descending.

    This op is intended to be lowered to an oblivious sorting-network or
    merge-based FHE sort over secret ciphertexts.

    Example:
    ```mlir
    // ascending sort
    %sorted = db.sort %scores : tensor<64xi32> -> tensor<64xi32>

    // descending sort
    %sorted_desc = db.sort %scores {descending = true} : tensor<64xi32> -> tensor<64xi32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_DB_IR_DBOPS_TD_
