#ifndef LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DBToLinalg : Pass<"db-to-linalg"> {
  let summary = "Lower DB dialect ops to linalg, arith, scf, and tensor ops";
  let description = [{
    Lowers each DB dialect operation to a combination of structured linalg ops,
    arith scalar ops, scf control flow, and tensor ops:

    - `db.max` / `db.min`  → `linalg.reduce` with arith max/min combiner
    - `db.search_similar`  → `linalg.matvec` (dot products) + `scf.for` argmax
                             + `tensor.extract_slice` (row extraction)
    - `db.sort`            → nested `scf.for` bubble-sort with tensor extract/insert
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_DB_TRANSFORMS_PASSES_TD_
