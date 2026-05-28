#ifndef LIB_DIALECT_DB_IR_DBDIALECT_TD_
#define LIB_DIALECT_DB_IR_DBDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def DB_Dialect : Dialect {
  let name = "db";

  let description = [{
    The DB dialect defines high-level database query operations for use in
    privacy-preserving computation pipelines.

    These operations represent scheme-agnostic, high-level database primitives
    such as similarity search, max/min reduction, and sorting. They are intended
    to be lowered to secret dialect operations and standard MLIR ops (arith,
    tensor, scf, affine) via dedicated lowering passes.

    Typical use cases include:
    - Private information retrieval (PIR)
    - Encrypted nearest-neighbor search (e.g., for RAG over FHE)
    - Oblivious sorting and ranking on encrypted data
  }];

  let cppNamespace = "::mlir::heir::db";
}

#endif  // LIB_DIALECT_DB_IR_DBDIALECT_TD_
