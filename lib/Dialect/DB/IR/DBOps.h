#ifndef LIB_DIALECT_DB_IR_DBOPS_H_
#define LIB_DIALECT_DB_IR_DBOPS_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/DB/IR/DBDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/DB/IR/DBOps.h.inc"

#endif  // LIB_DIALECT_DB_IR_DBOPS_H_