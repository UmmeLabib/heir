#ifndef LIB_DIALECT_DB_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_DB_TRANSFORMS_PASSES_H_

#include "lib/Dialect/DB/IR/DBDialect.h"
#include "lib/Dialect/DB/Transforms/DBToCiphertextSemantic.h"

namespace mlir {
namespace heir {
namespace db {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/DB/Transforms/Passes.h.inc"

}  // namespace db
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_DB_TRANSFORMS_PASSES_H_
