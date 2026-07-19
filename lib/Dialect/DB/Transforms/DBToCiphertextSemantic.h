#ifndef LIB_DIALECT_DB_TRANSFORMS_DBTOCIPHERTEXTSEMANTIC_H_
#define LIB_DIALECT_DB_TRANSFORMS_DBTOCIPHERTEXTSEMANTIC_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace db {

#define GEN_PASS_DECL_DBTOCIPHERTEXTSEMANTIC
#include "lib/Dialect/DB/Transforms/Passes.h.inc"

}  // namespace db
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_DB_TRANSFORMS_DBTOCIPHERTEXTSEMANTIC_H_
