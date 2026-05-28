#include "lib/Dialect/DB/IR/DBDialect.h"

#include "lib/Dialect/DB/IR/DBOps.h"

// Generated definitions
#include "lib/Dialect/DB/IR/DBDialect.cpp.inc"

namespace mlir {
namespace heir {
namespace db {

//===----------------------------------------------------------------------===//
// DB dialect.
//===----------------------------------------------------------------------===//

void DBDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/DB/IR/DBOps.cpp.inc"
      >();
}

}  // namespace db
}  // namespace heir
}  // namespace mlir
