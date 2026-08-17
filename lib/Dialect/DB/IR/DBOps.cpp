#include "lib/Dialect/DB/IR/DBOps.h"

#include <cmath>

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#define GET_OP_CLASSES
#include "lib/Dialect/DB/IR/DBOps.cpp.inc"

namespace mlir {
namespace heir {
namespace db {

//===----------------------------------------------------------------------===//
// Shared verification helpers (reusable by SearchTopKOp and similar ops)
//===----------------------------------------------------------------------===//

// db.max/db.min take the n*n rank-kernel hall (values in row 0), not a bare
// length-n vector, so the lowering never has to grow a secret tensor.
static LogicalResult verifyIsSquareHall(mlir::Operation *op,
                                        RankedTensorType inputType) {
  if (inputType.getRank() != 1)
    return op->emitOpError("input must be a 1-D tensor, but got rank ")
           << inputType.getRank();
  int64_t len = inputType.getDimSize(0);
  int64_t n = static_cast<int64_t>(std::llround(std::sqrt(double(len))));
  if (n * n != len)
    return op->emitOpError("input length must be a perfect square n*n (the "
                           "rank-kernel hall), but got ")
           << len;
  if (n <= 0 || (n & (n - 1)) != 0)
    return op->emitOpError("hall side n must be a power of two, but got ") << n;
  return success();
}

static LogicalResult verifyQueryIs1D(mlir::Operation *op,
                                     RankedTensorType queryType) {
  if (queryType.getRank() != 1)
    return op->emitOpError("query must be a 1-D tensor, but got rank ")
           << queryType.getRank();
  return success();
}

static LogicalResult verifyDatabaseIs2D(mlir::Operation *op,
                                        RankedTensorType dbType) {
  if (dbType.getRank() != 2)
    return op->emitOpError("database must be a 2-D tensor, but got rank ")
           << dbType.getRank();
  return success();
}

static LogicalResult verifyQueryDbLengthsMatch(mlir::Operation *op,
                                               RankedTensorType queryType,
                                               RankedTensorType dbType) {
  int64_t queryLen = queryType.getDimSize(0);
  int64_t dbFeatureDim = dbType.getDimSize(1);
  if (queryLen != ShapedType::kDynamic &&
      dbFeatureDim != ShapedType::kDynamic &&
      queryLen != dbFeatureDim)
    return op->emitOpError("query length (")
           << queryLen << ") must match database feature dimension ("
           << dbFeatureDim << ")";
  return success();
}

static LogicalResult verifyQueryDbElementTypesMatch(mlir::Operation *op,
                                                    RankedTensorType queryType,
                                                    RankedTensorType dbType) {
  if (queryType.getElementType() != dbType.getElementType())
    return op->emitOpError(
               "query and database must have the same element type, but got ")
           << queryType.getElementType() << " vs " << dbType.getElementType();
  return success();
}

static LogicalResult verifySearchSimilarResult(mlir::Operation *op,
                                               RankedTensorType queryType,
                                               RankedTensorType dbType,
                                               RankedTensorType resultType,
                                               int64_t k) {
  if (resultType.getRank() != 1)
    return op->emitOpError("result must be a 1-D tensor, but got rank ")
           << resultType.getRank();
  if (resultType.getElementType() != queryType.getElementType())
    return op->emitOpError(
               "result element type must match query element type, but got ")
           << resultType.getElementType() << " vs "
           << queryType.getElementType();
  int64_t numRows = dbType.getDimSize(0);
  if (k < 1 || (numRows != ShapedType::kDynamic && k > numRows))
    return op->emitOpError("k (")
           << k << ") must be in [1, numRows (" << numRows << ")]";
  // Result concatenates the top-k rows: length k * dim.
  int64_t queryLen = queryType.getDimSize(0);
  int64_t resultLen = resultType.getDimSize(0);
  if (queryLen != ShapedType::kDynamic && resultLen != ShapedType::kDynamic &&
      resultLen != k * queryLen)
    return op->emitOpError("result length (")
           << resultLen << ") must be k * query length (" << k << " * "
           << queryLen << " = " << k * queryLen << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// SearchSimilarOp
//===----------------------------------------------------------------------===//

LogicalResult SearchSimilarOp::verify() {
  auto queryType = llvm::cast<RankedTensorType>(getQuery().getType());
  auto dbType = llvm::cast<RankedTensorType>(getDatabase().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());
  mlir::Operation *op = this->getOperation();

  if (failed(verifyQueryIs1D(op, queryType))) return failure();
  if (failed(verifyDatabaseIs2D(op, dbType))) return failure();
  if (failed(verifyQueryDbLengthsMatch(op, queryType, dbType))) return failure();
  if (failed(verifyQueryDbElementTypesMatch(op, queryType, dbType)))
    return failure();
  if (failed(verifySearchSimilarResult(op, queryType, dbType, resultType,
                                       getK())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  if (getResult().getType() != inputType.getElementType()) {
    return emitOpError("result type must match the element type of the input "
                       "tensor");
  }
  return verifyIsSquareHall(getOperation(), inputType);
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

LogicalResult MinOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  if (getResult().getType() != inputType.getElementType()) {
    return emitOpError("result type must match the element type of the input "
                       "tensor");
  }
  return verifyIsSquareHall(getOperation(), inputType);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (inputType.getRank() != 1) {
    return emitOpError("input must be a 1-D tensor, but got rank ")
           << inputType.getRank();
  }

  if (inputType != resultType) {
    return emitOpError(
               "result type must match the input type, but got ")
           << resultType << " vs " << inputType;
  }

  return success();
}

}  // namespace db
}  // namespace heir
}  // namespace mlir