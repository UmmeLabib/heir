#include "lib/Dialect/DB/Transforms/DBToLinalg.h"

#include <utility>

#include "lib/Dialect/DB/IR/DBOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/APInt.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace db {

#define GEN_PASS_DEF_DBTOLINALG
#include "lib/Dialect/DB/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type-dispatch helpers for init values and combiners
//
// LINALG CONCEPT: linalg.reduce needs an identity element (init value) for
// the reduction.  For max, the identity is -infinity (any real value beats it).
// For min, the identity is +infinity.  The combiner is the actual reduction
// function applied element-by-element.
//===----------------------------------------------------------------------===//

Value createMaxInit(OpBuilder &b, Location loc, Type elemType) {
  if (auto intTy = llvm::dyn_cast<IntegerType>(elemType)) {
    APInt minVal = APInt::getSignedMinValue(intTy.getWidth());
    return arith::ConstantOp::create(b, loc, IntegerAttr::get(intTy, minVal));
  }
  auto floatTy = llvm::cast<FloatType>(elemType);
  APFloat negInf =
      APFloat::getInf(floatTy.getFloatSemantics(), /*negative=*/true);
  return arith::ConstantOp::create(b, loc, FloatAttr::get(floatTy, negInf));
}

Value createMinInit(OpBuilder &b, Location loc, Type elemType) {
  if (auto intTy = llvm::dyn_cast<IntegerType>(elemType)) {
    APInt maxVal = APInt::getSignedMaxValue(intTy.getWidth());
    return arith::ConstantOp::create(b, loc, IntegerAttr::get(intTy, maxVal));
  }
  auto floatTy = llvm::cast<FloatType>(elemType);
  APFloat posInf =
      APFloat::getInf(floatTy.getFloatSemantics(), /*negative=*/false);
  return arith::ConstantOp::create(b, loc, FloatAttr::get(floatTy, posInf));
}

Value buildMaxCombiner(OpBuilder &b, Location loc, Value elem, Value acc) {
  if (llvm::isa<IntegerType>(elem.getType()))
    return arith::MaxSIOp::create(b, loc, elem, acc);
  return arith::MaximumFOp::create(b, loc, elem, acc);
}

Value buildMinCombiner(OpBuilder &b, Location loc, Value elem, Value acc) {
  if (llvm::isa<IntegerType>(elem.getType()))
    return arith::MinSIOp::create(b, loc, elem, acc);
  return arith::MinimumFOp::create(b, loc, elem, acc);
}

//===----------------------------------------------------------------------===//
// LowerMaxOp
//
// LINALG CONCEPT: linalg.reduce is a structured op that:
//   1. Takes an input tensor and an init tensor (same type but lower rank)
//   2. Iterates over the reduction dimensions
//   3. Combines each element with the accumulator via the body region
//
// For db.max on tensor<Nxt>, we reduce dimension 0 to a scalar (0-D tensor),
// then extract the scalar with tensor.extract.
//===----------------------------------------------------------------------===//

struct LowerMaxOp : public OpRewritePattern<MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType =
        llvm::cast<RankedTensorType>(op.getInput().getType()).getElementType();

    // 0-D output tensor holds the accumulator (scalar wrapped in tensor)
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, SmallVector<OpFoldResult>{}, elemType);
    Value initVal = createMaxInit(rewriter, loc, elemType);
    Value initTensor = linalg::FillOp::create(
        rewriter, loc, ValueRange{initVal}, ValueRange{emptyTensor})
        .getResult(0);

    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{op.getInput()}, ValueRange{initTensor},
        ArrayRef<int64_t>{0},
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value newAcc = buildMaxCombiner(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, newAcc);
        });

    // Extract scalar from the 0-D result tensor
    Value result = tensor::ExtractOp::create(rewriter, loc,
                                             reduceOp.getResult(0),
                                             ValueRange{});
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerMinOp  (mirror of LowerMaxOp with min combiner)
//===----------------------------------------------------------------------===//

struct LowerMinOp : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType =
        llvm::cast<RankedTensorType>(op.getInput().getType()).getElementType();

    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, SmallVector<OpFoldResult>{}, elemType);
    Value initVal = createMinInit(rewriter, loc, elemType);
    Value initTensor = linalg::FillOp::create(
        rewriter, loc, ValueRange{initVal}, ValueRange{emptyTensor})
        .getResult(0);

    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{op.getInput()}, ValueRange{initTensor},
        ArrayRef<int64_t>{0},
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value newAcc = buildMinCombiner(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, newAcc);
        });

    Value result = tensor::ExtractOp::create(rewriter, loc,
                                             reduceOp.getResult(0),
                                             ValueRange{});
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerSearchSimilarOp
//
// Three-step lowering:
//
// Step 1 — linalg.matvec: compute scores[i] = dot(db[i,:], query)
//   LINALG CONCEPT: linalg.matvec is a named structured op for y += A*x.
//   "Named" = the loop structure and indexing maps are fixed (unlike
//   linalg.generic where you define them yourself).  It accumulates into the
//   output tensor, so we pre-fill it with zeros.
//
// Step 2 — scf.for argmax: find the row index with the highest score.
//   LINALG CONCEPT: linalg has no argmax (it can't return an index alongside
//   the value in the same reduction), so we fall back to scf.for loops.
//
// Step 3 — tensor.extract_slice: extract that best row from the database.
//   rank-reducing extract_slice drops the leading dim (size=1), giving a 1-D
//   result tensor instead of a 2-D slice.
//===----------------------------------------------------------------------===//

struct LowerSearchSimilarOp : public OpRewritePattern<SearchSimilarOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SearchSimilarOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dbType =
        llvm::cast<RankedTensorType>(op.getDatabase().getType());
    auto elemType = dbType.getElementType();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Step 1: matvec → scores tensor
    Value M = tensor::DimOp::create(rewriter, loc, op.getDatabase(), c0);
    Value scoresEmpty = tensor::EmptyOp::create(
        rewriter, loc, SmallVector<OpFoldResult>{M}, elemType);

    Value zero;
    if (llvm::isa<IntegerType>(elemType))
      zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(elemType));
    else
      zero = arith::ConstantOp::create(
          rewriter, loc, FloatAttr::get(elemType, 0.0));

    // Compute scores with linalg.matvec
    Value scoresInit = linalg::FillOp::create(
        rewriter, loc, ValueRange{zero}, ValueRange{scoresEmpty}).getResult(0);

    Value scores = linalg::MatvecOp::create(
        rewriter, loc, TypeRange{scoresInit.getType()},
        ValueRange{op.getDatabase(), op.getQuery()},
        ValueRange{scoresInit}).getResult(0);

    // Step 2: argmax over scores
    Value initScore =
        tensor::ExtractOp::create(rewriter, loc, scores, ValueRange{c0});

    auto forOp = scf::ForOp::create(rewriter, loc, c1, M, c1,
                                    ValueRange{initScore, c0});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value i = forOp.getInductionVar();
      Value bestScore = forOp.getRegionIterArgs()[0];
      Value bestIdx = forOp.getRegionIterArgs()[1];

      Value score =
          tensor::ExtractOp::create(rewriter, loc, scores, ValueRange{i});

      Value isBetter;
      if (llvm::isa<IntegerType>(elemType))
      //PROBLEM 1: arith.CmpIOp / arith.CmpFOp means "compare integers/floats"
      //not possible in FHE
        isBetter = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::sgt, score,
                                         bestScore);
      else
        isBetter = arith::CmpFOp::create(rewriter, loc,
                                         arith::CmpFPredicate::OGT, score,
                                         bestScore);
      // PROBLEM 2: Select (SelectOp) is comparison 
      Value newScore =
          arith::SelectOp::create(rewriter, loc, isBetter, score, bestScore);
      Value newIdx =
          arith::SelectOp::create(rewriter, loc, isBetter, i, bestIdx);
      scf::YieldOp::create(rewriter, loc, ValueRange{newScore, newIdx});
    }
    Value bestIdx = forOp.getResult(1);  // ← PROBLEM 3: this index is SECRET because it was computed from comparing encrypted scores.

    // Step 3: rank-reducing extract_slice to get the best row.
    // sizes[1] must use a static attr when the result dim is static, otherwise
    // the static_sizes in the generated op would be kDynamic but the result
    // type would have a known size — causing a verification error.
    auto resultType =
        llvm::cast<RankedTensorType>(op.getResult().getType());
    OpFoldResult colSize;
    if (resultType.getDimSize(0) == ShapedType::kDynamic) {
      Value N = tensor::DimOp::create(rewriter, loc, op.getDatabase(), c1);
      colSize = N;
    } else {
      colSize = rewriter.getIndexAttr(resultType.getDimSize(0));
    }

    SmallVector<OpFoldResult> offsets = {bestIdx, rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), colSize};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};

    Value row = tensor::ExtractSliceOp::create(
        rewriter, loc, op.getResult().getType(), op.getDatabase(), offsets,
        sizes, strides);

    rewriter.replaceOp(op, row);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerSortOp
//
// LINALG CONCEPT: linalg has no sort op, so we use nested scf.for loops.
// Bubble sort — O(N²) passes, each swapping adjacent out-of-order elements.
// Uses tensor.extract / tensor.insert for functional (immutable) updates.
//
// Functional tensors: every tensor.insert creates a NEW tensor with the
// updated element; the original tensor is unchanged.  This matches MLIR's
// SSA semantics — no mutation, only new Values.
//===----------------------------------------------------------------------===//

struct LowerSortOp : public OpRewritePattern<SortOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType =
        llvm::cast<RankedTensorType>(op.getInput().getType()).getElementType();
    bool descending = op.getDescending();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value N = tensor::DimOp::create(rewriter, loc, op.getInput(), c0);
    Value Nminus1 = arith::SubIOp::create(rewriter, loc, N, c1);

    // Outer loop: N bubble-sort passes
    auto outerLoop = scf::ForOp::create(rewriter, loc, c0, N, c1,
                                        ValueRange{op.getInput()});
    {
      OpBuilder::InsertionGuard outerGuard(rewriter);
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      Value outerT = outerLoop.getRegionIterArgs()[0];

      // Inner loop: compare-and-swap adjacent pairs
      auto innerLoop = scf::ForOp::create(rewriter, loc, c0, Nminus1, c1,
                                          ValueRange{outerT});
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());

        Value j = innerLoop.getInductionVar();
        Value t = innerLoop.getRegionIterArgs()[0];
        Value jPlus1 = arith::AddIOp::create(rewriter, loc, j, c1);

        Value a =
            tensor::ExtractOp::create(rewriter, loc, t, ValueRange{j});
        Value b =
            tensor::ExtractOp::create(rewriter, loc, t, ValueRange{jPlus1});

        // Swap condition: ascending → a > b;  descending → a < b
        // PROBLEM 4: This comparison is not possible in FHE as shouldSwap is comparison
        Value shouldSwap;
        if (llvm::isa<IntegerType>(elemType)) {
          auto pred = descending ? arith::CmpIPredicate::slt
                                 : arith::CmpIPredicate::sgt;
          shouldSwap = arith::CmpIOp::create(rewriter, loc, pred, a, b);
        } else {
          auto pred = descending ? arith::CmpFPredicate::OLT
                                 : arith::CmpFPredicate::OGT;
          shouldSwap = arith::CmpFOp::create(rewriter, loc, pred, a, b);
        }

        Value newA =
            arith::SelectOp::create(rewriter, loc, shouldSwap, b, a);
        Value newB =
            arith::SelectOp::create(rewriter, loc, shouldSwap, a, b);

        Value t2 =
            tensor::InsertOp::create(rewriter, loc, newA, t, ValueRange{j});
        Value t3 = tensor::InsertOp::create(rewriter, loc, newB, t2,
                                            ValueRange{jPlus1});
        scf::YieldOp::create(rewriter, loc, ValueRange{t3});
      }

      scf::YieldOp::create(rewriter, loc, innerLoop.getResult(0));
    }

    rewriter.replaceOp(op, outerLoop.getResult(0));
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

struct DBToLinalg : impl::DBToLinalgBase<DBToLinalg> {
  using DBToLinalgBase::DBToLinalgBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerMaxOp, LowerMinOp, LowerSearchSimilarOp, LowerSortOp>(
        context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace db
}  // namespace heir
}  // namespace mlir
