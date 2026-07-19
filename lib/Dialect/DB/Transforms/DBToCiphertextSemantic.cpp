#include "lib/Dialect/DB/Transforms/DBToCiphertextSemantic.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/DB/IR/DBOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"            // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

// Paper 2 rank-kernel lowering (Mazzone et al., USENIX Sec '25); worked
// walkthrough in docs/db_max_pipeline.md.

namespace mlir {
namespace heir {
namespace db {

#define GEN_PASS_DEF_DBTOCIPHERTEXTSEMANTIC
#include "lib/Dialect/DB/Transforms/Passes.h.inc"

namespace {

bool isPowerOfTwo(int64_t len) { return len > 0 && (len & (len - 1)) == 0; }

// f/g composition counts; raise kNumG to resolve smaller gaps.
constexpr int64_t kNumG = 3;
constexpr int64_t kNumF = 2;

// Client contract: data normalized into [-bound, bound] before encryption.
constexpr double kSortInputAbsBound = 4.0;

APFloat apFloatIn(FloatType fty, double value) {
  APFloat v(value);
  bool losesInfo = false;
  v.convert(fty.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  return v;
}

Value constSplat(OpBuilder &b, Location loc, RankedTensorType ty,
                 double value) {
  auto fty = llvm::cast<FloatType>(ty.getElementType());
  auto attr = DenseElementsAttr::get(
      ty, llvm::ArrayRef<APFloat>{apFloatIn(fty, value)});
  return arith::ConstantOp::create(b, loc, ty, attr);
}

Value constDense(OpBuilder &b, Location loc, RankedTensorType ty,
                 llvm::ArrayRef<double> values) {
  auto fty = llvm::cast<FloatType>(ty.getElementType());
  llvm::SmallVector<APFloat> vals;
  vals.reserve(values.size());
  for (double v : values) vals.push_back(apFloatIn(fty, v));
  return arith::ConstantOp::create(b, loc, ty,
                                   DenseElementsAttr::get(ty, vals));
}

// tensor_ext.rotate is a left rotation: result[i] = input[i + k mod size].
Value rotateLeft(OpBuilder &b, Location loc, Value vec, int64_t k) {
  auto ty = llvm::cast<RankedTensorType>(vec.getType());
  int64_t size = ty.getDimSize(0);
  k = ((k % size) + size) % size;
  if (k == 0) return vec;
  Value shift = arith::ConstantIndexOp::create(b, loc, k);
  return tensor_ext::RotateOp::create(b, loc, ty, vec, shift);
}

Value rotateRight(OpBuilder &b, Location loc, Value vec, int64_t k) {
  auto ty = llvm::cast<RankedTensorType>(vec.getType());
  return rotateLeft(b, loc, vec, ty.getDimSize(0) - k);
}

// After log2(n) fold rounds every slot holds the sum of all slots.
Value rotateReduceAddVector(OpBuilder &b, Location loc, Value vec) {
  auto ty = llvm::cast<RankedTensorType>(vec.getType());
  int64_t n = ty.getDimSize(0);
  Value acc = vec;
  for (int64_t shift = n / 2; shift > 0; shift /= 2)
    acc = arith::AddFOp::create(b, loc, acc, rotateLeft(b, loc, acc, shift));
  return acc;
}

// Odd degree-7 polynomial via Horner in x^2 (5 multiplies).
Value evalOddDeg7(OpBuilder &b, Location loc, Value x, double c1, double c3,
                  double c5, double c7) {
  auto ty = llvm::cast<RankedTensorType>(x.getType());
  Value x2 = arith::MulFOp::create(b, loc, x, x);
  Value acc = constSplat(b, loc, ty, c7);
  acc = arith::AddFOp::create(b, loc, arith::MulFOp::create(b, loc, acc, x2),
                              constSplat(b, loc, ty, c5));
  acc = arith::AddFOp::create(b, loc, arith::MulFOp::create(b, loc, acc, x2),
                              constSplat(b, loc, ty, c3));
  acc = arith::AddFOp::create(b, loc, arith::MulFOp::create(b, loc, acc, x2),
                              constSplat(b, loc, ty, c1));
  return arith::MulFOp::create(b, loc, acc, x);
}

// sign(x) ~= f^df(g^dg(x)) on [-1,1]; both odd, so sign(0) = 0 exactly.
Value sharpSign(OpBuilder &b, Location loc, Value x) {
  Value acc = x;
  for (int64_t i = 0; i < kNumG; ++i)
    acc = evalOddDeg7(b, loc, acc, 4589.0 / 1024.0, -16577.0 / 1024.0,
                      25614.0 / 1024.0, -12860.0 / 1024.0);
  for (int64_t i = 0; i < kNumF; ++i)
    acc = evalOddDeg7(b, loc, acc, 35.0 / 16.0, -35.0 / 16.0, 21.0 / 16.0,
                      -5.0 / 16.0);
  return acc;
}

// Cmp = (sign(diff*scale)+1)/2 in {0, 0.5, 1}; scale brings diff into [-1,1].
Value cmpFromDiff(OpBuilder &b, Location loc, Value diff, double scale) {
  auto ty = llvm::cast<RankedTensorType>(diff.getType());
  Value scaled =
      arith::MulFOp::create(b, loc, diff, constSplat(b, loc, ty, scale));
  Value sign = sharpSign(b, loc, scaled);
  Value plusOne =
      arith::AddFOp::create(b, loc, sign, constSplat(b, loc, ty, 1.0));
  return arith::MulFOp::create(b, loc, plusOne, constSplat(b, loc, ty, 0.5));
}

// Ind_[k-0.5,k+0.5](x): fences keep comparator inputs >= 0.5 from zero.
Value indicatorAround(OpBuilder &b, Location loc, Value x, double k,
                      double scale) {
  auto ty = llvm::cast<RankedTensorType>(x.getType());
  Value lo = arith::SubFOp::create(b, loc, x, constSplat(b, loc, ty, k - 0.5));
  Value hi = arith::SubFOp::create(b, loc, x, constSplat(b, loc, ty, k + 0.5));
  Value cmpLo = cmpFromDiff(b, loc, lo, scale);
  Value cmpHi = cmpFromDiff(b, loc, hi, scale);
  Value oneMinusHi =
      arith::SubFOp::create(b, loc, constSplat(b, loc, ty, 1.0), cmpHi);
  return arith::MulFOp::create(b, loc, cmpLo, oneMinusHi);
}

// N x N matrix lives row-major in one n*n-slot tensor; all masks plaintext.

Value maskRow0(OpBuilder &b, Location loc, Value x, int64_t n) {
  auto ty = llvm::cast<RankedTensorType>(x.getType());
  llvm::SmallVector<double> m(n * n, 0.0);
  for (int64_t j = 0; j < n; ++j) m[j] = 1.0;
  return arith::MulFOp::create(b, loc, x, constDense(b, loc, ty, m));
}

Value maskCol0(OpBuilder &b, Location loc, Value x, int64_t n) {
  auto ty = llvm::cast<RankedTensorType>(x.getType());
  llvm::SmallVector<double> m(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i) m[i * n] = 1.0;
  return arith::MulFOp::create(b, loc, x, constDense(b, loc, ty, m));
}

// SumR (Alg 9): fold all rows into row 0.
Value sumRows(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 0; i < llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(b, loc, acc,
                                rotateLeft(b, loc, acc, n * (1 << i)));
  return maskRow0(b, loc, acc, n);
}

// SumC (Alg 10): fold all columns into column 0.
Value sumCols(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 0; i < llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(b, loc, acc, rotateLeft(b, loc, acc, 1 << i));
  return maskCol0(b, loc, acc, n);
}

// ReplR (Alg 11): copy row 0 into every row.
Value replRow0(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 0; i < llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(b, loc, acc,
                                rotateRight(b, loc, acc, n * (1 << i)));
  return acc;
}

// ReplC (Alg 12): copy column 0 into every column.
Value replCol0(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 0; i < llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(b, loc, acc, rotateRight(b, loc, acc, 1 << i));
  return acc;
}

// TransR (Alg 1): move row 0 into column 0 (hops n(n-1)/2^i, then mask).
Value transRow0ToCol0(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 1; i <= llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(
        b, loc, acc, rotateRight(b, loc, acc, n * (n - 1) / (1 << i)));
  return maskCol0(b, loc, acc, n);
}

// TransC (Alg 2): move column 0 into row 0 (same hops, rotating left).
Value transCol0ToRow0(OpBuilder &b, Location loc, Value x, int64_t n) {
  Value acc = x;
  for (int64_t i = 1; i <= llvm::Log2_64(n); ++i)
    acc = arith::AddFOp::create(
        b, loc, acc, rotateLeft(b, loc, acc, n * (n - 1) / (1 << i)));
  return maskRow0(b, loc, acc, n);
}

// Scalar loop: layout-propagation rejects bulk slice/reshape ops here.
Value embedAsRow0(OpBuilder &b, Location loc, Value vec, int64_t n) {
  auto vecTy = llvm::cast<RankedTensorType>(vec.getType());
  auto matTy = RankedTensorType::get({n * n}, vecTy.getElementType());
  Value acc = constSplat(b, loc, matTy, 0.0);
  for (int64_t j = 0; j < n; ++j) {
    Value idx = arith::ConstantIndexOp::create(b, loc, j);
    Value elem = tensor::ExtractOp::create(b, loc, vec, ValueRange{idx});
    acc = tensor::InsertOp::create(b, loc, elem, acc, ValueRange{idx});
  }
  return acc;
}

// Mirror of embedAsRow0: row 0 of the hall -> a length-n vector.
Value extractRow0AsVec(OpBuilder &b, Location loc, Value mat, int64_t n) {
  auto matTy = llvm::cast<RankedTensorType>(mat.getType());
  auto vecTy = RankedTensorType::get({n}, matTy.getElementType());
  Value acc = constSplat(b, loc, vecTy, 0.0);
  for (int64_t j = 0; j < n; ++j) {
    Value idx = arith::ConstantIndexOp::create(b, loc, j);
    Value elem = tensor::ExtractOp::create(b, loc, mat, ValueRange{idx});
    acc = tensor::InsertOp::create(b, loc, elem, acc, ValueRange{idx});
  }
  return acc;
}

struct RankResult {
  Value rankMat;         // tie-corrected ranks 1..n in row 0, zeros elsewhere
  Value valuesReplRows;  // VR: input vector in every row
};

// Algs 3+6: one comparison ranks all pairs; K = SumR(C)+U-0.5T is an exact
// permutation of 1..n even with duplicates.
RankResult rankWithTieCorrection(OpBuilder &b, Location loc, Value vec,
                                 int64_t n, double cmpScale) {
  Value v = embedAsRow0(b, loc, vec, n);
  auto matTy = llvm::cast<RankedTensorType>(v.getType());

  Value vR = replRow0(b, loc, v, n);
  Value vC = replCol0(b, loc, transRow0ToCol0(b, loc, v, n), n);

  Value diff = arith::SubFOp::create(b, loc, vR, vC);
  Value c = cmpFromDiff(b, loc, diff, cmpScale);

  Value r = sumRows(b, loc, c, n);

  // Equality matrix recycled from C: maps 0.5 -> 1, {0,1} -> 0.
  Value one = constSplat(b, loc, matTy, 1.0);
  Value four = constSplat(b, loc, matTy, 4.0);
  Value oneMinusC = arith::SubFOp::create(b, loc, one, c);
  Value e = arith::MulFOp::create(
      b, loc, four, arith::MulFOp::create(b, loc, c, oneMinusC));

  // Upper-triangle censor: count only tie-partners at or before position j.
  llvm::SmallVector<double> upper(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i; j < n; ++j) upper[i * n + j] = 1.0;
  Value eUpper =
      arith::MulFOp::create(b, loc, e, constDense(b, loc, matTy, upper));

  Value u = sumRows(b, loc, eUpper, n);
  Value t = sumRows(b, loc, e, n);
  Value halfT =
      arith::MulFOp::create(b, loc, t, constSplat(b, loc, matTy, 0.5));

  Value k = arith::AddFOp::create(b, loc, r, u);
  k = arith::SubFOp::create(b, loc, k, halfT);
  return {k, vR};
}

// Alg 4: order statistic = indicator on the ranks (k = n max, 1 min).
LogicalResult lowerOrderStatistic(Operation *op, Value input, Type resultType,
                                  bool isMax, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto ty = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!ty || ty.getRank() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected a 1-D ranked tensor input");
  int64_t n = ty.getDimSize(0);
  if (!isPowerOfTwo(n))
    return rewriter.notifyMatchFailure(
        op, "matrix-encoded ranking requires a power-of-two length");
  if (!llvm::isa<FloatType>(ty.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Paper-2 ranking requires a floating-point element type (CKKS)");
  if (resultType != ty.getElementType())
    return rewriter.notifyMatchFailure(
        op, "result type must match the input element type");

  RankResult rank = rankWithTieCorrection(rewriter, loc, input, n,
                                          1.0 / (2.0 * kSortInputAbsBound));

  // Junk slots hold 0 and Ind_k(0) = 0; scale 1/(n+1) keeps -(k+0.5) in [-1,1].
  double k = isMax ? static_cast<double>(n) : 1.0;
  Value mask =
      indicatorAround(rewriter, loc, rank.rankMat, k, 1.0 / (n + 1.0));

  Value masked =
      arith::MulFOp::create(rewriter, loc, mask, rank.valuesReplRows);
  Value total = rotateReduceAddVector(rewriter, loc, masked);
  Value idx0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value result =
      tensor::ExtractOp::create(rewriter, loc, total, ValueRange{idx0});
  rewriter.replaceOp(op, result);
  return success();
}

struct LowerMaxOp : public OpRewritePattern<MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter &rewriter) const override {
    return lowerOrderStatistic(op, op.getInput(), op.getResult().getType(),
                               /*isMax=*/true, rewriter);
  }
};

// Client padding note: min pads with the MAXIMUM (max pads with the minimum).
struct LowerMinOp : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter &rewriter) const override {
    return lowerOrderStatistic(op, op.getInput(), op.getResult().getType(),
                               /*isMax=*/false, rewriter);
  }
};

// Alg 5: replicate badges, subtract per-row targets, one zero-indicator
// performs all N hunts; comparison depth stays 2.
LogicalResult lowerSort(SortOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value input = op.getInput();

  auto ty = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!ty || ty.getRank() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected a 1-D ranked tensor input");
  int64_t n = ty.getDimSize(0);
  if (!isPowerOfTwo(n))
    return rewriter.notifyMatchFailure(
        op, "matrix-encoded ranking requires a power-of-two length");
  if (!llvm::isa<FloatType>(ty.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Paper-2 ranking requires a floating-point element type (CKKS)");
  if (op.getResult().getType() != ty)
    return rewriter.notifyMatchFailure(
        op, "result type must match the input type");

  RankResult rank = rankWithTieCorrection(rewriter, loc, input, n,
                                          1.0 / (2.0 * kSortInputAbsBound));
  auto matTy = llvm::cast<RankedTensorType>(rank.rankMat.getType());

  Value kR = replRow0(rewriter, loc, rank.rankMat, n);

  // Row i hunts badge i+1 (n-i when descending); plaintext constant.
  bool descending = op.getDescending();
  llvm::SmallVector<double> targets(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    double t = descending ? static_cast<double>(n - i)
                          : static_cast<double>(i + 1);
    for (int64_t j = 0; j < n; ++j) targets[i * n + j] = t;
  }
  Value d = arith::SubFOp::create(rewriter, loc, kR,
                                  constDense(rewriter, loc, matTy, targets));

  // D in [-(n-1), n-1], no junk slots; 1/(n+1) keeps fences in [-1,1].
  Value m = indicatorAround(rewriter, loc, d, 0.0, 1.0 / (n + 1.0));

  Value picked =
      arith::MulFOp::create(rewriter, loc, m, rank.valuesReplRows);
  Value s =
      transCol0ToRow0(rewriter, loc, sumCols(rewriter, loc, picked, n), n);
  rewriter.replaceOp(op, extractRow0AsVec(rewriter, loc, s, n));
  return success();
}

struct LowerSortOp : public OpRewritePattern<SortOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    return lowerSort(op, rewriter);
  }
};

struct DBToCiphertextSemantic
    : public impl::DBToCiphertextSemanticBase<DBToCiphertextSemantic> {
  using DBToCiphertextSemanticBase::DBToCiphertextSemanticBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // db.search_similar migrates onto the same kernel next.
    patterns.add<LowerMaxOp, LowerMinOp, LowerSortOp>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace
}  // namespace db
}  // namespace heir
}  // namespace mlir
