// db.max end-to-end: raw values in, client normalizes into [-4,4] and packs
// them into row 0 of the 4x4 hall (slots 0..3; slots 4..15 stay zero).
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_max_4(StridedMemRefType<float, 2>* result,
                           StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_max_4__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                          StridedMemRefType<float>* input);

float _mlir_ciface_db_max_4__decrypt__result0(
    StridedMemRefType<float, 2>* input);
}

namespace {

// Pack 4 already-normalized values into the hall, run db.max, decrypt.
float runMax(const float vals[4]) {
  float input[16] = {0.0f};
  for (int i = 0; i < 4; ++i) input[i] = vals[i];

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, 16, 1};
  _mlir_ciface_db_max_4__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_max_4(&memref, &encArg0);

  float result = _mlir_ciface_db_max_4__decrypt__result0(&memref);
  free(encArg0.basePtr);
  free(memref.basePtr);
  return result;
}

}  // namespace

TEST(DbMax4Test, MaxOfNormalized) {
  float raw[4] = {-100.0f, -1000.0f, 42.0f, 7.0f};
  float rawExpected = 42.0f;  // true max of raw

  // Min-max normalize into [-4,4]; gaps below ~1% of spread may misrank.
  float mn = raw[0], mx = raw[0];
  for (float v : raw) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  float center = (mn + mx) / 2.0f;
  float halfSpread = (mx - mn) / 2.0f;

  float input[4];
  for (int i = 0; i < 4; ++i)
    input[i] = (raw[i] - center) * (4.0f / halfSpread);
  float expected = (rawExpected - center) * (4.0f / halfSpread);

  float result = runMax(input);
  float rawResult = result * (halfSpread / 4.0f) + center;

  std::printf("raw input      : [%.2f, %.2f, %.2f, %.2f]\n", raw[0], raw[1],
              raw[2], raw[3]);
  std::printf("normalized     : [%.4f, %.4f, %.4f, %.4f]\n", input[0],
              input[1], input[2], input[3]);
  std::printf("db.max (norm)  : %.5f   (expected %.5f)\n", result, expected);
  std::printf("db.max (raw)   : %.5f   (true max = %.5f)\n", rawResult,
              rawExpected);

  EXPECT_NEAR(result, expected, 1e-3f);
}

// The index tie-break (e * (L - 0.5)) must leave exactly one winning column,
// so duplicates of the maximum still sum to the maximum and not a multiple or
// a fraction of it.
TEST(DbMax4Test, TwoWayTieOnMax) {
  float input[4] = {4.0f, 4.0f, -4.0f, 1.0f};
  float result = runMax(input);
  std::printf("two-way tie   : [4, 4, -4, 1] -> %.5f\n", result);
  EXPECT_NEAR(result, 4.0f, 1e-3f);
}

TEST(DbMax4Test, ThreeWayTieOnMax) {
  float input[4] = {4.0f, 4.0f, 4.0f, -4.0f};
  float result = runMax(input);
  std::printf("three-way tie : [4, 4, 4, -4] -> %.5f\n", result);
  EXPECT_NEAR(result, 4.0f, 1e-3f);
}

TEST(DbMax4Test, AllFourEqual) {
  float input[4] = {2.5f, 2.5f, 2.5f, 2.5f};
  float result = runMax(input);
  std::printf("all equal     : [2.5 x4] -> %.5f\n", result);
  EXPECT_NEAR(result, 2.5f, 1e-3f);
}

TEST(DbMax4Test, TwoDistinctPairs) {
  float input[4] = {-3.0f, 3.0f, 3.0f, -3.0f};
  float result = runMax(input);
  std::printf("two pairs     : [-3, 3, 3, -3] -> %.5f\n", result);
  EXPECT_NEAR(result, 3.0f, 1e-3f);
}
