// db.min end-to-end with a tie at the minimum (exercises tie-correction).
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_min_4(StridedMemRefType<float, 2>* result,
                           StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_min_4__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                          StridedMemRefType<float>* input);

float _mlir_ciface_db_min_4__decrypt__result0(
    StridedMemRefType<float, 2>* input);
}

TEST(DbMin4Test, MinWithTieAtMinimum) {
  float raw[4] = {42.0f, -7.5f, 100.0f, -7.5f};  // min -7.5 appears twice
  float rawExpected = -7.5f;                     // true min of raw

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

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, 4, 1};
  _mlir_ciface_db_min_4__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_min_4(&memref, &encArg0);

  float result = _mlir_ciface_db_min_4__decrypt__result0(&memref);
  float rawResult = result * (halfSpread / 4.0f) + center;

  std::printf("raw input      : [42, -7.5, 100, -7.5]  (tied minimum)\n");
  std::printf("normalized     : [%.4f, %.4f, %.4f, %.4f]\n", input[0],
              input[1], input[2], input[3]);
  std::printf("db.min (norm)  : %.5f   (expected %.5f)\n", result, expected);
  std::printf("db.min (raw)   : %.5f   (true min = %.5f)\n", rawResult,
              rawExpected);

  EXPECT_NEAR(result, expected, 1e-3f);

  free(encArg0.basePtr);
  free(memref.basePtr);
}
