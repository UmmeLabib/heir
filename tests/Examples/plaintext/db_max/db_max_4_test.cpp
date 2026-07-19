// db.max end-to-end: raw values in, client normalizes into [-4,4].
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

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, 4, 1};
  _mlir_ciface_db_max_4__encrypt__arg0(&encArg0, &input0);

  // Peek at the packed input buffer (plaintext backend = simulation).
  std::printf("enc input  : %lld ct x %lld slots, first 8: [",
              (long long)encArg0.sizes[0], (long long)encArg0.sizes[1]);
  for (int i = 0; i < 8; ++i)
    std::printf("%s%.3f", i ? ", " : "", encArg0.basePtr[encArg0.offset + i]);
  std::printf(", ...]\n");

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_max_4(&memref, &encArg0);

  // Peek at the result buffer before decryption.
  std::printf("enc result : %lld ct x %lld slots, first 8: [",
              (long long)memref.sizes[0], (long long)memref.sizes[1]);
  for (int i = 0; i < 8; ++i)
    std::printf("%s%.3f", i ? ", " : "", memref.basePtr[memref.offset + i]);
  std::printf(", ...]\n");

  float result = _mlir_ciface_db_max_4__decrypt__result0(&memref);

  float rawResult = result * (halfSpread / 4.0f) + center;

  std::printf("raw input      : [%.2f, %.2f, %.2f, %.2f]\n", raw[0], raw[1],
              raw[2], raw[3]);
  std::printf("normalized     : [%.4f, %.4f, %.4f, %.4f]\n", input[0],
              input[1], input[2], input[3]);
  std::printf("db.max (norm)  : %.5f   (expected %.5f)\n", result, expected);
  std::printf("db.max (raw)   : %.5f   (true max = %.5f)\n", rawResult,
              rawExpected);

  EXPECT_NEAR(result, expected, 1e-3f);

  free(encArg0.basePtr);
  free(memref.basePtr);
}
