// db.max on 16 slots: client pads shorter inputs with the minimum.
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_max_16(StridedMemRefType<float, 2>* result,
                            StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_max_16__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                           StridedMemRefType<float>* input);

float _mlir_ciface_db_max_16__decrypt__result0(
    StridedMemRefType<float, 2>* input);
}

TEST(DbMax16Test, MaxOfTenRawValuesPadded) {
  constexpr int kRaw = 10;
  constexpr int kSlots = 16;

  float raw[kRaw] = {-100.0f, 900.0f, 42.0f,  7.0f,   100.0f,
                     97.0f,  421.0f,   71.0f,  -12.0f, 18.0f};
  float rawExpected = 900.0f;  // true max of raw

  float mn = raw[0], mx = raw[0];
  for (float v : raw) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  float center = (mn + mx) / 2.0f;
  float halfSpread = (mx - mn) / 2.0f;

  // Pad with the minimum, normalize into [-4,4], then pack into row 0 of the
  // 16x16 hall; slots 16..255 stay zero.
  constexpr int kHall = kSlots * kSlots;
  float input[kHall] = {0.0f};
  for (int i = 0; i < kSlots; ++i) {
    float v = i < kRaw ? raw[i] : mn;
    input[i] = (v - center) * (4.0f / halfSpread);
  }
  float expected = (rawExpected - center) * (4.0f / halfSpread);

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, kHall, 1};
  _mlir_ciface_db_max_16__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_max_16(&memref, &encArg0);

  float result = _mlir_ciface_db_max_16__decrypt__result0(&memref);
  float rawResult = result * (halfSpread / 4.0f) + center;

  std::printf("raw input (10) : [-100, -1000, 42, 7, 100, 987, 421, 71, -12, 18]\n");
  std::printf("padded to 16 with min (%.1f)\n", mn);
  std::printf("db.max (norm)  : %.5f   (expected %.5f)\n", result, expected);
  std::printf("db.max (raw)   : %.5f   (true max = %.5f)\n", rawResult,
              rawExpected);

  EXPECT_NEAR(result, expected, 1e-3f);

  free(encArg0.basePtr);
  free(memref.basePtr);
}
