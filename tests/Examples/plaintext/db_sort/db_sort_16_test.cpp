// Sort playground: edit kRaw and raw[] only.  Client pads to 16 with the
// maximum (pads sink to the end), normalizes into [-4,4], checks first kRaw.
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_sort_16(StridedMemRefType<float, 2>* result,
                             StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_sort_16__encrypt__arg0(
    StridedMemRefType<float, 2>* result, StridedMemRefType<float>* input);

void _mlir_ciface_db_sort_16__decrypt__result0(
    StridedMemRefType<float>* result, StridedMemRefType<float, 2>* input);
}

TEST(DbSort16Test, SortUpTo16RawValues) {
  constexpr int kRaw = 7;   // <-- how many real values (1..16)
  constexpr int kSlots = 16;

  float raw[kRaw] = {42.0f, 700.5f, 100.0f, 7.5f, 35.0f, 76.0f, 876.0f};

  // Answer key: computed here on the plain data, so you only edit raw[].
  float rawExpected[kRaw];
  std::copy(raw, raw + kRaw, rawExpected);
  std::sort(rawExpected, rawExpected + kRaw);

  float mn = raw[0], mx = raw[0];
  for (int i = 0; i < kRaw; ++i) {
    mn = std::min(mn, raw[i]);
    mx = std::max(mx, raw[i]);
  }
  float center = (mn + mx) / 2.0f;
  float halfSpread = (mx - mn) / 2.0f;

  // Pad with the maximum, then normalize everything into [-4, 4].
  float input[kSlots];
  for (int i = 0; i < kSlots; ++i) {
    float v = i < kRaw ? raw[i] : mx;
    input[i] = (v - center) * (4.0f / halfSpread);
  }

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, kSlots, 1};
  _mlir_ciface_db_sort_16__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_sort_16(&memref, &encArg0);

  StridedMemRefType<float> decRes;
  _mlir_ciface_db_sort_16__decrypt__result0(&decRes, &memref);

  float rawResult[kSlots];
  for (int i = 0; i < kSlots; ++i)
    rawResult[i] =
        decRes.data[decRes.offset + i] * (halfSpread / 4.0f) + center;

  std::printf("raw input (%d) : [", kRaw);
  for (int i = 0; i < kRaw; ++i)
    std::printf("%s%.2f", i ? ", " : "", raw[i]);
  std::printf("]\n");
  std::printf("db.sort (raw)  : [");
  for (int i = 0; i < kRaw; ++i)
    std::printf("%s%.2f", i ? ", " : "", rawResult[i]);
  std::printf("]   (+ %d pads at the end)\n", kSlots - kRaw);
  std::printf("expected       : [");
  for (int i = 0; i < kRaw; ++i)
    std::printf("%s%.2f", i ? ", " : "", rawExpected[i]);
  std::printf("]\n");

  for (int i = 0; i < kRaw; ++i)
    EXPECT_NEAR(rawResult[i], rawExpected[i], 0.05f * halfSpread / 4.0f + 0.05f);

  free(encArg0.basePtr);
  free(memref.basePtr);
  free(decRes.basePtr);
}
