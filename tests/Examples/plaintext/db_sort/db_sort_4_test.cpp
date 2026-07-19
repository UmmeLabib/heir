// End-to-end plaintext-backend run of db.sort on 4 raw values INCLUDING a
// duplicate (the minimum appears twice): tie-corrected badges put both
// copies into the sorted output at neighboring positions.
//
// Client flow as in db_max/db_min: min-max normalize into [-4, 4] before
// encrypting, un-normalize each output element after decrypting.
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_sort_4(StridedMemRefType<float, 2>* result,
                            StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_sort_4__encrypt__arg0(StridedMemRefType<float, 2>* result,
                                           StridedMemRefType<float>* input);

void _mlir_ciface_db_sort_4__decrypt__result0(
    StridedMemRefType<float>* result, StridedMemRefType<float, 2>* input);
}

TEST(DbSort4Test, SortWithDuplicate) {
  float raw[4] = {42.0f, -7.5f, 100.0f, -7.5f};
  float rawExpected[4] = {-7.5f, -7.5f, 42.0f, 100.0f};  // ascending

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

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> input0 = {input, input, 0, 4, 1};
  _mlir_ciface_db_sort_4__encrypt__arg0(&encArg0, &input0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_sort_4(&memref, &encArg0);

  StridedMemRefType<float> decRes;
  _mlir_ciface_db_sort_4__decrypt__result0(&decRes, &memref);

  float rawResult[4];
  for (int i = 0; i < 4; ++i)
    rawResult[i] =
        decRes.data[decRes.offset + i] * (halfSpread / 4.0f) + center;

  std::printf("raw input      : [42, -7.5, 100, -7.5]  (duplicate min)\n");
  std::printf("db.sort (raw)  : [%.4f, %.4f, %.4f, %.4f]\n", rawResult[0],
              rawResult[1], rawResult[2], rawResult[3]);
  std::printf("expected       : [-7.5, -7.5, 42, 100]\n");

  for (int i = 0; i < 4; ++i) EXPECT_NEAR(rawResult[i], rawExpected[i], 0.05f);

  free(encArg0.basePtr);
  free(memref.basePtr);
  free(decRes.basePtr);
}
