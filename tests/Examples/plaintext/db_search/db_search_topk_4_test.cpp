// End-to-end top-2 db.search_similar: the result concatenates the two most
// similar rows, best first.  Query "marathon training" (endurance-heavy):
//   note2 running [0.6,0.8,0,0] wins (score ~0.993),
//   note0 gym     [0.8,0.6,0,0] second (score ~0.920).
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_search_topk_4(StridedMemRefType<float, 2>* result,
                                   StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_search_topk_4__encrypt__arg0(
    StridedMemRefType<float, 2>* result, StridedMemRefType<float>* input);

void _mlir_ciface_db_search_topk_4__decrypt__result0(
    StridedMemRefType<float>* result, StridedMemRefType<float, 2>* input);
}

TEST(DbSearchTopk4Test, ReturnsTopTwoRows) {
  float query[4] = {0.5f, 0.866f, 0.0f, 0.0f};
  float expected[8] = {0.6f, 0.8f, 0.0f, 0.0f,   // best: note2 (running)
                       0.8f, 0.6f, 0.0f, 0.0f};  // 2nd: note0 (gym)

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> query0 = {query, query, 0, 4, 1};
  _mlir_ciface_db_search_topk_4__encrypt__arg0(&encArg0, &query0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_search_topk_4(&memref, &encArg0);

  StridedMemRefType<float> decRes;
  _mlir_ciface_db_search_topk_4__decrypt__result0(&decRes, &memref);

  std::printf("query        : [0.5, 0.866, 0, 0]\n");
  std::printf("top-2 (rows) : [%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f]\n",
              decRes.data[decRes.offset + 0], decRes.data[decRes.offset + 1],
              decRes.data[decRes.offset + 2], decRes.data[decRes.offset + 3],
              decRes.data[decRes.offset + 4], decRes.data[decRes.offset + 5],
              decRes.data[decRes.offset + 6], decRes.data[decRes.offset + 7]);
  std::printf("expected     : [0.6 0.8 0 0 | 0.8 0.6 0 0]  (note2, note0)\n");

  for (int i = 0; i < 8; ++i)
    EXPECT_NEAR(decRes.data[decRes.offset + i], expected[i], 0.05f);

  free(encArg0.basePtr);
  free(memref.basePtr);
  free(decRes.basePtr);
}
