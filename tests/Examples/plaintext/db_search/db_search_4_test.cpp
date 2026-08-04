// End-to-end plaintext-backend run of db.search_similar: encrypt a query,
// find the most similar database row, decrypt the winning row.  The database
// is baked into the circuit as a plaintext constant (the server's data);
// only the query is secret.  Query and rows are unit-normalized.
#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

extern "C" {
void _mlir_ciface_db_search_4(StridedMemRefType<float, 2>* result,
                              StridedMemRefType<float, 2>* input);

void _mlir_ciface_db_search_4__encrypt__arg0(
    StridedMemRefType<float, 2>* result, StridedMemRefType<float>* input);

void _mlir_ciface_db_search_4__decrypt__result0(
    StridedMemRefType<float>* result, StridedMemRefType<float, 2>* input);
}

TEST(DbSearch4Test, ReturnsMostSimilarRow) {
  // Database (baked into the circuit): note0 gym, note1 pasta, note2 running,
  // note3 taxes -- each unit-normalized.
  //   note0 [0.8, 0.6, 0, 0]   note1 [0, 0, 1, 0]
  //   note2 [0.6, 0.8, 0, 0]   note3 [0, 0, 0, 1]
  // Query "marathon training", endurance-heavy, unit length.
  float query[4] = {0.5f, 0.866f, 0.0f, 0.0f};
  float expected[4] = {0.6f, 0.8f, 0.0f, 0.0f};  // note2 (running) wins

  StridedMemRefType<float, 2> encArg0;
  StridedMemRefType<float> query0 = {query, query, 0, 4, 1};
  _mlir_ciface_db_search_4__encrypt__arg0(&encArg0, &query0);

  StridedMemRefType<float, 2> memref;
  _mlir_ciface_db_search_4(&memref, &encArg0);

  StridedMemRefType<float> decRes;
  _mlir_ciface_db_search_4__decrypt__result0(&decRes, &memref);

  std::printf("query          : [0.5, 0.866, 0, 0]  (endurance-heavy)\n");
  std::printf("db.search (row): [%.4f, %.4f, %.4f, %.4f]\n",
              decRes.data[decRes.offset + 0], decRes.data[decRes.offset + 1],
              decRes.data[decRes.offset + 2], decRes.data[decRes.offset + 3]);
  std::printf("expected (note2): [0.6, 0.8, 0, 0]\n");

  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(decRes.data[decRes.offset + i], expected[i], 0.05f);

  free(encArg0.basePtr);
  free(memref.basePtr);
  free(decRes.basePtr);
}
