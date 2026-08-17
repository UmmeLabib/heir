#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/db_max/db_max_4_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

namespace {

// Dump the raw CKKS ciphertext: shape, level, and the leading coefficients of
// each polynomial part, so the encrypted form is visible next to the cleartext.
void dumpCiphertext(const char* label, const std::vector<CiphertextT>& cts) {
  std::cout << "\n--- " << label << " ---\n";
  std::cout << "  ciphertexts in result : " << cts.size() << "\n";
  const auto& ct = cts[0];
  const auto& parts = ct->GetElements();
  std::cout << "  polynomial parts      : " << parts.size() << "  (c0, c1)\n";
  std::cout << "  RNS limbs (towers)    : " << parts[0].GetNumOfElements()
            << "\n";
  std::cout << "  ring dimension        : " << parts[0].GetRingDimension()
            << "\n";
  std::cout << "  level                 : " << ct->GetLevel() << "\n";
  for (size_t p = 0; p < parts.size(); ++p) {
    const auto& values = parts[p].GetElementAtIndex(0).GetValues();
    std::cout << "  c" << p << " (limb 0), first 6 coefficients:\n      ";
    for (size_t i = 0; i < 6 && i < values.GetLength(); ++i)
      std::cout << values[i] << (i < 5 ? ", " : "");
    std::cout << ", ...\n";
  }
}

}  // namespace

// db.max of 4 raw values under real CKKS.  The client normalizes into
// [-4, 4] (the comparator's input contract) and packs the values into row 0
// of the 4x4 hall; slots 4..15 stay zero.
TEST(DbMax4Test, RunTest) {
  auto cryptoContext = db_max_4__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = db_max_4__configure_crypto_context(cryptoContext, secretKey);

  float raw[4] = {-100.0f, -1000.0f, 42.0f, 7.0f};
  float rawExpected = 42.0f;

  float mn = raw[0], mx = raw[0];
  for (float v : raw) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  float center = (mn + mx) / 2.0f;
  float halfSpread = (mx - mn) / 2.0f;

  std::vector<float> arg0(16, 0.0f);
  for (int i = 0; i < 4; ++i)
    arg0[i] = (raw[i] - center) * (4.0f / halfSpread);
  float expected = (rawExpected - center) * (4.0f / halfSpread);

  std::printf("raw input   : [%.2f, %.2f, %.2f, %.2f]\n", raw[0], raw[1],
              raw[2], raw[3]);
  std::printf("normalized  : [%.4f, %.4f, %.4f, %.4f]\n", arg0[0], arg0[1],
              arg0[2], arg0[3]);
  std::printf("packed hall : slots 0..3 hold the values, slots 4..15 are 0\n");

  auto arg0Encrypted = db_max_4__encrypt__arg0(cryptoContext, arg0, publicKey);
  dumpCiphertext("ENCRYPTED INPUT (this is what the server sees)",
                 arg0Encrypted);

  auto outputEncrypted = db_max_4(cryptoContext, arg0Encrypted);
  dumpCiphertext("ENCRYPTED RESULT (still unreadable without the key)",
                 outputEncrypted);

  auto actual =
      db_max_4__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  std::printf("\ndb.max      : %.5f   (expected %.5f)\n", actual, expected);
  std::printf("db.max (raw): %.5f   (true max = %.5f)\n",
              actual * (halfSpread / 4.0f) + center, rawExpected);

  EXPECT_NEAR(expected, actual, 1e-2);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
