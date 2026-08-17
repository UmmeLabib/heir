# See tests/Examples/openfhe/README.md for setup required to run these tests.

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# db.max under real CKKS: the Paper 2 rank kernel encrypted, evaluated, and
# decrypted through OpenFHE.  The client packs its 4 values into row 0 of the
# 4x4 hall before encrypting (see the db.max op description).
openfhe_end_to_end_test(
    name = "db_max_4_test",
    size = "enormous",
    generated_lib_header = "db_max_4_lib.h",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--annotate-module=backend=openfhe scheme=ckks",
        # waterline=0 keeps the mgmt pass from inserting bootstraps the kernel
        # no longer needs: the row-product indicator fits in 31 levels, while
        # each bootstrap adds its encode/decode budget on top (mulDepth 35 vs
        # 51) and overflows OpenFHE's modulus arithmetic.
        "--mlir-to-ckks=ciphertext-degree=1024 ckks-bootstrap-waterline=0",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/plaintext/db_max:db_max_4.mlir",
    tags = ["notap"],
    test_src = "db_max_4_test.cpp",
)
