# See tests/Examples/openfhe/README.md for setup required to run these tests.

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# db.min under real CKKS, with the minimum duplicated so the rank kernel's
# index tie-break (e * (L - 0.5)) is exercised on encrypted data.
openfhe_end_to_end_test(
    name = "db_min_4_test",
    size = "enormous",
    generated_lib_header = "db_min_4_lib.h",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--annotate-module=backend=openfhe scheme=ckks",
        # waterline=0 keeps the mgmt pass from inserting bootstraps the kernel
        # no longer needs; see the db_max test for the level accounting.
        "--mlir-to-ckks=ciphertext-degree=1024 ckks-bootstrap-waterline=0",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/plaintext/db_min:db_min_4.mlir",
    tags = ["notap"],
    test_src = "db_min_4_test.cpp",
)
