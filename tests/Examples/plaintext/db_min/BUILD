load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# db.min end-to-end (tie at the minimum).
llvm_runner_test(
    name = "db_min_4",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "db_min_4_test.cpp",
    mlir_src = "db_min_4.mlir",
)
