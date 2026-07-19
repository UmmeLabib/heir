load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# db.sort end-to-end (4 slots, duplicated minimum).
llvm_runner_test(
    name = "db_sort_4",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "db_sort_4_test.cpp",
    mlir_src = "db_sort_4.mlir",
)

# db.sort playground (16 slots, client-side padding).
llvm_runner_test(
    name = "db_sort_16",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "db_sort_16_test.cpp",
    mlir_src = "db_sort_16.mlir",
)
