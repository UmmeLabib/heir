load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# db.search_similar end-to-end (4 docs, database as plaintext constant).
llvm_runner_test(
    name = "db_search_4",
    heir_opt_flags = [
        "--db-to-ciphertext-semantic",
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "db_search_4_test.cpp",
    mlir_src = "db_search_4.mlir",
)
