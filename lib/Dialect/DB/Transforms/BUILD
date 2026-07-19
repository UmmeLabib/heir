load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":DBToCiphertextSemantic",
        ":pass_inc_gen",
        "@heir//lib/Dialect/DB/IR:Dialect",
    ],
)

cc_library(
    name = "DBToCiphertextSemantic",
    srcs = ["DBToCiphertextSemantic.cpp"],
    hdrs = ["DBToCiphertextSemantic.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/DB/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "Passes",
    td_file = "Passes.td",
)
