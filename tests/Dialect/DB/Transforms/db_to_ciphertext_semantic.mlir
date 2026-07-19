// RUN: heir-opt --db-to-ciphertext-semantic %s | FileCheck %s

// All ops lower via the Paper 2 rank kernel: no db op survives, and the
// output is only slot-wise arith, tensor_ext.rotate, and tensor ops with
// public indices (no math_ext.sign -- the f/g comparator is inlined).

// CHECK-LABEL: func.func @lower_max
// CHECK-NOT: db.max
// CHECK-NOT: math_ext.sign
// CHECK: tensor_ext.rotate
// CHECK: arith.mulf
// CHECK: tensor.extract
func.func @lower_max(%input: tensor<4xf32>) -> f32 {
  %m = db.max %input : tensor<4xf32> -> f32
  return %m : f32
}

// CHECK-LABEL: func.func @lower_min
// CHECK-NOT: db.min
// CHECK: tensor_ext.rotate
// CHECK: arith.mulf
// CHECK: tensor.extract
func.func @lower_min(%input: tensor<4xf32>) -> f32 {
  %m = db.min %input : tensor<4xf32> -> f32
  return %m : f32
}

// CHECK-LABEL: func.func @lower_sort
// CHECK-NOT: db.sort
// CHECK: tensor_ext.rotate
// CHECK: arith.subf
// CHECK: arith.mulf
func.func @lower_sort(%input: tensor<4xf32>) -> tensor<4xf32> {
  %s = db.sort %input : tensor<4xf32> -> tensor<4xf32>
  return %s : tensor<4xf32>
}

// CHECK-LABEL: func.func @lower_sort_descending
// CHECK-NOT: db.sort
func.func @lower_sort_descending(%input: tensor<8xf32>) -> tensor<8xf32> {
  %s = db.sort %input {descending = true} : tensor<8xf32> -> tensor<8xf32>
  return %s : tensor<8xf32>
}

// Unsupported inputs are left untouched (integer element type: the 0.5-valued
// Cmp arithmetic is CKKS/float-only).

// CHECK-LABEL: func.func @reject_integer_max
// CHECK: db.max
func.func @reject_integer_max(%input: tensor<8xi16>) -> i16 {
  %m = db.max %input : tensor<8xi16> -> i16
  return %m : i16
}
