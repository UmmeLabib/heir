// RUN: heir-opt --db-to-linalg %s | FileCheck %s

// After lowering, no DB ops should remain.

// CHECK-LABEL: func.func @lower_max
// CHECK-NOT: db.max
// CHECK: linalg.reduce
// CHECK: arith.maxsi
func.func @lower_max(%input: tensor<1024xi16>) -> i16 {
  %m = db.max %input : tensor<1024xi16> -> i16
  return %m : i16
}

// CHECK-LABEL: func.func @lower_min
// CHECK-NOT: db.min
// CHECK: linalg.reduce
// CHECK: arith.minsi
func.func @lower_min(%input: tensor<1024xi16>) -> i16 {
  %m = db.min %input : tensor<1024xi16> -> i16
  return %m : i16
}

// CHECK-LABEL: func.func @lower_sort_ascending
// CHECK-NOT: db.sort
// CHECK: scf.for
// CHECK: tensor.insert
func.func @lower_sort_ascending(%input: tensor<64xi32>) -> tensor<64xi32> {
  %sorted = db.sort %input : tensor<64xi32> -> tensor<64xi32>
  return %sorted : tensor<64xi32>
}

// CHECK-LABEL: func.func @lower_sort_descending
// CHECK-NOT: db.sort
// CHECK: scf.for
// CHECK: arith.cmpi slt
func.func @lower_sort_descending(%input: tensor<64xi32>) -> tensor<64xi32> {
  %sorted = db.sort %input {descending = true} : tensor<64xi32> -> tensor<64xi32>
  return %sorted : tensor<64xi32>
}

// CHECK-LABEL: func.func @lower_search_similar
// CHECK-NOT: db.search_similar
// CHECK: linalg.matvec
// CHECK: scf.for
// CHECK: tensor.extract_slice
func.func @lower_search_similar(
    %query: tensor<128xf32>,
    %db: tensor<1024x128xf32>) -> tensor<128xf32> {
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<1024x128xf32> -> tensor<128xf32>
  return %row : tensor<128xf32>
}
