// RUN: heir-opt %s | FileCheck %s

module {

  // CHECK-LABEL: func.func @search_similar_static
  func.func @search_similar_static(
      %query: tensor<128xf32>,
      %db: tensor<1024x128xf32>) -> tensor<128xf32> {
    // CHECK: db.search_similar
    %row = db.search_similar %query, %db
        : tensor<128xf32>, tensor<1024x128xf32> -> tensor<128xf32>
    return %row : tensor<128xf32>
  }

  // CHECK-LABEL: func.func @search_similar_dynamic_rows
  func.func @search_similar_dynamic_rows(
      %query: tensor<128xf32>,
      %db: tensor<?x128xf32>) -> tensor<128xf32> {
    // CHECK: db.search_similar
    %row = db.search_similar %query, %db
        : tensor<128xf32>, tensor<?x128xf32> -> tensor<128xf32>
    return %row : tensor<128xf32>
  }

  // CHECK-LABEL: func.func @search_similar_fully_dynamic
  func.func @search_similar_fully_dynamic(
      %query: tensor<?xf32>,
      %db: tensor<?x?xf32>) -> tensor<?xf32> {
    // CHECK: db.search_similar
    %row = db.search_similar %query, %db
        : tensor<?xf32>, tensor<?x?xf32> -> tensor<?xf32>
    return %row : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @max_1d
  func.func @max_1d(%input: tensor<1024xi16>) -> i16 {
    // CHECK: db.max
    %m = db.max %input : tensor<1024xi16> -> i16
    return %m : i16
  }

  // CHECK-LABEL: func.func @min_1d
  func.func @min_1d(%input: tensor<1024xi16>) -> i16 {
    // CHECK: db.min
    %m = db.min %input : tensor<1024xi16> -> i16
    return %m : i16
  }

  // CHECK-LABEL: func.func @sort_ascending
  func.func @sort_ascending(%input: tensor<64xi32>) -> tensor<64xi32> {
    // CHECK: db.sort
    %sorted = db.sort %input : tensor<64xi32> -> tensor<64xi32>
    return %sorted : tensor<64xi32>
  }

  // CHECK-LABEL: func.func @sort_descending
  func.func @sort_descending(%input: tensor<64xi32>) -> tensor<64xi32> {
    // CHECK: db.sort
    // CHECK-SAME: descending = true
    %sorted = db.sort %input {descending = true} : tensor<64xi32> -> tensor<64xi32>
    return %sorted : tensor<64xi32>
  }

}
