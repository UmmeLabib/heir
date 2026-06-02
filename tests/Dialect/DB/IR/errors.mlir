// RUN: heir-opt --split-input-file --verify-diagnostics %s

// -----

// SearchSimilarOp: query must be 1-D.
func.func @search_similar_query_not_1d(
    %query: tensor<4x128xf32>, // intentionally wrong: rank 2
    %db: tensor<1024x128xf32>) -> tensor<128xf32> {
  // expected-error@+1 {{query must be a 1-D tensor, but got rank 2}}
  %row = db.search_similar %query, %db
      : tensor<4x128xf32>, tensor<1024x128xf32> -> tensor<128xf32>
  return %row : tensor<128xf32>
}

// -----

// SearchSimilarOp: database must be 2-D.
func.func @search_similar_db_not_2d(
    %query: tensor<128xf32>,
    %db: tensor<4x256x128xf32>) -> tensor<128xf32> {
  // expected-error@+1 {{database must be a 2-D tensor, but got rank 3}}
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<4x256x128xf32> -> tensor<128xf32>
  return %row : tensor<128xf32>
}

// -----

// SearchSimilarOp: query length must match database feature dimension.
func.func @search_similar_dim_mismatch(
    %query: tensor<64xf32>,
    %db: tensor<1024x128xf32>) -> tensor<64xf32> {
  // expected-error@+1 {{query length (64) must match database feature dimension (128)}}
  %row = db.search_similar %query, %db
      : tensor<64xf32>, tensor<1024x128xf32> -> tensor<64xf32>
  return %row : tensor<64xf32>
}

// -----

// SearchSimilarOp: element types must agree.
func.func @search_similar_type_mismatch(
    %query: tensor<128xf32>,
    %db: tensor<1024x128xi16>) -> tensor<128xf32> {
  // expected-error@+1 {{query and database must have the same element type}}
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<1024x128xi16> -> tensor<128xf32>
  return %row : tensor<128xf32>
}

// -----

// SearchSimilarOp: result must be 1-D.
func.func @search_similar_result_not_1d(
    %query: tensor<128xf32>,
    %db: tensor<1024x128xf32>) -> tensor<1x128xf32> {
  // expected-error@+1 {{result must be a 1-D tensor, but got rank 2}}
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<1024x128xf32> -> tensor<1x128xf32>
  return %row : tensor<1x128xf32>
}

// -----

// SearchSimilarOp: result element type must match query element type.
func.func @search_similar_result_elem_mismatch(
    %query: tensor<128xf32>,
    %db: tensor<1024x128xf32>) -> tensor<128xf64> {
  // expected-error@+1 {{result element type must match query element type}}
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<1024x128xf32> -> tensor<128xf64>
  return %row : tensor<128xf64>
}

// -----

// SearchSimilarOp: result length must match query length.
func.func @search_similar_result_len_mismatch(
    %query: tensor<128xf32>,
    %db: tensor<1024x128xf32>) -> tensor<64xf32> {
  // expected-error@+1 {{result length (64) must match query length (128)}}
  %row = db.search_similar %query, %db
      : tensor<128xf32>, tensor<1024x128xf32> -> tensor<64xf32>
  return %row : tensor<64xf32>
}

// -----

// MaxOp: result type must match element type.
func.func @max_wrong_result(%input: tensor<1024xi16>) -> i32 {
  // expected-error@+1 {{result type must match the element type of the input tensor}}
  %m = db.max %input : tensor<1024xi16> -> i32
  return %m : i32
}

// -----

// MinOp: result type must match element type.
func.func @min_wrong_result(%input: tensor<1024xi16>) -> i32 {
  // expected-error@+1 {{result type must match the element type of the input tensor}}
  %m = db.min %input : tensor<1024xi16> -> i32
  return %m : i32
}

// -----

// SortOp: input must be 1-D.
func.func @sort_input_not_1d(%input: tensor<4x64xi32>) -> tensor<4x64xi32> {
  // expected-error@+1 {{input must be a 1-D tensor, but got rank 2}}
  %sorted = db.sort %input : tensor<4x64xi32> -> tensor<4x64xi32>
  return %sorted : tensor<4x64xi32>
}

// -----

// SortOp: result type must match input type.
func.func @sort_result_type_mismatch(%input: tensor<64xi32>) -> tensor<64xi16> {
  // expected-error@+1 {{result type must match the input type}}
  %sorted = db.sort %input : tensor<64xi32> -> tensor<64xi16>
  return %sorted : tensor<64xi16>
}
