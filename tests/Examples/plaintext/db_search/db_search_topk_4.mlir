// db.search_similar top-2: return the two most similar rows, concatenated.
// Database is a plaintext constant; only the query is secret.
func.func @db_search_topk_4(%q: tensor<4xf32> {secret.secret}) -> tensor<8xf32> {
  %db = arith.constant dense<[[0.8, 0.6, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.6, 0.8, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
  %r = db.search_similar %q, %db {k = 2 : i64}
      : tensor<4xf32>, tensor<4x4xf32> -> tensor<8xf32>
  return %r : tensor<8xf32>
}
