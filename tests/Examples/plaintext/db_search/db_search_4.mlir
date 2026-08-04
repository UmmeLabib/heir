// db.search_similar: return the database row most similar to the query, via
// the Paper 2 rank kernel (score -> rank -> spotlight -> masked row).  The
// database is a plaintext constant (the server's data); only the query is
// secret.  Rows and query are unit-normalized (client contract).
func.func @db_search_4(%q: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %db = arith.constant dense<[[0.8, 0.6, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.6, 0.8, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
  %r = db.search_similar %q, %db : tensor<4xf32>, tensor<4x4xf32> -> tensor<4xf32>
  return %r : tensor<4xf32>
}
