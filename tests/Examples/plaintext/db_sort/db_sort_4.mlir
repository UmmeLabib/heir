// db.sort of 4 secret values (Alg 5: one zero-indicator, all hunts).
func.func @db_sort_4(%arg0: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %s = db.sort %arg0 : tensor<4xf32> -> tensor<4xf32>
  return %s : tensor<4xf32>
}
