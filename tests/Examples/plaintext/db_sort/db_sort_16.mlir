// db.sort of 16 secret values; client pads with the maximum.
func.func @db_sort_16(%arg0: tensor<16xf32> {secret.secret}) -> tensor<16xf32> {
  %s = db.sort %arg0 : tensor<16xf32> -> tensor<16xf32>
  return %s : tensor<16xf32>
}
