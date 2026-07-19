// db.max of 16 secret values; client pads shorter inputs with the minimum.
func.func @db_max_16(%arg0: tensor<16xf32> {secret.secret}) -> f32 {
  %m = db.max %arg0 : tensor<16xf32> -> f32
  return %m : f32
}
