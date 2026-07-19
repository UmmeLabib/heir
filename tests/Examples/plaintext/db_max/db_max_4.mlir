// db.max of 4 secret values via the Paper 2 rank kernel.
func.func @db_max_4(%arg0: tensor<4xf32> {secret.secret}) -> f32 {
  %m = db.max %arg0 : tensor<4xf32> -> f32
  return %m : f32
}
