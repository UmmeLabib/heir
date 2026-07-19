// db.min of 4 secret values (indicator hunts badge 1).
func.func @db_min_4(%arg0: tensor<4xf32> {secret.secret}) -> f32 {
  %m = db.min %arg0 : tensor<4xf32> -> f32
  return %m : f32
}
