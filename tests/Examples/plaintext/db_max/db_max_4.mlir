// db.max of 4 secret values via the Paper 2 rank kernel.
// The client packs the 4 values into row 0 of a 4x4 hall (slots 0..3) and
// zeros slots 4..15 before encrypting.
func.func @db_max_4(%arg0: tensor<16xf32> {secret.secret}) -> f32 {
  %m = db.max %arg0 : tensor<16xf32> -> f32
  return %m : f32
}
