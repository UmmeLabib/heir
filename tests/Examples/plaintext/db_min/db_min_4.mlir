// db.min of 4 secret values (indicator hunts badge 1).
// The client packs the 4 values into row 0 of a 4x4 hall (slots 0..3) and
// zeros slots 4..15 before encrypting.
func.func @db_min_4(%arg0: tensor<16xf32> {secret.secret}) -> f32 {
  %m = db.min %arg0 : tensor<16xf32> -> f32
  return %m : f32
}
