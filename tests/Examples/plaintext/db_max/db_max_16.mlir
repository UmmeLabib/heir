// db.max of 16 secret values; client pads shorter inputs with the minimum.
// The client packs the 16 values into row 0 of a 16x16 hall (slots 0..15) and
// zeros slots 16..255 before encrypting.
func.func @db_max_16(%arg0: tensor<256xf32> {secret.secret}) -> f32 {
  %m = db.max %arg0 : tensor<256xf32> -> f32
  return %m : f32
}
