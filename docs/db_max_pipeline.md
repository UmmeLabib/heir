# db.max — Full Pipeline Walkthrough

How `db.max` computes the maximum of an encrypted vector using the Paper 2
rank kernel (Mazzone, Everts, Hahn, Peter. *Efficient Ranking, Order
Statistics, and Sorting under CKKS*, USENIX Security 2025). Lowering lives in
`lib/Dialect/DB/Transforms/DBToCiphertextSemantic.cpp`.

Running example: the user's raw tensor `[50, 90, 10, 70]`. True max = 90.

```
════════════════════════ CLIENT SIDE (plaintext, trusted) ═══════════════════════

   user's tensor:      [ 50 │ 90 │ 10 │ 70 ]        any range, any values

   Step 0a: find the box          min = 10, max = 90
                                  center     = (10+90)/2 = 50
                                  halfSpread = (90−10)/2 = 40

   Step 0b: normalize             x' = (x − 50) × (4/40)
             into [−4, 4]
                       [ 50 │ 90 │ 10 │ 70 ]
                              │ shift & stretch
                              ▼
                       [  0 │  4 │ −4 │  2 ]

   Step 0c: ENCRYPT  🔒          values sealed; only their COUNT (4) is public
                              │
                              ▼  send to server
══════════════════════ SERVER SIDE (encrypted, untrusted) ═══════════════════════

   S1  bouncer: 1-D ✓  power-of-2 ✓  float ✓  types ✓
   S2  contract fixed: differences ±8 → judge window via ×1/8

   S3  embedAsRow0 — move into the 16-seat hall:
                               ┌────┬────┬────┬────┐
                               │  0 │  4 │ −4 │  2 │
                               │  0 │  0 │  0 │  0 │
                               │  0 │  0 │  0 │  0 │
                               │  0 │  0 │  0 │  0 │
                               └────┴────┴────┴────┘
                            TransR ↙ (S5)        ↘ ReplR (S4)
              ┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
              │  0 │  0 │  0 │  0 │          │  0 │  4 │ −4 │  2 │
              │  4 │  0 │  0 │  0 │          │  0 │  4 │ −4 │  2 │
              │ −4 │  0 │  0 │  0 │          │  0 │  4 │ −4 │  2 │
              │  2 │  0 │  0 │  0 │          │  0 │  4 │ −4 │  2 │
              └────┴────┴────┴────┘          └────┴────┴────┴────┘
                     │ ReplC (S6)                    │ = VR (kept!)
              ┌────┬────┬────┬────┐                  │
              │  0 │  0 │  0 │  0 │                  │
              │  4 │  4 │  4 │  4 │ = VC             │
              │ −4 │ −4 │ −4 │ −4 │                  │
              │  2 │  2 │  2 │  2 │                  │
              └────┴────┴────┴────┘                  │
                     └───────────┬───────────────────┘
                                 ▼
                   S7  diff = VR − VC        (all 16 fights, one subtraction)
                   S8  ×1/8 → g,g,g → f,f → (+1)·0.5   (the judge)
                               ┌────┬────┬────┬────┐
                               │ .5 │  1 │  0 │  1 │
                          C =  │  0 │ .5 │  0 │  0 │
                               │  1 │  1 │ .5 │  1 │
                               │  0 │  1 │  0 │ .5 │
                               └────┴────┴────┴────┘
                 SumR ↙ (S9)                  ↘ e = 4·C·(1−C)  (S10)
     R = [ 1.5 │ 3.5 │ 0.5 │ 2.5 ]        E = identity here (no ties)
                 │                  S11:  U = SumR(E·uppertri) = [1 1 1 1]
                 │                        T = SumR(E)          = [1 1 1 1]
                 └──────────┬──────────────────┘
                            ▼
                 K = R + U − 0.5·T          badges, whole & unique
                     [  2 │  4 │  1 │  3 ]      (junk seats: 0)
                            │
                 S12  fences 3.5 / 4.5, scale 1/(4+1):
                      cmpLo = [0 1 0 0]   cmpHi = [0 0 0 0]
                      mask  = [0 1 0 0]        ← spotlight
                            │ × VR  (S13)
                      [  0 │  4 │  0 │  0 ]
                            │ fold-sum (rotate 8,4,2,1)
                      [  4 │  4 │  4 │  4 ]
                            │ extract seat 0
                            ▼
                   encrypted result: 🔒 4
                            │  send back
════════════════════════ CLIENT SIDE (plaintext) ════════════════════════════════

   Step 14a: DECRYPT  🔓          → 4.00000        (normalized scale)
   Step 14b: un-normalize          x = 4 × (40/4) + 50
                                     = 4 × 10 + 50 = 90  ✓

   answer: max of [50, 90, 10, 70] = 90
```

## Step legend

| Step | What happens | Function |
|---|---|---|
| 0 | client normalizes into [−4, 4], encrypts | test harness / client code |
| S1 | four safety checks (1-D, power-of-2, float, types) | `lowerOrderStatistic` |
| S2 | shrink factor 1/(2·bound) = 1/8 fixed at compile time | `kSortInputAbsBound` |
| S3 | vector → row 0 of the N²-slot hall, rest exact zeros | `embedAsRow0` |
| S4 | row copied to all rows = VR (rotations N·2ⁱ) | `replRow0` |
| S5 | row flipped into column (hops N(N−1)/2ⁱ + mask) | `transRow0ToCol0` |
| S6 | column smeared across columns = VC (rotations 2ⁱ) | `replCol0` |
| S7 | one subtraction = all N² pairwise fights | inline in kernel |
| S8 | judge: shrink → g,g,g → f,f → verdicts 1 / 0 / 0.5 | `cmpFromDiff`, `sharpSign`, `evalOddDeg7` |
| S9 | fold rows into row 0: win counts R | `sumRows` |
| S10 | tie radar e = 4c(1−c): lights up equal pairs | inline in kernel |
| S11 | tie split: U (place in group), T (group size), K = R+U−0.5T | inline in kernel |
| S12 | badge check at k = N: fences k±0.5, scale 1/(N+1) | `indicatorAround` |
| S13 | spotlight × VR → fold-sum → extract seat 0 | `rotateReduceAddVector` |
| 14 | client decrypts, un-normalizes | test harness / client code |

## Key facts

- **Comparison depth 2** regardless of N (rank judging + fence judging);
  old rotate-and-reduce needed log₂N sequential comparisons.
- **Everything public except values**: length N, rotation amounts, masks,
  fences, loop counts. The server never learns which slot won — the fold-sum
  smears the winner across all slots before extraction.
- **Ties are exact**: both f and g are odd polynomials, so sign(0) = 0 and
  Cmp = 0.5 precisely; tie-corrected badges are always a permutation of 1…N.
- **Client contract**: values normalized into [−4, 4]; gaps below ~1% of the
  spread may misrank (comparator resolution; raise `kNumG` for finer sight).
- **Scars** (bugs found by end-to-end testing, now guarded in code):
  1-D `insert_slice` crashes layout-propagation → scalar inserts;
  `expand_shape` unsupported → junk slots self-mask via Ind_k(0)=0;
  junk slot −(N+0.5) escaped the polynomial window → indicator scale 1/(N+1);
  piped bazel exit codes hide failures → check the log, verify binary mtime.

## Verified

- `tests/Examples/plaintext/db_max/db_max_4` — 4 values, exact to 1e-3.
- `tests/Examples/plaintext/db_max/db_max_16` — 10 raw values (any range)
  padded to 16 with the minimum; 7-way tie handled; max 987 → 987.00464.

## Still to migrate onto this kernel

db.min (indicator at k = 1), db.sort (Alg 5: replicate badges, all N fences
in one indicator), db.search_similar (rank dot-product scores, spotlight
badge N, return the row), then block-tournament for N > 128.
