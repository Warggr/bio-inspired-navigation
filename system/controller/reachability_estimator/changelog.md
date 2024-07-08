# Changelog
## Jul 01
make the FC layers of the network RE take (always) `$INPUT_DIM -> 256 -> 32 -> 4`, instead of `$INPUT_DIM -> $INPUT_DIM // 2 -> $INPUT_DIM // 2 -> 4`
## Jul 05, ~19:00
use different loss weights for reachable and non-reachable
