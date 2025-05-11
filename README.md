mkdir cmake

cd cmake

cmake ../

make -j8

PARLAY_NUM_THREADS=8 ./alligator_play ../data/bunny.off 10 10 0.0001 0

PARLAY_NUM_THREADS=8 ./alligator_play ../data/bunny.off 10 10 0.0001 1

PARLAY_NUM_THREADS=8 ./alligator_play ../data/bunny.off 10 10 0.0001 2