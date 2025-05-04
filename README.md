mkdir cmake

cd cmake

cmake ../

make -j8

./alligator_play ../data/bunny.off 10 10 0.0001 0

./alligator_play ../data/bunny.off 10 10 0.0001 1

./alligator_play ../data/bunny.off 10 10 0.0001 2