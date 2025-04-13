set(SRC_FILES
    Sampling.hpp
    Sampling.cpp
    LaplaceEvaluator.hpp
    LaplaceEvaluator.cpp
    PoissonEvaluator.hpp
    PoissonEvaluator.cpp
)
target_sources(TinyAlligator PUBLIC ${SRC_FILES})