list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/tests")
include(catch2)
FetchContent_GetProperties(catch2)
include(Catch)

# Sources
set(TEST_SOURCES
    tiny_alligator.cpp
)
add_executable(alligator_tests ${TEST_SOURCES})

target_link_libraries(alligator_tests PUBLIC
    Catch2::Catch2WithMain
    TinyAlligator
    )

catch_discover_tests(alligator_tests)