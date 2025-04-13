#include <catch2/catch_test_macros.hpp>
#include "tiny_alligator/Alligator.hpp"

TEST_CASE("Hello_TinyAlligator!", "[tiny_alligator]")
{
    TinyAlligator::play_with_tiny_alligator();
    TinyAlligator*& alligator = TinyAlligator::find_tiny_alligator();
    alligator->play_with_tiny_alligator();
    alligator->say_bye();
    CHECK(alligator == nullptr);
}