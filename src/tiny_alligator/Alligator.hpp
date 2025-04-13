#pragma once
#include <iostream>

class TinyAlligator
{
private:
    static TinyAlligator* tiny_alligator;

public:
    static void say_bye()
    {
        if (!tiny_alligator) {
            std::cout << "nobody there..." << std::endl;
        } else {
            delete tiny_alligator;
            tiny_alligator = nullptr;
            std::cout << "Tiny Alligator: See ya!" << std::endl;
        }
    }
    static void play_with_tiny_alligator()
    {
        if (!tiny_alligator) {
            std::cout << "nobody there..." << std::endl;
        } else {
            std::cout << "Tiny Alligator: I really really like water!" << std::endl;
        }
    }
    static TinyAlligator*& find_tiny_alligator()
    {
        if (tiny_alligator == nullptr) {
            tiny_alligator = new TinyAlligator();
        } else {
            std::cout << "Tiny Alligator: Hey! where you want to go, I am already here!"
                      << std::endl;
        }
        return tiny_alligator;
    }

private:
    TinyAlligator()
    {
        std::cout << "Tiny Alligator: Hey, I am tiny alligator. Welcome to my home!" << std::endl;
    }
    ~TinyAlligator() {}
};

TinyAlligator* TinyAlligator::tiny_alligator = nullptr;