#pragma once
#include <iostream>
#include <chrono>

using namespace std::literals::chrono_literals;

class ScopedTimer {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;
public:
    ScopedTimer(const std::string& in);
    ~ScopedTimer();
};
