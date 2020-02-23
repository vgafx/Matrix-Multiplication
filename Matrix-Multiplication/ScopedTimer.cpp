#include "ScopedTimer.h"


ScopedTimer::ScopedTimer(const std::string& ver) {
    std::cout << "Running " << ver << " version...\n";
    start = std::chrono::high_resolution_clock::now();
}

ScopedTimer::~ScopedTimer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Version took " << duration.count() << " seconds.\n";
}
