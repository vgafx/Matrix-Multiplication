#include <iostream>
#include <string>
#include "ScopedTimer.h"

int main()
{

    std::string version_name = "Sequential";
    ScopedTimer timer(version_name);
    std::cout << "Hello World!" << std::endl;
    return 0;


}
