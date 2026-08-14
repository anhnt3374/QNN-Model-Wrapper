#include "inference/shared_library.hpp"

#include <cassert>
#include <iostream>

using TestAddFn = int (*)(int, int);

int main()
{
    inference::SharedLibrary library;

    std::cout
        << "[INFO] Plugin path: "
        << TEST_PLUGIN_PATH
        << '\n';

    const bool loaded =
        library.open(TEST_PLUGIN_PATH);

    if (!loaded) {
        std::cerr
            << "[ERROR] Cannot load library: "
            << library.lastError()
            << '\n';

        return 1;
    }

    assert(loaded);
    assert(library.isOpen());

    auto testAdd =
        library.getSymbol<TestAddFn>("test_add");

    assert(testAdd != nullptr);

    const int result = testAdd(10, 20);

    assert(result == 30);

    library.close();

    assert(!library.isOpen());

    std::cout
        << "[PASS] SharedLibrary test\n";

    return 0;
}
