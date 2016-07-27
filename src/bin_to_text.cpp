#include "dataset.h"

#include <iostream>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <iterator>

// Reads binary samples from file pointed by argv[1] and writes text features to stdout
int main(int argc, const char ** argv)
{
    assert(argc == 2);
    std::cout.sync_with_stdio(false);

    Dataset dataset(argv[1]);
    while (dataset.hasNext()) {
        auto const & sample = dataset.next();
        std::copy(sample.begin(), sample.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }

    return 0;
}
