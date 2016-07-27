#ifndef CUDA_CLI_H
#define CUDA_CLI_H

#include <vector>
#include <string>

std::vector<std::string> argvToArgs(int const argc, const char ** argv)
{
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

#endif // CUDA_CLI_H
