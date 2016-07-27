#include "constants.h"
#include "dataset_writer.h"

#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>

// Converts text samples from stdin (one per line, fields separated by space) to binary dataset (stdout)

static const char FieldSeparator = ' ';

std::vector<std::string> splitLine(std::string const & line)
{
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, FieldSeparator)) {
        tokens.push_back(std::move(token));
    }

    return tokens;
}

// Parses line translating y (target value) into -1 or 1.
Dataset::Sample parseLine(std::string const & line)
{
    const std::vector<std::string> & strings = splitLine(line);

    Dataset::Sample sample;
    for (int idx = 0, endIdx = strings.size() - 1; idx < endIdx; ++idx) {
        sample.push_back(std::stoi(strings[idx]));
    }

    const std::string & yString = strings[strings.size() - 1];
    int const y = (yString == "0" || yString == "-1") ? -1 : 1;
    sample.push_back(y);

    return sample;
}

int main()
{
    std::cin.sync_with_stdio(false);

    DatasetWriter datasetWriter(stdout);
    std::string line;
    while (std::getline(std::cin, line) && line != "") {
        Dataset::Sample const & xy = parseLine(line);
        datasetWriter.writeSample(xy);
    }

    datasetWriter.writeFooter();

    return 0;
}
