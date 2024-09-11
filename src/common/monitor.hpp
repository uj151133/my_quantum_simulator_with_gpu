#ifndef MONITOR_HPP
#define MONITOR_HPP

#include <yaml-cpp/yaml.h>
#include <omp.h>
#include <boost/fiber/all.hpp>
#include <mach/mach.h>

using namespace std;

string getProcessType();
void parallelProcessing();
void sequentialProcessing();
void fiberProcessing();
void printMemoryUsage();
void printMemoryUsageOnMac();
bool isExecuteGui();

#endif