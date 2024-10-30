#ifndef MONITOR_HPP
#define MONITOR_HPP

#include <yaml-cpp/yaml.h>
#include <omp.h>
#include <boost/fiber/all.hpp>
#include <Eigen/Core>
#include <mach/mach.h>
#include <chrono>
#include <iostream>
#include <functional>

using namespace std;

string getProcessType();
void parallelProcessing();
void sequentialProcessing();
void fiberProcessing();
void simdProcessing();
void printMemoryUsage();
void printMemoryUsageOnMac();
void measureExecutionTime(function<void()> func);
bool isExecuteGui();

#endif