#ifndef MONITOR_HPP
#define MONITOR_HPP

#pragma once

#include <yaml-cpp/yaml.h>
#include <omp.h>
#include <boost/fiber/all.hpp>
#include <fstream>
#include <limits.h>
#ifdef __APPLE__
    #include <mach/mach.h>
#elif defined(__linux__)
    #include <sys/sysinfo.h>
    #include <sys/resource.h>
    #include <unistd.h>
#endif

#include <chrono>
#include <iostream>
#include <functional>

using namespace std;

string getProcessType();
void parallelProcessing();
void sequentialProcessing();
void fiberProcessing();
void printMemoryUsage();
#ifdef __APPLE__
void printMemoryUsageOnMac();
#elif defined(__linux__)
void printMemoryUsageOnLinux();
#endif
void measureExecutionTime(function<void()> func);
bool isExecuteGui();

#endif