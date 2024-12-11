#pragma once
#include <yaml-cpp/yaml.h>
#include <string>

#ifndef ENTRY_COUNT
#define ENTRY_COUNT 1000000
#endif

class Config {
public:
    struct GuiSettings {
        bool enabled;
    } gui;

    struct ProcessSettings {
        int concurrency;
        int parallelism;
    } process;

    static Config& getInstance();
    void loadFromFile(const std::string& filepath);
    void printConfig() const;

private:
    Config();
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
};

#define CONFIG Config::getInstance()