#pragma once
#include <yaml-cpp/yaml.h>
#include <string>
#include <mutex>

using namespace std;

class Config {
public:
    struct GuiSettings {
        bool enabled;
    } gui;

    struct ProcessSettings {
        int concurrency;
        int parallelism;
    } process;

    struct TableSettings {
        int size;
    } table;

    struct CacheSettings {
        bool alive;
        int size;
    } cache;

    struct CircuitSettings {
        bool shuffle;
        int cancerMax;
        bool joker;
        bool verbose;
        bool timer;
    } circuit;

    static Config& getInstance();
    void loadFromFile(const std::string& filepath);
    void printConfig() const;

private:
    Config();
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
};

#define CONFIG Config::getInstance()