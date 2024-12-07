#include "config.hpp"
#include <iostream>

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

Config::Config() : gui{false}, process{4, 2} {} // デフォルト値を設定

void Config::loadFromFile(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        
        if (config["gui"]) {
            gui.enabled = config["gui"]["enabled"].as<bool>();
        }

        if (config["process"]) {
            process.concurrency = config["process"]["concurrency"].as<int>();
            process.parallelism = config["process"]["parallelism"].as<int>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "設定ファイルの読み込みに失敗: " << e.what() << std::endl;
    }
}

void Config::printConfig() const {
    std::cout << "GUI設定:" << std::endl;
    std::cout << "  enabled: " << (gui.enabled ? "true" : "false") << std::endl;
    std::cout << "プロセス設定:" << std::endl;
    std::cout << "  concurrency: " << process.concurrency << std::endl;
    std::cout << "  parallelism: " << process.parallelism << std::endl;
}