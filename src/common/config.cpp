#include "config.hpp"
#include <iostream>

std::once_flag load_flag;

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

Config::Config() {}

void Config::loadFromFile(const std::string& filepath) {
    std::call_once(load_flag, [this, &filepath]() {
        cout << "Loading config file: " << std::endl;
        try {
            YAML::Node config = YAML::LoadFile(filepath);
            
            if (config["gui"]) {
                gui.enabled = config["gui"]["enabled"].as<bool>();
            }

            if (config["process"]) {
                process.concurrency = config["process"]["concurrency"].as<int>();
                process.parallelism = config["process"]["parallelism"].as<int>();
            }

            if (config["table"]) {
                table.size = config["table"]["size"].as<int>();
            }
        } catch (const YAML::Exception& e) {
            std::cerr << "設定ファイルの読み込みに失敗: " << e.what() << std::endl;
        }
    });
}

void Config::printConfig() const {
    std::cout << "GUI設定:" << std::endl;
    std::cout << "  enabled: " << (gui.enabled ? "true" : "false") << std::endl;
    std::cout << "プロセス設定:" << std::endl;
    std::cout << "  concurrency: " << process.concurrency << std::endl;
    std::cout << "  parallelism: " << process.parallelism << std::endl;
}