#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include <yaml-cpp/yaml.h>
#include <string>
#include <mutex>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "fileUtils.hpp"

using namespace std;

class Parameter {
public:
    // YAML settings
    struct GuiSettings {
        bool enabled = false;
    } gui;
    struct ProcessSettings {
        // int concurrency = 0;
        bool parallel = false;
        int parallelism = 0;
    } process;
    struct TableSettings {
        int size = 0;
    } table;
    struct CacheSettings {
        bool alive = false;
        int size = 0;
        int TLSSize = 0;
    } cache;
    struct DealerSettings {
        bool alive = false;
        double nonzeroWeight = .0;
        double controlBitWeight = .0;
        double targetBitWeight = .0;
        double shorteningWeight = .0;
    };
    struct CircuitSettings {
        int cancerMax = 0;
        string mode;
        bool verbose = false;
        bool timer = false;
        DealerSettings dealer;
    } circuit;

    // INI settings
    struct SchedulerHeuristics {
        bool alive = true;
        double costDiag = 2.0;
        double costAnti = 2.5;
        double costPerm = 3.0;
        double costGeneral = 4.0;
    } schedulerHeuristics;
    struct SchedulerAI {
        bool alive = true;
    } schedulerAI;
    struct FuserHeuristics {
        bool alive = true;
        double scoreDiagPerGate = 2.0;
        double scoreSameAxisPerGate = 1.0;
        double scorePhaseGadget = 4.0;
        double scoreHcxh = 3.0;
        double modelBonus = 0.5;
    } fuserHeuristics;

    struct FuserAI {
        bool alive = true;
    } fuserAI;
    struct General {
        bool rl = false;
        int windowSize = 32;
        int topKLevels = 8;
        int sigDim = 6;
        int gateFeatDim = 128;
        int maxQubits = 64;
        double gamma = 0.995;
        double lam = 0.95;
        double lr = 3e-4;
        double clipEps = 0.2;
        double entCoef = 0.01;
        double vfCoef = 0.5;
        int updateEpochs = 5;
        string device = "cpu";
        bool useCppReward = true;
        double cppRewardProb = 0.02;
        double cppRewardAlpha = 0.2;
    } general;

    static Parameter& getInstance();

    // 既定グローバルパス（fileUtils::CONFIG_PATH / fileUtils::SETTING_PATH）を使用
    void load();
    // 明示パス指定（必要なら）
    void loadFromFile(const string& yamlFilepath = fileUtils::getConfigPath().string(),
                      const string& iniFilepath  = fileUtils::getSettingPath().string());
    void print() const;

private:
    Parameter();
    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;
};

#define PARAMETER Parameter::getInstance()

#endif