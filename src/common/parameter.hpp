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
    struct GuiSettings { bool enabled = false; } gui;
    struct ProcessSettings { int concurrency = 0; int parallelism = 0; } process;
    struct TableSettings { int size = 0; } table;
    struct CacheSettings { bool alive = false; int size = 0; } cache;
    struct CircuitSettings { bool shuffle = false; int cancerMax = 0; string mode; bool verbose = false; bool timer = false; } circuit;

    // INI settings
    struct SchedulerHeuristics { bool alive = true; double cost_diag = 2.0; double cost_anti = 2.5; double cost_perm = 3.0; double cost_general = 4.0; } scheduler_heuristics;
    struct SchedulerAI { bool alive = true; } scheduler_ai;
    struct FuserHeuristics { bool alive = true; double score_diag_per_gate = 2.0; double score_same_axis_per_gate = 1.0; double score_phase_gadget = 4.0; double score_hcxh = 3.0; double model_bonus = 0.5; } fuser_heuristics;
    struct FuserAI { bool alive = true; } fuser_ai;
    struct General {
        bool rl = false; int window_size = 32; int top_k_levels = 8; int sig_dim = 6; int gate_feat_dim = 128; int max_qubits = 64;
        double gamma = 0.995; double lam = 0.95; double lr = 3e-4; double clip_eps = 0.2; double ent_coef = 0.01; double vf_coef = 0.5; int update_epochs = 5;
        string device = "cpu"; bool use_cpp_reward = true; double cpp_reward_prob = 0.02; double cpp_reward_alpha = 0.2;
    } general;

    static Parameter& getInstance();

    // 既定グローバルパス（fileutils::CONFIG_PATH / fileutils::SETTING_PATH）を使用
    void load();
    // 明示パス指定（必要なら）
    void loadFromFiles(const string& yamlFilepath = fileutils::CONFIG_PATH.string(),
                       const string& iniFilepath  = fileutils::SETTING_PATH.string());
    void print() const;

private:
    Parameter();
    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;
};

#define PARAMETER Parameter::getInstance()

#endif