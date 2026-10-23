#include "parameter.hpp"
#include "fileUtils.hpp"
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <string>
#include <iostream>

namespace {
    once_flag parameter_load_flag;

    // 簡易INIパーサ（依存なし）
    class IniConfig {
    public:
        bool load(const string& path) {
            ifstream in(path);
            if (!in.is_open()) return false;

            string line;
            string section;
            while (getline(in, line)) {
                trim(line);
                if (line.empty()) continue;
                if (line[0] == ';' || line[0] == '#') continue;
                if (line.front() == '[' && line.back() == ']') {
                    section = line.substr(1, line.size() - 2);
                    trim(section);
                    continue;
                }
                auto pos = line.find('=');
                if (pos == string::npos) continue;
                string key = line.substr(0, pos);
                string val = line.substr(pos + 1);
                trim(key);
                trim(val);
                this->data_[section][key] = val;
            }
            return true;
        }

        string getString(const string& section, const string& key, const string& def) const {
            auto sit = this->data_.find(section);
            if (sit == this->data_.end()) return def;
            auto kit = sit->second.find(key);
            if (kit == sit->second.end()) return def;
            return kit->second;
        }

        bool getBool(const string& section, const string& key, bool def) const {
            auto s = toLower(this->getString(section, key, def ? "true" : "false"));
            if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
            if (s == "false" || s == "0" || s == "no" || s == "off") return false;
            return def;
        }

        int getInt(const string& section, const string& key, int def) const {
            auto s = this->getString(section, key, to_string(def));
            try { return stoi(s); } catch (...) { return def; }
        }

        double getDouble(const string& section, const string& key, double def) const {
            auto s = this->getString(section, key, to_string(def));
            try { return stod(s); } catch (...) { return def; }
        }

    private:
        static void ltrim(string& s) {
            s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char c){ return !isspace(c); }));
        }
        static void rtrim(string& s) {
            s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char c){ return !isspace(c); }).base(), s.end());
        }
        static void trim(string& s) { ltrim(s); rtrim(s); }
        static string toLower(string s) {
            transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return tolower(c); });
            return s;
        }

        unordered_map<string, unordered_map<string, string>> data_;
    };
} // namespace

Parameter& Parameter::getInstance() {
    static Parameter instance;
    return instance;
}

Parameter::Parameter() {}

void Parameter::load() {
    fileUtils::resolveFilePath();
    this->loadFromFile(fileUtils::CONFIG_PATH.string(), fileUtils::SETTING_PATH.string());
}

void Parameter::loadFromFile(const string& yamlFilepath, const string& iniFilepath) {
    call_once(parameter_load_flag, [this, &yamlFilepath, &iniFilepath]() {
        cout << "Loading " << yamlFilepath << " and " << iniFilepath << " ..." << endl;

        // YAML
        try {
            YAML::Node config = YAML::LoadFile(yamlFilepath);

            if (config["gui"]) {
                this->gui.enabled = config["gui"]["enabled"].as<bool>();
            }

            if (config["parallelism"]) {
                // this->process.concurrency = config["process"]["concurrency"].as<int>();
                this->parallelism.fiber = config["parallelism"]["fiber"].as<int>();
                this->parallelism.GPU = config["parallelism"]["GPU"].as<int>();
                if (config["parallelism"]["threads"]) {
                    this->parallelism.threads = config["parallelism"]["threads"].as<int>();
                }
            }

            if (config["table"]) {
                this->table.size = config["table"]["size"].as<int>();
            }

            if (config["cache"]) {
                this->cache.alive = config["cache"]["alive"].as<bool>();
                this->cache.size = config["cache"]["size"].as<int>();
                this->cache.FLSSize = config["cache"]["FLSSize"].as<int>();
            }

            if (config["circuit"]) {
                const auto& circuitConfig = config["circuit"];
                if (circuitConfig["mode"])      this->circuit.mode = circuitConfig["mode"].as<string>();
                if (circuitConfig["cancerMax"]) this->circuit.cancerMax = circuitConfig["cancerMax"].as<int>();
                if (circuitConfig["verbose"])   this->circuit.verbose = circuitConfig["verbose"].as<bool>();
                if (circuitConfig["timer"])     this->circuit.timer = circuitConfig["timer"].as<bool>();

                if (circuitConfig["Dealer"]) {
                    const auto& dealerConfig = circuitConfig["Dealer"];
                    if (dealerConfig["alive"])            this->circuit.dealer.alive = dealerConfig["alive"].as<bool>();
                    if (dealerConfig["nonzeroWeight"])    this->circuit.dealer.nonzeroWeight = dealerConfig["nonzeroWeight"].as<double>();
                    if (dealerConfig["controlBitWeight"]) this->circuit.dealer.controlBitWeight = dealerConfig["controlBitWeight"].as<double>();
                    if (dealerConfig["targetBitWeight"])  this->circuit.dealer.targetBitWeight = dealerConfig["targetBitWeight"].as<double>();
                    if (dealerConfig["shorteningWeight"]) this->circuit.dealer.shorteningWeight = dealerConfig["shorteningWeight"].as<double>();
                }
            }
        } catch (const YAML::Exception& e) {
            cerr << "YAML設定ファイルの読み込みに失敗: " << e.what() << endl;
        }

        // INI
        IniConfig ini;
        if (!ini.load(iniFilepath)) {
            cerr << "INI設定ファイルの読み込みに失敗: " << iniFilepath << endl;
            return;
        }

        // Scheduler.Heuristics
        this->schedulerHeuristics.alive        = ini.getBool("Scheduler.Heuristics", "alive", this->schedulerHeuristics.alive);
        this->schedulerHeuristics.costDiag    = ini.getDouble("Scheduler.Heuristics", "cost_diag", this->schedulerHeuristics.costDiag);
        this->schedulerHeuristics.costAnti    = ini.getDouble("Scheduler.Heuristics", "cost_anti", this->schedulerHeuristics.costAnti);
        this->schedulerHeuristics.costPerm    = ini.getDouble("Scheduler.Heuristics", "cost_perm", this->schedulerHeuristics.costPerm);
        this->schedulerHeuristics.costGeneral = ini.getDouble("Scheduler.Heuristics", "cost_general", this->schedulerHeuristics.costGeneral);

        // Scheduler.AI
        this->schedulerAI.alive                = ini.getBool("Scheduler.AI", "alive", this->schedulerAI.alive);
        // Fuser.Heuristics
        this->fuserHeuristics.alive                    = ini.getBool("Fuser.Heuristics", "alive", this->fuserHeuristics.alive);
        this->fuserHeuristics.scoreDiagPerGate      = ini.getDouble("Fuser.Heuristics", "score_diag_per_gate", this->fuserHeuristics.scoreDiagPerGate);
        this->fuserHeuristics.scoreSameAxisPerGate = ini.getDouble("Fuser.Heuristics", "score_same_axis_per_gate", this->fuserHeuristics.scoreSameAxisPerGate);
        this->fuserHeuristics.scorePhaseGadget       = ini.getDouble("Fuser.Heuristics", "score_phase_gadget", this->fuserHeuristics.scorePhaseGadget);
        this->fuserHeuristics.scoreHcxh               = ini.getDouble("Fuser.Heuristics", "score_hcxh", this->fuserHeuristics.scoreHcxh);
        this->fuserHeuristics.modelBonus              = ini.getDouble("Fuser.Heuristics", "model_bonus", this->fuserHeuristics.modelBonus);

        // Fuser.AI
        this->fuserAI.alive                     = ini.getBool("Fuser.AI", "alive", this->fuserAI.alive);

        // General
        this->general.rl                 = ini.getBool("General", "rl", this->general.rl);
        this->general.windowSize        = ini.getInt("General", "window_size", this->general.windowSize);
        this->general.topKLevels       = ini.getInt("General", "top_k_levels", this->general.topKLevels);
        this->general.sigDim            = ini.getInt("General", "sig_dim", this->general.sigDim);
        this->general.gateFeatDim      = ini.getInt("General", "gate_feat_dim", this->general.gateFeatDim);
        this->general.maxQubits         = ini.getInt("General", "max_qubits", this->general.maxQubits);

        this->general.gamma              = ini.getDouble("General", "gamma", this->general.gamma);
        this->general.lam                = ini.getDouble("General", "lam", this->general.lam);
        this->general.lr                 = ini.getDouble("General", "lr", this->general.lr);
        this->general.clipEps           = ini.getDouble("General", "clip_eps", this->general.clipEps);
        this->general.entCoef           = ini.getDouble("General", "ent_coef", this->general.entCoef);
        this->general.vfCoef            = ini.getDouble("General", "vf_coef", this->general.vfCoef);
        this->general.updateEpochs      = ini.getInt("General", "update_epochs", this->general.updateEpochs);
        this->general.device             = ini.getString("General", "device", this->general.device);
        this->general.useCppReward     = ini.getBool("General", "use_cpp_reward", this->general.useCppReward);
        this->general.cppRewardProb    = ini.getDouble("General", "cpp_reward_prob", this->general.cppRewardProb);
        this->general.cppRewardAlpha   = ini.getDouble("General", "cpp_reward_alpha", this->general.cppRewardAlpha);
    });
}

void Parameter::print() const {
    cout << "GUI:\n  enabled: " << (this->gui.enabled ? "true" : "false") << endl;
    cout << "Parallelism:\n  fiber: " << this->parallelism.fiber
              << "\n  GPU: " << this->parallelism.GPU << endl;
    cout << "Scheduler.Heuristics:\n  alive: " << (this->schedulerHeuristics.alive ? "true" : "false")
              << "\n  costDiag: " << this->schedulerHeuristics.costDiag
              << "\n  costAnti: " << this->schedulerHeuristics.costAnti
              << "\n  costPerm: " << this->schedulerHeuristics.costPerm
              << "\n  costGeneral: " << this->schedulerHeuristics.costGeneral << endl;
    cout << "General:\n  device: " << this->general.device
              << "\n  lr: " << this->general.lr << endl;
}