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
                data_[section][key] = val;
            }
            return true;
        }

        string getString(const string& section, const string& key, const string& def) const {
            auto sit = data_.find(section);
            if (sit == data_.end()) return def;
            auto kit = sit->second.find(key);
            if (kit == sit->second.end()) return def;
            return kit->second;
        }

        bool getBool(const string& section, const string& key, bool def) const {
            auto s = toLower(getString(section, key, def ? "true" : "false"));
            if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
            if (s == "false" || s == "0" || s == "no" || s == "off") return false;
            return def;
        }

        int getInt(const string& section, const string& key, int def) const {
            auto s = getString(section, key, to_string(def));
            try { return stoi(s); } catch (...) { return def; }
        }

        double getDouble(const string& section, const string& key, double def) const {
            auto s = getString(section, key, to_string(def));
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
    loadFromFile(fileUtils::CONFIG_PATH.string(), fileUtils::SETTING_PATH.string());
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

            if (config["process"]) {
                // this->process.concurrency = config["process"]["concurrency"].as<int>();
                this->process.parallel = config["process"]["parallel"].as<bool>();
                this->process.parallelism = config["process"]["parallelism"].as<int>();
                this->process.GPU = config["process"]["GPU"].as<int>();
            }

            if (config["table"]) {
                this->table.size = config["table"]["size"].as<int>();
            }

            if (config["cache"]) {
                this->cache.alive = config["cache"]["alive"].as<bool>();
                this->cache.size = config["cache"]["size"].as<int>();
                this->cache.TLSSize = config["cache"]["TLSSize"].as<int>();
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
        schedulerHeuristics.alive        = ini.getBool("Scheduler.Heuristics", "alive", schedulerHeuristics.alive);
        schedulerHeuristics.costDiag    = ini.getDouble("Scheduler.Heuristics", "cost_diag", schedulerHeuristics.costDiag);
        schedulerHeuristics.costAnti    = ini.getDouble("Scheduler.Heuristics", "cost_anti", schedulerHeuristics.costAnti);
        schedulerHeuristics.costPerm    = ini.getDouble("Scheduler.Heuristics", "cost_perm", schedulerHeuristics.costPerm);
        schedulerHeuristics.costGeneral = ini.getDouble("Scheduler.Heuristics", "cost_general", schedulerHeuristics.costGeneral);

        // Scheduler.AI
        schedulerAI.alive                = ini.getBool("Scheduler.AI", "alive", schedulerAI.alive);
        // Fuser.Heuristics
        fuserHeuristics.alive                    = ini.getBool("Fuser.Heuristics", "alive", fuserHeuristics.alive);
        fuserHeuristics.scoreDiagPerGate      = ini.getDouble("Fuser.Heuristics", "score_diag_per_gate", fuserHeuristics.scoreDiagPerGate);
        fuserHeuristics.scoreSameAxisPerGate = ini.getDouble("Fuser.Heuristics", "score_same_axis_per_gate", fuserHeuristics.scoreSameAxisPerGate);
        fuserHeuristics.scorePhaseGadget       = ini.getDouble("Fuser.Heuristics", "score_phase_gadget", fuserHeuristics.scorePhaseGadget);
        fuserHeuristics.scoreHcxh               = ini.getDouble("Fuser.Heuristics", "score_hcxh", fuserHeuristics.scoreHcxh);
        fuserHeuristics.modelBonus              = ini.getDouble("Fuser.Heuristics", "model_bonus", fuserHeuristics.modelBonus);

        // Fuser.AI
        fuserAI.alive                     = ini.getBool("Fuser.AI", "alive", fuserAI.alive);

        // General
        general.rl                 = ini.getBool("General", "rl", general.rl);
        general.windowSize        = ini.getInt("General", "window_size", general.windowSize);
        general.topKLevels       = ini.getInt("General", "top_k_levels", general.topKLevels);
        general.sigDim            = ini.getInt("General", "sig_dim", general.sigDim);
        general.gateFeatDim      = ini.getInt("General", "gate_feat_dim", general.gateFeatDim);
        general.maxQubits         = ini.getInt("General", "max_qubits", general.maxQubits);

        general.gamma              = ini.getDouble("General", "gamma", general.gamma);
        general.lam                = ini.getDouble("General", "lam", general.lam);
        general.lr                 = ini.getDouble("General", "lr", general.lr);
        general.clipEps           = ini.getDouble("General", "clip_eps", general.clipEps);
        general.entCoef           = ini.getDouble("General", "ent_coef", general.entCoef);
        general.vfCoef            = ini.getDouble("General", "vf_coef", general.vfCoef);
        general.updateEpochs      = ini.getInt("General", "update_epochs", general.updateEpochs);
        general.device             = ini.getString("General", "device", general.device);
        general.useCppReward     = ini.getBool("General", "use_cpp_reward", general.useCppReward);
        general.cppRewardProb    = ini.getDouble("General", "cpp_reward_prob", general.cppRewardProb);
        general.cppRewardAlpha   = ini.getDouble("General", "cpp_reward_alpha", general.cppRewardAlpha);
    });
}

void Parameter::print() const {
    cout << "GUI:\n  enabled: " << (gui.enabled ? "true" : "false") << endl;
    cout << "Process:\n  concurrency: " << 0 /*process.concurrency*/
              << "\n  parallelism: " << process.parallel << endl;
    cout << "Scheduler.Heuristics:\n  alive: " << (schedulerHeuristics.alive ? "true" : "false")
              << "\n  costDiag: " << schedulerHeuristics.costDiag
              << "\n  costAnti: " << schedulerHeuristics.costAnti
              << "\n  costPerm: " << schedulerHeuristics.costPerm
              << "\n  costGeneral: " << schedulerHeuristics.costGeneral << endl;
    cout << "General:\n  device: " << general.device
              << "\n  lr: " << general.lr << endl;
}