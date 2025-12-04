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
    // グローバルパス初期化
    fileUtils::FilePath();
    loadFromFiles(fileUtils::CONFIG_PATH.string(), fileUtils::SETTING_PATH.string());
}

void Parameter::loadFromFiles(const string& yamlFilepath, const string& iniFilepath) {
    call_once(parameter_load_flag, [this, &yamlFilepath, &iniFilepath]() {
        cout << "Loading " << yamlFilepath << " and " << iniFilepath << " ..." << endl;

        // YAML
        try {
            YAML::Node config = YAML::LoadFile(yamlFilepath);

            if (config["gui"]) {
                this->gui.enabled = config["gui"]["enabled"].as<bool>();
            }

            if (config["process"]) {
                this->process.concurrency = config["process"]["concurrency"].as<int>();
                this->process.parallelism = config["process"]["parallelism"].as<int>();
            }

            if (config["table"]) {
                this->table.size = config["table"]["size"].as<int>();
            }

            if (config["cache"]) {
                this->cache.alive = config["cache"]["alive"].as<bool>();
                this->cache.size = config["cache"]["size"].as<int>();
            }

            if (config["circuit"]) {
                this->circuit.mode = config["circuit"]["mode"].as<string>();
                this->circuit.cancerMax = config["circuit"]["cancerMax"].as<int>();
                this->circuit.shuffle = config["circuit"]["shuffle"].as<bool>();
                this->circuit.verbose = config["circuit"]["verbose"].as<bool>();
                this->circuit.timer = config["circuit"]["timer"].as<bool>();
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
        scheduler_heuristics.alive        = ini.getBool("Scheduler.Heuristics", "alive", scheduler_heuristics.alive);
        scheduler_heuristics.cost_diag    = ini.getDouble("Scheduler.Heuristics", "cost_diag", scheduler_heuristics.cost_diag);
        scheduler_heuristics.cost_anti    = ini.getDouble("Scheduler.Heuristics", "cost_anti", scheduler_heuristics.cost_anti);
        scheduler_heuristics.cost_perm    = ini.getDouble("Scheduler.Heuristics", "cost_perm", scheduler_heuristics.cost_perm);
        scheduler_heuristics.cost_general = ini.getDouble("Scheduler.Heuristics", "cost_general", scheduler_heuristics.cost_general);

        // Scheduler.AI
        scheduler_ai.alive                = ini.getBool("Scheduler.AI", "alive", scheduler_ai.alive);

        // Fuser.Heuristics
        fuser_heuristics.alive                    = ini.getBool("Fuser.Heuristics", "alive", fuser_heuristics.alive);
        fuser_heuristics.score_diag_per_gate      = ini.getDouble("Fuser.Heuristics", "score_diag_per_gate", fuser_heuristics.score_diag_per_gate);
        fuser_heuristics.score_same_axis_per_gate = ini.getDouble("Fuser.Heuristics", "score_same_axis_per_gate", fuser_heuristics.score_same_axis_per_gate);
        fuser_heuristics.score_phase_gadget       = ini.getDouble("Fuser.Heuristics", "score_phase_gadget", fuser_heuristics.score_phase_gadget);
        fuser_heuristics.score_hcxh               = ini.getDouble("Fuser.Heuristics", "score_hcxh", fuser_heuristics.score_hcxh);
        fuser_heuristics.model_bonus              = ini.getDouble("Fuser.Heuristics", "model_bonus", fuser_heuristics.model_bonus);

        // Fuser.AI
        fuser_ai.alive                     = ini.getBool("Fuser.AI", "alive", fuser_ai.alive);

        // General
        general.rl                 = ini.getBool("General", "rl", general.rl);
        general.window_size        = ini.getInt("General", "window_size", general.window_size);
        general.top_k_levels       = ini.getInt("General", "top_k_levels", general.top_k_levels);
        general.sig_dim            = ini.getInt("General", "sig_dim", general.sig_dim);
        general.gate_feat_dim      = ini.getInt("General", "gate_feat_dim", general.gate_feat_dim);
        general.max_qubits         = ini.getInt("General", "max_qubits", general.max_qubits);

        general.gamma              = ini.getDouble("General", "gamma", general.gamma);
        general.lam                = ini.getDouble("General", "lam", general.lam);
        general.lr                 = ini.getDouble("General", "lr", general.lr);
        general.clip_eps           = ini.getDouble("General", "clip_eps", general.clip_eps);
        general.ent_coef           = ini.getDouble("General", "ent_coef", general.ent_coef);
        general.vf_coef            = ini.getDouble("General", "vf_coef", general.vf_coef);
        general.update_epochs      = ini.getInt("General", "update_epochs", general.update_epochs);

        general.device             = ini.getString("General", "device", general.device);

        general.use_cpp_reward     = ini.getBool("General", "use_cpp_reward", general.use_cpp_reward);
        general.cpp_reward_prob    = ini.getDouble("General", "cpp_reward_prob", general.cpp_reward_prob);
        general.cpp_reward_alpha   = ini.getDouble("General", "cpp_reward_alpha", general.cpp_reward_alpha);
    });
}

void Parameter::print() const {
    cout << "GUI:\n  enabled: " << (gui.enabled ? "true" : "false") << endl;
    cout << "Process:\n  concurrency: " << process.concurrency
              << "\n  parallelism: " << process.parallelism << endl;
    cout << "Scheduler.Heuristics:\n  alive: " << (scheduler_heuristics.alive ? "true" : "false")
              << "\n  cost_diag: " << scheduler_heuristics.cost_diag
              << "\n  cost_anti: " << scheduler_heuristics.cost_anti
              << "\n  cost_perm: " << scheduler_heuristics.cost_perm
              << "\n  cost_general: " << scheduler_heuristics.cost_general << endl;
    cout << "General:\n  device: " << general.device
              << "\n  lr: " << general.lr << endl;
}