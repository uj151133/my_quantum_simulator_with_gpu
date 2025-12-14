#include "fileUtils.hpp"

namespace fileUtils {

// グローバル実体
filesystem::path PROJECT_ROOT;
filesystem::path CONFIG_PATH;
filesystem::path SETTING_PATH;

void resolveFilePath() {
    // PROJECT_ROOT
    const char* env = getenv("PROJECT_ROOT");
    if (env && *env) {
        PROJECT_ROOT = filesystem::path(env);
    } else {
        PROJECT_ROOT = filesystem::current_path();
    }

    PROJECT_ROOT = filesystem::weakly_canonical(PROJECT_ROOT);

    // PROJECT_ROOT/paths.json を読む
    nlohmann::json j;
    ifstream in(PROJECT_ROOT / "paths.json");
    if (in.is_open()) {
        try {
            in >> j;
        } catch (...) {
            j = nlohmann::json{};
        }
    }

    // 既定値補完
    if (!j.contains("config_yaml")) j["config_yaml"] = "config.yaml";
    if (!j.contains("setting_ini")) j["setting_ini"] = "setting.ini";

    CONFIG_PATH  = filesystem::weakly_canonical(PROJECT_ROOT / j.at("config_yaml").get<string>());
    SETTING_PATH = filesystem::weakly_canonical(PROJECT_ROOT / j.at("setting_ini").get<string>());
}

}