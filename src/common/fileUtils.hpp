#ifndef FILEUTILS_HPP
#define FILEUTILS_HPP
#include <filesystem>
#include <string>
#include <fstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

using namespace std;

namespace fileUtils {
    // .env の PROJECT_ROOT を使用（未設定なら current_path）
    extern filesystem::path PROJECT_ROOT;

    // 参照したいグローバル変数
    extern filesystem::path CONFIG_PATH;   // PROJECT_ROOT/paths.json の "config_yaml"
    extern filesystem::path SETTING_PATH;  // PROJECT_ROOT/paths.json の "setting_ini"

    // 初期化（起動時に一度呼ぶ）
    void resolveFilePath();
}

#endif