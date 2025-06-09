// config.yamlを読み込む関数

#include "monitor.hpp"

string getProcessType() {
    YAML::Node config = YAML::LoadFile("config.yaml");
    return config["process"]["type"].as<string>();
}


// 逐次処理する関数
void sequentialProcessing() {
    for (int i = 0; i < 10; ++i) {
        cout << "逐次処理: " << i << endl;
    }
}

// マルチファイバー処理する関数
void fiberProcessing() {
    boost::fibers::fiber fibers[10];
    for (int i = 0; i < 10; ++i) {
        fibers[i] = boost::fibers::fiber([i]() {
            cout << "マルチファイバー処理: " << i << endl;
        });
    }
    for (int i = 0; i < 10; ++i) {
        fibers[i].join();
    }
}

void simdProcessing() {
    cout << "Eigen SIMD Support: ";
    #ifdef EIGEN_VECTORIZE
        cout << "Enabled" << std::endl;
    #else
        cout << "Disabled" << std::endl;
    #endif
}

void printMemoryUsage() {
    pid_t pid = getpid();
    string command = "ps -o rss= -p " + to_string(pid);

    // Create a pipe to capture the output of the command
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        cerr << "\033[1;31mFailed to run command.\033[0m\n";
        return;
    }

    // Read the output of the command
    char buffer[128];
    string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    // Close the pipe and check for errors
    pclose(pipe);

    // Remove any trailing whitespace from the result
    result.erase(result.find_last_not_of(" \n\r\t") + 1);

    cout << "\033[1;34mMemory usage: " << result << " KB\033[0m" << endl;
}

#ifdef __APPLE__
void printMemoryUsageOnMac() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        std::cout << "Failed to get task info" << std::endl;
        return;
    }
    
    cout << "\033[1;34mMemory usage on mac environment: " << info.resident_size / 1024 << " KB\033[0m\n";
}
#elif defined(__linux__)
void printMemoryUsageOnLinux() {
    // Linuxでのメモリ使用量取得処理
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        cout << "\033[1;34mMemory usage on Linux environment: " << usage.ru_maxrss << " KB\033[0m\n";
    }
}
#endif

void measureExecutionTime(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    // ホスト名を取得
    char hostname[255];
    gethostname(hostname, 255);

    string branchName = "unknown";
    FILE* pipe = popen("git rev-parse --abbrev-ref HEAD 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            branchName = string(buffer);
            // 改行文字を削除
            branchName.erase(branchName.find_last_not_of(" \n\r\t") + 1);
        }
        pclose(pipe);
    }
    
    // 時刻を取得
    time_t now = time(nullptr);
    char timestamp[26];
    ctime_r(&now, timestamp);
    timestamp[24] = '\0';
    
    // コンソール出力
    cout << "\033[1;32mExecution time: " << duration.count() << " ms\033[0m" << endl;
    
    // ファイル出力
    ofstream logFile("record.log", ios::app);
    if (logFile.is_open()) {
        logFile << "[" << timestamp << "] "
                << "Host: " << hostname << " | "
                << "Branch: " << branchName << " | "
                << "Execution time: " << duration.count() << " ms" 
                << endl;
        logFile.close();
    }
}

bool isExecuteGui() {
    YAML::Node config = YAML::LoadFile("config.yaml");
    bool guiEnabled = config["gui"]["enabled"].as<bool>();

    return guiEnabled;
}