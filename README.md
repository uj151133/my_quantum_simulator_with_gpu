# my_quantum_simulator_wiθ_gpu
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/uj151133/my_quantum_simulator_with_gpu)

A high-performance QMDD (Quantum Multiple-valued Decision Diagrams) based quantum simulator with GPU support and web-based GUI interface.

## Features

- **QMDD-based quantum simulation** with GPU acceleration
- **Web-based GUI** built with Blazor Server for circuit composition
- **IPC communication** between GUI and simulator using Unix domain sockets
- **Multi-platform support** (macOS, Linux)

## GUI-Simulator Communication

The project includes a communication system between the Blazor web GUI and the C++ QMDD simulator:

### Architecture
- **GUI (Blazor Server)**: Provides a web-based interface for quantum circuit composition
- **QMDD Simulator**: C++ backend that performs quantum simulations
- **IPC Layer**: Shared memory-based communication for high-performance circuit data exchange

### Usage

1. **Start the QMDD Simulator in Shared Memory IPC mode:**
   ```bash
   cd build
   ./qmdd_sim -s
   ```

2. **Start the GUI:**
   ```bash
   cd GUI/GUI/GUI.Web
   dotnet run
   ```

3. **Access the GUI** in your web browser (typically `http://localhost:5000`)

4. **Use the Composer page** to build circuits and click "Run Simulation" to execute them on the backend simulator

## Requirements
do this

### Dependencies

#### C++ Dependencies
- **yaml-cpp**: For configuration file parsing
- **nlohmann/json**: For IPC communication data serialization
- **Boost**: For various utilities
- Other dependencies as specified in CMakeLists.txt

#### .NET Dependencies
- **.NET 8.0 or later**: For the Blazor GUI

### yaml
please install and build yaml.
```zsh
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build
cd build
cmake ..
make
sudo make install
```

### nlohmann/json (macOS)
```zsh
brew install nlohmann-json
```

### OpenMP
```zsh
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```


### .NET MAUI
↓download from this link
https://dotnet.microsoft.com/ja-jp/download/dotnet/thank-you/sdk-9.0.101-macos-arm64-installer

### others
```zsh
pip install -r requirements.txt
```



