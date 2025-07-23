#!/bin/bash

# QMDD Simulator GUI Communication Setup Script

echo "QMDD Simulator GUI Communication Setup"
echo "======================================="

# 必要なライブラリのインストール確認
echo "Checking required libraries..."

# nlohmann/json for C++
if ! brew list nlohmann-json &>/dev/null; then
    echo "Installing nlohmann-json..."
    brew install nlohmann-json
else
    echo "✓ nlohmann-json is already installed"
fi

# Build the C++ simulator
echo "Building QMDD Simulator..."
cd /Users/mitsuishikaito/my_quantum_simulator_with_gpu
mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

if [ $? -eq 0 ]; then
    echo "✓ C++ Simulator built successfully"
else
    echo "✗ Failed to build C++ Simulator"
    exit 1
fi

# Build the GUI
echo "Building GUI..."
cd /Users/mitsuishikaito/my_quantum_simulator_with_gpu/GUI/GUI
dotnet build

if [ $? -eq 0 ]; then
    echo "✓ GUI built successfully"
else
    echo "✗ Failed to build GUI"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "To run the system:"
echo "1. Start the QMDD Simulator in Shared Memory IPC mode:"
echo "   cd /Users/mitsuishikaito/my_quantum_simulator_with_gpu/build"
echo "   ./qmdd_sim -s"
echo ""
echo "2. In another terminal, start the GUI:"
echo "   cd /Users/mitsuishikaito/my_quantum_simulator_with_gpu/GUI/GUI/GUI.Web"
echo "   dotnet run"
echo ""
echo "3. Open your browser and navigate to the GUI URL (usually http://localhost:5000)"
echo "4. Go to the Composer page and use the 'Run Simulation' button"
