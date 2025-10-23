#!/bin/bash

set -e

CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -c, --clean    Clean build directory before building"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0             # Normal build"
            echo "  $0 -c          # Clean build"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# クリーンビルドの場合はbuildディレクトリを削除
if [[ "$CLEAN_BUILD" = true ]]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# ビルドディレクトリを作成
mkdir -p build

# ビルドディレクトリに移動
cd build

# CMake実行
echo "Running cmake..."
cmake ..

# Make実行
echo "Building..."
make

echo "Build completed successfully!"