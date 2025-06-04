#!/bin/bash

# =========================
# GraalVM Native Image 共有ライブラリビルドスクリプト
# =========================

# ---- 設定 ----
CAFFEINE_JAR="caffeine-3.2.0.jar"
OUTPUT_NAME="liboperation-cache"

# プラットフォームによるクラスパス区切り
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then
    CP_DELIM=";"
else
    CP_DELIM=":"
fi

CLASSPATH="$CAFFEINE_JAR${CP_DELIM}."

echo "🔄 GraalVM Native Image 共有ライブラリのビルドを開始..."

# 必要なファイルのチェック
if [[ ! -f "$CAFFEINE_JAR" ]]; then
    echo "❌ $CAFFEINE_JAR が見つかりません。"
    exit 1
fi

for file in OperationCache.java OperationResult.java NativeImageBridge.java; do
    if [[ ! -f "$file" ]]; then
        echo "❌ $file が見つかりません。"
        exit 1
    fi
done

# 1. Javaファイルのコンパイル
echo "📝 Javaファイルをコンパイル中..."
javac -cp "$CLASSPATH" OperationCache.java OperationResult.java NativeImageBridge.java

if [[ $? -ne 0 ]]; then
    echo "❌ Javaコンパイルに失敗しました。"
    exit 1
fi

# 2. Native Imageビルド（共有ライブラリ）
echo "🔧 Native Image 共有ライブラリをビルド中..."
native-image \
    --shared \
    --no-fallback \
    -cp "$CLASSPATH" \
    -H:Name="$OUTPUT_NAME" \
    -H:+UnlockExperimentalVMOptions \
    -H:+ReportExceptionStackTraces \
    --enable-monitoring \

if [[ $? -eq 0 ]]; then
    echo "✅ 共有ライブラリのビルドが完了しました！"
    echo "📁 出力ファイル："
    ls -la liboperation-cache.*
else
    echo "❌ Native Imageビルドに失敗しました。"
    exit 1
fi

# クリーンアップ
echo "🧹 一時ファイルをクリーンアップ中..."
rm -f *.class

echo "🎉 ビルド完了！"