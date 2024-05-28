SRC = gate.cpp myadd.cpp main.cpp  # ソースファイル
OBJ = $(SRC:.cpp=.o) # オブジェクトファイル

CXX = g++ # コンパイラ

# 手動で Python と pybind11 のパスを指定
PYTHON_INCLUDE_DIR = /Users/mitsuishikaito/anaconda3/include/python3.10
PYTHON_LIB_DIR = /Users/mitsuishikaito/anaconda3/lib
PYBIND11_INCLUDE_DIR = /Users/mitsuishikaito/anaconda3/lib/python3.10/site-packages/pybind11/include

# インクルードとライブラリ用の指定されたパスを使用
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE_DIR) -I$(PYBIND11_INCLUDE_DIR) # コンパイルオプション (フラグ)
LDFLAGS = -L$(PYTHON_LIB_DIR) -lpython3.10

main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o main $(OBJ) $(LDFLAGS) # 実行ファイルを生成するコマンド

# オブジェクトファイルの関係性
myadd.o: myadd.cpp myadd.hpp
gate.o: gate.cpp gate.hpp
main.o: main.cpp myadd.hpp 

# clean
.PHONY: clean
clean:
	rm -f $(OBJ) main

.PHONY: install
install:
	pip install -r requirements.txt
