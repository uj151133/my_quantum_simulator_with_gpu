# clean
.PHONY: clean
clean:
	rm -f *.cpython-310-darwin.so

.PHONY: install
install:
	pip install -r requirements.txt
	ifeq ($(shell uname), Darwin)
		brew update
		brew install libomp yaml-cpp gmp gsl cmake boost
	else ifeq ($(shell uname), Linux)
		if [ -f /etc/fedora-release ]; then \
            sudo dnf install -y libomp yaml-cpp gmp-devel gsl-devel cmake boost-devel; \
        else \
			sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev; \

			git clone https://github.com/pyenv/pyenv.git ~/.pyenv; \
			echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc; \
			echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc; \
			echo 'eval "$(pyenv init -)"' >> ~/.bashrc; \
			source ~/.bashrc; \
			pyenv install 3.12.7; \
			pyenv global 3.12.7; \
			sudo snap install cmake --classic; \
			export PATH=$PATH:/snap/bin; \
            sudo apt-get update; \
            sudo apt-get install -y libomp-dev libyaml-cpp-dev libgmp-dev libgsl-dev cmake libboost-all-dev; \
		fi

.PHONY: setup
setup:
	chmod +x ./bin/bash/convert_to_csv.zsh
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

.PHONY: run
run:
	python main.py
