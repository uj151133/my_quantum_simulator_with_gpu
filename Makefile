# clean
.PHONY: clean
clean:
	rm -f *.cpython-310-darwin.so

.PHONY: install
install:
	pip install -r requirements.txt
	ifeq ($(shell uname), Darwin)
		brew update
		brew install libomp yaml-cpp gmp gsl eigen xsimd

	else ifeq ($(shell uname), Linux)
		if [ -f /etc/fedora-release ]; then \
            sudo dnf install -y libomp yaml-cpp gmp-devel gsl-devel; \
        else \
            sudo apt-get update; \
            sudo apt-get install -y libomp-dev libyaml-cpp-dev libgmp-dev libgsl-dev; \
        fi

.PHONY: setup
setup:
	chmod +x ./bin/bash/convert_to_csv.zsh
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

.PHONY: run
run:
	python main.py
