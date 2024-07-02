# clean
.PHONY: clean
clean:
	rm -f $(OBJ) main

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: setup
setup:
	chmod +x ./bin/bash/convert_to_csv.zsh
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
