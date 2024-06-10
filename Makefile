# clean
.PHONY: clean
clean:
	rm -f $(OBJ) main

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: setup
setup:
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
