# clean
.PHONY: clean
clean:
	rm -f $(OBJ) main

.PHONY: install
install:
	pip install -r requirements.txt
