.PHONY: install
install:
	pip install pip-tools
	pip-sync requirements.txt
	pip install .