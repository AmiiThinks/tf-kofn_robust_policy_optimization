PIP :=pip
PYTHON :=python
NAME :=k_of_n_mdp_policy_opt
LIB_NAME :=k_of_n_mdp_policy_opt

.PHONY: default
default: install

.PHONY: test
test: setup.py
	$(PYTHON) setup.py -q test --addopts '-x $(ARGS)'

.PHONY: test-cov
test-cov: ARGS=--cov $(LIB_NAME) --cov-report term:skip-covered
test-cov: test

.PHONY: install
install: requirements.txt
	$(PIP) install -r requirements.txt $(ARGS)

.PHONY: clean
clean:
	-find . -name ".coverage" -delete
	-find . -name ".DS_Store" -delete
	-find . -name "*.pyc" -delete
	-find . -name "__pycache__" -delete
	-find $(LIB_NAME) -name "*.so" -delete
	-rm -rf .cache *.egg .eggs *.egg-info build 2> /dev/null

.PHONY: clean-tmp
clean-tmp:
	-rm -rf tmp

.PHONY: build
build: .build

.build: setup.py $(LIB_NAME)/__init__.py
	$(PYTHON) setup.py sdist bdist_wheel
	touch $@

.PHONY: release
release: .build
	twine upload dist/*

.PHONY: release-test
release-test: .build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
