.PHONY: lint format test clean train eval pyreqs fullreqs package package-all $(MODEL_DIRS)

lint:
	find . -name "*.py" | xargs pylint --rcfile=.pylintrc

format:
	black .

test:
	pytest -m unittest discover -s tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage \
		trained_models/ packaged_models/ logs/ test_results/

train:
	python train.py

eval:
	python test.py

pyreqs:
	pipdeptree --freeze --warn silence | grep -E '^[a-zA-Z0-9\-]+' > requirements.txt

fullreqs:
	pip freeze > requirements.txt

MODEL_DIRS := $(shell find trained_models -type d -maxdepth 1 -mindepth 1 -exec basename {} \;)
package-all: $(addprefix package-, $(MODEL_DIRS))

package-%:
	python package_model_worker.py $*
