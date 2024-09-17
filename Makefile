.PHONY: lint format test clean runtests runtrain

lint:
	find . -name "*.py" | xargs pylint

format:
	black .

test:
	pytest -m unittest discover -s tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage \
		trained_models/ logs/ test_results/

runtrain:
	python train.py

runtests:
	python test.py
