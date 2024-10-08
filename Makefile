lint: FORCE
		flake8
		black --check .
		isort --check .

format: FORCE
		isort .
		black .
		flake8 .

test: lint FORCE
		pytest -v tests/unit_test/

FORCE: