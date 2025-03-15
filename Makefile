check:
	ruff format .
	ruff check --fix .
	pyright clusfl/

run:
	python -m examples.fed_kmedian