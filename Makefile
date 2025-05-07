_ruff_lint:
	uv run ruff check

_ruff_isort:
	uv run ruff check --select I --fix

_ruff_fmt:
	uv run ruff format

_mypy:
	uv run mypy ./

lint:
	make -j _ruff_lint _mypy

fmt:
	make -j _ruff_isort _ruff_fmt 