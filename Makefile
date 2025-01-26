demo:
	@echo "Synthetic demo"
	python3 -m main --demo synthetic &
	@echo "Surface demo"
	python3 -m main --demo surface &

tests:
	python3 -m pytest tests/

tests_verbose:
	python3 -m pytest tests/ -v

.PHONY: demo tests tests_verbose