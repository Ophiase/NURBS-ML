demo:
	@echo "Basic demo"
	python3 -m main --demonstration basic
	@echo "Interpolation demo"
	python3 -m main --demonstration interpolation

tests:
	python3 -m pytest tests/

.PHONY: demo tests