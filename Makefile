.PHONY: env-update test test-integration up down logs

# Create or update the local conda env (single source of truth for local deps).
env-update:
	conda env update -n dft-workflow -f environment.yml --prune

# Run the unit test suite from the conda env.
test:
	PYTHONPATH=. conda run -n dft-workflow pytest tests/unit -q

# Run the integration tests that don't require a live docker stack.
test-integration:
	PYTHONPATH=. conda run -n dft-workflow pytest \
		tests/integration/test_real_pyscf.py \
		tests/integration/test_storage_minio.py -q

# Convenience wrappers around the local docker stack.
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100
