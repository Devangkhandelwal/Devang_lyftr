.PHONY: up down logs test build restart ps clean

# Build the images and start both services in the background.
up:
	docker compose up --build -d
	@echo ""
	@echo "Backend:  http://localhost:8000/health"
	@echo "Frontend: http://localhost:3000"

# Stop and remove the containers. The app_data volume is kept, so the
# chat history survives.
down:
	docker compose down

# Follow the logs of both services.
logs:
	docker compose logs -f

# Run the backend test suite inside the backend image.
# The tests mock the provider, so no API key and no network are needed.
test:
	docker compose run --rm --no-deps backend python -m pytest -q

# Build the images without starting anything.
build:
	docker compose build

restart:
	docker compose restart

ps:
	docker compose ps

# Remove the containers AND the stored data. This deletes the chat history.
clean:
	docker compose down -v
