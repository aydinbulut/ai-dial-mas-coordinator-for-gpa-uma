.PHONY: reset

reset:
	docker compose down -v --remove-orphans
	rm -rf core-data core-logs
	docker compose up -d --build