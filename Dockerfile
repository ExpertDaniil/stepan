# Local validation image; the competition platform supplies its own container.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.runtime.txt ./
RUN pip install --no-cache-dir -r requirements.runtime.txt
COPY solution.py ./
COPY agent ./agent
COPY tools ./tools
COPY recipes ./recipes
COPY prompts ./prompts
COPY contracts/contracts.json contracts/contracts.schema.json ./contracts/
ENTRYPOINT ["python", "solution.py"]
CMD ["/task/task.md"]
