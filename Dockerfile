FROM python:3.13-slim

WORKDIR /app

# install exact library versions first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project code, config and test sample
COPY src/ src/
COPY configs/ configs/
COPY data/sample/ data/sample/

# test run on data/sample, results are written to /app/results
CMD ["python", "src/main.py", "--config", "configs/config.yaml"]
