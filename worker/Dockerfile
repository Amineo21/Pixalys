FROM python:3.11-slim

WORKDIR /worker

COPY main.py .

RUN pip install redis rq

CMD ["python", "main.py"]
