FROM python:3.11.1

WORKDIR /app

RUN python -m ensurepip
RUN python -m pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .

ENV FLASK_APP main.py

ENV PORT 8080
EXPOSE 8080

CMD ["python", "main.py"]