FROM python:3.11
COPY . /app_bertopic
WORKDIR /app_bertopic
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=3 --bind 0.0.0.0:$PORT app:app_bertopic