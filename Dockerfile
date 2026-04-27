FROM python:3.9-slim
COPY  . /app 
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD python flask_app.py