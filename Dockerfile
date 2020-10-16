FROM python:3.6-slim

ADD ./ /fl-client
WORKDIR /fl-client
RUN pip install -r requirements.txt

CMD python -u main.py