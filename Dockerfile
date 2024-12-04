FROM python:3.10.6-slim


COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt

COPY wattsquad wattsquad
RUN pip install .

COPY rnn_consumption.h5 rnn_consumption.h5
COPY rnn_solar.h5 rnn_solar.h5

COPY raw_data raw_data

# RUN CONTAINER LOCALLY
# CMD uvicorn wattsquad.api.fast:app --host 0.0.0.0

# RUN CONTAINER DEPLOYED
CMD uvicorn wattsquad.api.fast:app --host 0.0.0.0 --port $PORT
