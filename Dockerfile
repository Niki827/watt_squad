FROM python:3.10.6

COPY wattsquad wattsquad
COPY requirements.txt requirements.txt
COPY setup.py setup.py
#COPY models models need to add the pickle file

RUN pip install -r requirements.txt
RUN pip install -e .

# RUN CONTAINER LOCALLY
CMD uvicorn wattsquad.api.fast:app --host 0.0.0.0

# RUN CONTAINER DEPLOYED
#CMD uvicorn wattsquad.api.fast:app --host 0.0.0.0 --port $PORT
