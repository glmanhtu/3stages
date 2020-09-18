FROM python:3.7-alpine
WORKDIR /code
ADD . .
RUN pip install -r ./requirements.txt

CMD [ "python", "./demo_cnn_lstm.py --webcam" ]