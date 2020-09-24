FROM nvidia/cuda:10.2-base

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

WORKDIR /code
ADD . .
RUN pip3 install -r ./requirements.txt

CMD [ "python", "./demo_cnn_lstm.py" ]