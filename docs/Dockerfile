FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3-pip libsm6 libxext6 libxrender-dev

ADD . /facecropper
RUN pip3 install /facecropper/