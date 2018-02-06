FROM tensorflow/tensorflow:1.4.1-gpu

MAINTAINER Ben Navetta <benjamin_navetta@brown.edu>

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential libssl-dev libffi-dev python-dev
RUN apt-get install -y python-scipy python-tk
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
COPY . .
COPY gpyopt_requirements.txt /tmp/
RUN pip install --requirement /tmp/gpyopt_requirements.txt
COPY . /tmp/
COPY . .

CMD ["cd", "/media/data_cifs/cluster_projects/contextual_circuit_bp"]
