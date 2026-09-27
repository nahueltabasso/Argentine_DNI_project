FROM python:3.12.13-slim

ENV DIRECTORY=/opt/project/
ENV PYTHONPATH=/opt/project/
ENV DEBIAN_FRONTEND="noninteractive"

RUN mkdir -p ${DIRECTORY}
RUN mkdir -p ${DIRECTORY}/output/

WORKDIR ${DIRECTORY}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl wget git vim iputils-ping gcc libpq-dev locales  && \
    apt-get install -y libqt5gui5 && \
    ln -f -s /usr/bin/python3 /usr/bin/python && \ 
    echo "es_AR.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    locale -a && \
    export LC_ALL="es_AR.utf8" && \
    export LC_CTYPE="es_AR.utf8" && \
    locale -a

COPY app/ ${DIRECTORY}/app
COPY weights/yolo/yolo26s-obb-DNI-det.pt ${DIRECTORY}/weights/yolo/
COPY weights/yolo/yolo26s_ID_elements_det.pt ${DIRECTORY}/weights/yolo/
COPY Dockerfile ${DIRECTORY}/Dockerfile
COPY requirements.txt ${DIRECTORY}/requirements.txt

RUN pip install --no-cache-dir -r ${DIRECTORY}/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]