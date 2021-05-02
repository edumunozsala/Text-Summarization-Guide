FROM python:3.6

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip3 install numpy==1.16.2 tensorflow==2.3 flask gevent gunicorn pandas dill
      
        
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Copy the directory 'script' which contains our script file to the /opt/program directory of the image.
COPY Files /opt/program
WORKDIR /opt/program