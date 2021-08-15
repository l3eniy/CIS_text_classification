FROM python:alpine
RUN adduser -D ml-user
RUN apk add build-base
USER ml-user
ENV PATH=$PATH:/home/ml-user/.local/bin
COPY . /home/ml-user/
RUN pip3 install -r /home/ml-user/requirements.txt
CMD python /home/ml-user/server.py