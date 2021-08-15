FROM python:slim
RUN adduser --gecos "" --disabled-password ml-user
USER ml-user
ENV PATH=$PATH:/home/ml-user/.local/bin
COPY . /home/ml-user/
RUN pip3 install -r /home/ml-user/requirements.txt
EXPOSE 5000
CMD python /home/ml-user/server.py
