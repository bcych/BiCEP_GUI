FROM python:3.7.13

ADD . .

RUN pip install -r requirements.txt

EXPOSE 8888

ENV NAME BiCEP

CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "-port=8888", "--no-browser","--allow-root"]