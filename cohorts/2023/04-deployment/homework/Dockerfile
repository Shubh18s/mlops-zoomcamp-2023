FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "./" ]

RUN mkdir -p ./output/yellow

ENV year=2022
ENV month=4

CMD [ "python3", "starter.py", "2022", "4"]