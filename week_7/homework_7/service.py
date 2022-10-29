import bentoml

from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("homework",runners=[model_runner])

@svc.api(input=NumpyNdarray(),output=NumpyNdarray())

def result(client):
    prediction = model_runner.predict.run(client)

    return prediction