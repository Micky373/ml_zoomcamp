import bentoml
import numpy as np

from bentoml.io import JSON,NumpyNdarray
from pydantic import BaseModel

class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: int
    assets: int
    debt: int
    amount: int
    price: int

model_ref = bentoml.xgboost.get("credit_risk_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier",runners=[model_runner])

@svc.api(input=JSON(pydantic_model=CreditApplication),output=JSON())
# @svc.api(input=NumpyNdarray(shape=(-1,29),dtype=np.float32,enforce_dtype=True,enforce_shape=True),output=JSON())

async def classify(credit_application):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    result = prediction[0]

    if result > 0.5:
        return {"status":"DECLINED"}
    elif result > 0.25:
        return {"status":f"MAYBE"}
    else :
        return {"status":"APROVED"}
