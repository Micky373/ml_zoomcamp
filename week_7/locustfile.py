from locust import task,between,HttpUser

sample = {"seniority": 10,
 "home": "owner",
 "time": 36,
 "age": 36,
 "marital": "married",
 "records": "no",
 "job": "freelance",
 "expenses": 75,
 "income": 0.0,
 "assets": 10000.0,
 "debt": 0.0,
 "amount": 1000,
 "price": 1400}



class CreditRiskTestUser(HttpUser):
    
    @task
    def classify(self):
        self.client.post('/classify',json=sample)

    wait_time = between(0.01,2)