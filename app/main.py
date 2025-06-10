from fastapi import FastAPI
from redis import Redis
from rq import Queue
import time

app = FastAPI()

# Connexion à Redis
redis_conn = Redis(host="redis", port=6379)
queue = Queue("default", connection=redis_conn)

# Dummy job
def slow_job(duration):
    print(f"⏳ Job lancé pour {duration} sec...")
    time.sleep(duration)
    print("✅ Job terminé !")
    return f"Job de {duration} sec terminé"

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur CloudVision 🚀"}

@app.post("/process/")
def process(duration: int = 5):
    job = queue.enqueue(slow_job, duration)
    return {
        "message": "Tâche en file d'attente",
        "job_id": job.get_id(),
        "status": job.get_status()
    }
