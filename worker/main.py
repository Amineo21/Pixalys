import time
from redis import Redis
from rq import Worker, Queue, Connection

listen = ['default']

redis_conn = Redis(host='redis', port=6379)

if __name__ == '__main__':
    print("🛠️  Starting worker with RQ...")
    with Connection(redis_conn):
        worker = Worker(map(Queue, listen))
        worker.work()
