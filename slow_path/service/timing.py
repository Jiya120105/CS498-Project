import time, heapq
class TimerStats:
    def __init__(self): self.samples=[]
    def observe_ms(self, t): heapq.heappush(self.samples, t)
    def percentiles(self):
        if not self.samples: return {}
        a = sorted(self.samples); n=len(a)
        p=lambda q: a[min(n-1,int(q*(n-1)))]
        return {"p50":p(0.5), "p95":p(0.95), "count":n}
