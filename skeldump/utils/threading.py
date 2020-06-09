import queue
import threading


def collect(inner_func, q, args, kwargs):
    try:
        for item in inner_func(*args, **kwargs):
            q.put(item)
    finally:
        q.put(StopIteration)


def thread_wrap_iter(inner_func, *args, maxsize=0, **kwargs):
    q = queue.Queue(maxsize=maxsize)
    thr = threading.Thread(target=collect, args=(inner_func, q, args, kwargs))
    thr.start()

    while True:
        item = q.get()
        if item is StopIteration:
            thr.join()
            break
        yield item
