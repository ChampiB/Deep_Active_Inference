import gc
from ray.thirdparty_files import psutil


def auto_garbage_collect(pct=80.0):
    """
    This code is a proposed solution to memory buildup in ray from https://stackoverflow.com/questions/55749394/how-to-fix-the-constantly-growing-memory-usage-of-ray
    Call the garbage collection if memory used is greater than 80% of total available memory to deal with Ray not freeing up used memory.
    :param pct: Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return
