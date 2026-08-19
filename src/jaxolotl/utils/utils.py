from joblib import Memory

from jaxolotl import CACHE_DIR

memory = Memory(CACHE_DIR, verbose=0)
