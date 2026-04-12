from .endee_db import db

def retrieve(query_embedding):
    results = db.search(query_embedding, k=3)
    return results