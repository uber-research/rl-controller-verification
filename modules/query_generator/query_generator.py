
import numpy as np

class QueryResult: 
    duration: float 
    query: np.ndarray 
    def __init__(self, duration=0.0, query=np.zeros(3)): 
        self.duration = duration 
        self.query = query

    def f_str(self): 
        return f"QueryResult: duration={self.duration}, query={self.query}"
    
    def get(self): 
        return self.duration, self.query 

class QueryArgs: 
    duration_min: float = 0.1 
    durantion_max: float = 1.0 
    query_dim: int = 3
    query_val_min: float = -0.6
    query_val_max: float = 0.6

    def __init__(self, duration_min=0.1, duration_max=1.0, query_dim=3, query_val_min=-0.6, query_val_max=0.6): 
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.query_dim = query_dim
        self.query_val_min = query_val_min
        self.query_val_max = query_val_max

    def f_str(self): 
        return f"QueryArgs (duration_min={duration_min}, durantion_max={duration_max}, query_dim={query_dim}, query_val_min={query_val_min}, query_val_max={query_val_max})"

class QueryGenerator: 
    """ Abstract Class providing the interface for Query Generation 
    """
    def get(self) -> QueryResult: 
        raise NotImplementedError()


class RandomQueryGenerator(QueryGenerator): 
    """ Concrete Class of Query Generation providing 
    """
    def __init__(self, pdf_name: str = 'uniform', seed=0): 
        self.seed = seed 
        self.pdfs = {'uniform': np.random.uniform}
        if pdf_name not in self.pdfs.keys(): 
            raise RuntimeError(f"Unsupported PDF {pdf_name}")
        self.pdf = self.pdfs[pdf_name]

    def get(self, a: QueryArgs = QueryArgs()) -> QueryResult: 
        return QueryResult(duration=self.pdf(a.duration_min, a.duration_max), query=self.pdf(a.query_val_min, a.query_val_max, a.query_dim))


