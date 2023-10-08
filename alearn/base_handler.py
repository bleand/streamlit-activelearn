class BaseHandler:

    def __init__(self, dtype, col_name):
        self.dtype = dtype
        self.col_name = col_name
        self.seeds = []
        self.vectorizer = None
        self.seeds_vectors = None
        self.pool_vectors = None

    def new_seed_row(self):
        raise NotImplementedError

    def add_seed(self):
        raise NotImplementedError

    def delete_seed(self, seed_ix):
        del self.seeds[seed_ix]

    def seed_exists(self):
        raise NotImplementedError

    def init_vectorizer(self, df):
        raise NotImplementedError

    def vectorize_seeds(self):
        raise NotImplementedError

    def vectorize_pool(self, df):
        raise NotImplementedError

    def vectorize(self, df):
        raise NotImplementedError

    def get_data(self, df):
        raise NotImplementedError

    def display_sample(self, df, ix):
        raise NotImplementedError