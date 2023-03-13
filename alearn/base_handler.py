class BaseHandler:

    def __init__(self, dtype, col_name):
        self.dtype = dtype
        self.col_name = col_name

    def new_seed_row(self):
        raise NotImplementedError

    def add_seed(self):
        raise NotImplementedError
