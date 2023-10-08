import streamlit as st
from .base_handler import BaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer


class TextHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super(TextHandler, self).__init__(*args, **kwargs)

    def new_seed_row(self):
        col1, col2 = st.columns((2, 8))
        with col1:
            st.text(self.col_name)
        with col2:
            st.text_input(label='a', label_visibility='collapsed', key=f'{self.col_name}_new_seed')

    def add_seed(self):
        # if st.session_state[f'{self.col_name}_new_seed'] not in self.seeds:
        self.seeds.append(st.session_state[f'{self.col_name}_new_seed'])
        st.session_state[f'{self.col_name}_new_seed'] = ''

    def delete_seed(self, seed_ix):
        del self.seeds[seed_ix]

    def seed_exists(self):
        return st.session_state[f'{self.col_name}_new_seed'] in self.seeds

    def init_vectorizer(self, df):
        self.vectorizer = TfidfVectorizer(max_features=1000)

        self.vectorizer.fit(self.get_data(df))

    def vectorize_seeds(self):
        seed_vectors = []
        for seed in self.seeds:
            seed_vectors.append(self.vectorizer.transform([seed]).todense())
        self.seeds_vectors = seed_vectors

    def vectorize_pool(self, df):
        self.pool_vectors = self.vectorizer.transform(self.get_data(df)).todense()

    def vectorize(self, df):
        return self.vectorizer.transform(self.get_data(df)).toarray()

    def get_data(self, df):
        # return df[df[self.col_name].fillna(' ').astype(str).str.strip().map(len) != 0][self.col_name].tolist()
        return df[self.col_name].fillna(' ').tolist()

    def display_sample(self, df, ix):
        sample = df.iloc[ix][self.col_name]
        st.subheader(self.col_name)
        st.markdown(sample)