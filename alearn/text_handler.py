import streamlit as st
from .base_handler import BaseHandler


class TextHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        self.seeds = []
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

    def seed_exists(self):
        return st.session_state[f'{self.col_name}_new_seed'] in self.seeds
