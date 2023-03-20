import streamlit as st
from .controller import AlearnController
from .handler_builder import handler_from_settings
from .utils import check_column_types


class MultiController(AlearnController):

    def __init__(self, *args, **kwargs):
        session = args[0]
        self.handlers = self.init_handlers(session)
        print(self.handlers)
        super(MultiController, self).__init__(*args, **kwargs)

    @staticmethod
    def init_handlers(session):
        handlers = []
        for col in session['col_status']:
            if session['col_status'][col]:
                handlers.append(handler_from_settings(session['col_type'][col], col))
        return handlers

    def check_run(self):
        status = {'error': False}
        return status

    def check_alearn_loop(self):
        if not self.alearn_loop:
            self.alearn_loop = {
                'step': 0,
                'seed_annotation': {},
            }

    def add_seed(self):
        # Check if seed exists in every handler
        seed_status = [handler.seed_exists() for handler in self.handlers]

        if False in seed_status:
            for handler in self.handlers:
                handler.add_seed()
            self.seeds += 1
        else:
            st.error("You've entered a seed that already exists. Please add verify.")

    def create_seeds(self):
        if not self.seeds:
            self.seeds = 0

        for ix_seed in range(self.seeds):
            with st.expander(f'Seed {ix_seed + 1}'):
                col1, col2 = st.columns((2, 8))
                for handler in self.handlers:
                    with col1:
                        st.text(handler.col_name)
                    with col2:
                        st.text(handler.seeds[ix_seed])

        with st.expander('New seed', expanded=True):
            for handler in self.handlers:
                handler.new_seed_row()

        st.button('Add seed', on_click=self.add_seed)

        if self.seeds < 2:
            st.warning("At least two seeds are needed.")
        else:
            st.text(self.seeds)
            st.button('Proceed to next step', on_click=self.next_alearn_step)
