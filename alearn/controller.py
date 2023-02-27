import streamlit as st
# from .utils import check_column_types, check_alearn_loop, next_alearn_step, evaluate_model
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier


class AlearnController:

    def __init__(self, session):
        self.alearn_loop = None
        self.annotator = None
        self.mtype = session['model_type']
        self.dtype = 'TEXT'
        self.col_status = session['col_status']
        self.col_type = session['col_type']
        self.labels = session['labels']
        self.df = session['df']
        self.seeds = None

    def add_seed(self):
        if st.session_state['new_seed'] not in self.seeds:
            self.seeds[st.session_state['new_seed']] = st.session_state['new_seed_label']
        st.session_state['new_seed'] = ''

    def delete_seed(self, seed):
        del self.seeds[seed]

    def next_alearn_step(self):
        self.alearn_loop['step'] += 1
        if self.alearn_loop['step'] >= 6:
            self.alearn_loop['step'] = 3
            st.experimental_rerun()

    def assign_annotation(self, ix, label):

        if 'annotations' not in self.annotator:
            self.annotator['annotations'] = {}
        if ix not in self.annotator['annotations']:
            self.annotator['annotations'][ix] = None
        self.annotator['annotations'][ix] = label
        self.annotator['pos'] += 1

    def annotate(self, df, ixs):

        if not self.annotator:
            self.annotator = {}

        if 'pos' not in self.annotator:
            self.annotator['pos'] = 0

        if self.annotator['pos'] >= len(ixs):
            self.split_annotations()
            st.button('Proceed to next step', on_click=self.next_alearn_step)

        else:

            self.annotate_visual(df, ixs)

    def create_initial_learner(self):

        my_bar = st.progress((0 / 2), text="Creating train features")

        train_features, labels = self.get_features(
            self.alearn_loop['data_train'],
            labels=True)

        my_bar.progress((2 / 2), text="Creating learner")

        self.alearn_loop['learner'] = ActiveLearner(
            estimator=RandomForestClassifier(), # TODO take this from config
            query_strategy=self.sampling,
            X_training=train_features, y_training=labels
        )

        my_bar.progress((3 / 3), text="Done")

        self.next_alearn_step()

    def split_annotations(self):
        raise NotImplementedError

    def annotate_visual(self, df, ixs):
        raise NotImplementedError

    def check_run(self):
        raise NotImplementedError

    def check_alearn_loop(self):
        raise NotImplementedError

    def create_seeds(self):
        raise NotImplementedError

    def annotate_seeds(self):
        raise NotImplementedError

    # def create_initial_learner(self):
    #     raise NotImplementedError

    def display_learner_metrics(self):
        raise NotImplementedError

    def annotate_new_samples(self):
        raise NotImplementedError

    def retrain_model(self):
        raise NotImplementedError

    def get_features(self, data, labels=False):
        raise NotImplementedError

    @staticmethod
    def sampling(*args, **kwargs):
        raise NotImplementedError

    def _run(self):
        # st.info(self.labels)
        
        self.check_alearn_loop()

        if self.alearn_loop['step'] == 0:
            self.create_seeds()

            st.info('Create initial seeds')

        if self.alearn_loop['step'] == 1:
            self.annotate_seeds()

            st.info('Annotate seeds')

        if self.alearn_loop['step'] == 2:
            self.create_initial_learner()

            st.info('Train initial learner')

        if self.alearn_loop['step'] == 3:
            self.display_learner_metrics()

            st.info('Metrics')

        if self.alearn_loop['step'] == 4:
            self.annotate_new_samples()

            st.info('More annotations samples')

        if self.alearn_loop['step'] == 5:
            self.retrain_model()

            st.info('Re-train and test')

    def run(self):
        status = self.check_run()
        if not status['error']:
            self._run()
        else:
            st.error(status['message'])

