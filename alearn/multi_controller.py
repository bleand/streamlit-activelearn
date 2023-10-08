import datetime
import os
from pathlib import Path
import pickle

import scipy
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .controller import AlearnController
from .handler_builder import handler_from_settings
from .utils import check_column_types
from .utils import format_metrics, display_metric_line

import numpy as np
import pandas as pd


class MultiController(AlearnController):

    def __init__(self, *args, **kwargs):
        session = args[0]

        # Create handlers for each type of column
        self.handlers = self.init_handlers(session)
        self.sampling = self.uncertainty_sampling
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

        # TODO add check for empty seeds

        if False in seed_status:
            for handler in self.handlers:
                handler.add_seed()
            self.seeds += 1
        else:
            st.error("You've entered a seed that already exists. Please add verify.")

    def delete_seed(self, seed_ix):
        for handler in self.handlers:
            handler.delete_seed(seed_ix)
        self.seeds -= 1

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
                st.button("🗑", key=f'delete_{ix_seed}', on_click=self.delete_seed, args=(ix_seed,))

        with st.expander('New seed', expanded=True):
            for handler in self.handlers:
                handler.new_seed_row()

        st.button('Add seed', on_click=self.add_seed)

        if self.seeds < 2:
            st.warning("At least two seeds are needed.")
        else:
            st.button('Proceed to next step', on_click=self.next_alearn_step)

    def annotate_seeds(self):
        print([not handler.vectorizer for handler in self.handlers])
        if any([not handler.vectorizer for handler in self.handlers]):
            st.info('We will we will ANNOTATE')

            steps = 6

            my_bar = st.progress((0 / steps), text="Creating vectorizers")

            [handler.init_vectorizer(self.df) for handler in self.handlers]

            my_bar.progress((1 / steps), text="Vectorizing seeds")
            [handler.vectorize_seeds() for handler in self.handlers]
            # seed_vectors = np.concatenate([handler.vectorize_seeds() for handler in self.handlers], axis=1)
            # print(seed_vectors.shape)

            my_bar.progress((2 / steps), text="Vectorizing pool")
            [handler.vectorize_pool(self.df) for handler in self.handlers]

            my_bar.progress((3 / steps), text="Calculating similarity")
            # Calculate cosine distance (opposite to cosine similarity)
            distances = []
            for handler in self.handlers:
                handler_distances = []
                for vector in handler.seeds_vectors:
                    distance = scipy.spatial.distance.cdist(vector,
                                                      handler.pool_vectors,
                                                      metric='cosine')
                    handler_distances.append(distance)
                distances.append(handler_distances)

            # distances = [col1, col2, ...]
            # col1 = [seed1, seed2, ...]
            # seed1 = array([d1, d2, d3... dN])
            # condensed_distances = [avg_seed1, avg_seed2, ... ]
            # avg_seed1 = array([avg_d1, avg_d2, ... ]

            condensed_distances = np.mean(distances, axis=0)

            my_bar.progress((4 / steps), text="Sorting distances")
            # Sort in ascending order and pick the first 10 of each
            num_samples = st.session_state['cfg']['NUM_SAMPLES']
            closest = []
            for dist in condensed_distances:
                closest.append(np.argsort(dist)[:, :num_samples].flatten())

            my_bar.progress((5 / steps), text="Collecting idxs")

            self.alearn_loop['seed_annotation']['idxs'] = []
            for close in closest:
                self.alearn_loop['seed_annotation']['idxs'].extend(list(close))

            my_bar.progress((6 / steps), text="Done")

            self.annotate(self.df, self.alearn_loop['seed_annotation']['idxs'])

        else:
            st.info("We are we are ANNOTATING")
            self.annotate(self.df, self.alearn_loop['seed_annotation']['idxs'])

    def annotate_visual(self, df, ixs):

        my_bar = st.progress((self.annotator['pos'] / len(ixs)))

        st.divider()
        cols = st.columns(tuple(1 for _ in self.labels))
        for ix, col in enumerate(cols):
            with col:
                st.button(self.labels[ix], key=f"apply_{self.labels[ix]}",
                          use_container_width=True, on_click=self.assign_annotation,
                          args=(ixs[self.annotator['pos']],
                                self.labels[ix],))

        if 'learner' in self.alearn_loop:
            sample = df.iloc[[ixs[self.annotator['pos']]]]
            sample_pred = self.alearn_loop['learner'].estimator.predict(self.get_features(sample))
            st.info(f'Learner predicted this as {sample_pred}')

        for handler in self.handlers:
            handler.display_sample(df, ixs[self.annotator['pos']])


    def split_annotations(self, test_size=0.2):

        if len(self.annotator['annotations']) > 0:
            annotations_df = pd.DataFrame(
                self.annotator['annotations'].items(), columns=['ix', 'label']
            )

            train_annotations, test_annotations = train_test_split(annotations_df, test_size=test_size)

            train_data = self.df.iloc[train_annotations.ix.values]
            train_data['label'] = train_annotations.label.values

            test_data = self.df.iloc[test_annotations.ix.values]
            test_data['label'] = test_annotations.label.values

            self.df = self.df.iloc[~self.df.index.isin(annotations_df.ix.values)]

            self.df.reset_index(inplace=True, drop=True)

            if 'data_train' not in self.alearn_loop:
                self.alearn_loop['data_train'] = train_data
            else:
                self.alearn_loop['data_train'] = pd.concat(
                    [self.alearn_loop['data_train'],
                     train_data])

            self.alearn_loop['data_train'].reset_index(inplace=True, drop=True)

            if 'data_test' not in self.alearn_loop:
                self.alearn_loop['data_test'] = test_data
            else:
                self.alearn_loop['data_test'] = pd.concat([self.alearn_loop['data_test'], test_data])

            self.alearn_loop['data_test'].reset_index(inplace=True, drop=True)

            self.annotator = None

    def get_features(self, data, labels=False):
        features = []
        for handler in self.handlers:
            handler_features = handler.vectorize(data)
            features.append(handler_features)

        print(np.concatenate(features, axis=1).shape)
        # print(np.concatenate(features, axis=0).shape)

        print(f"Before {type(features[0])} {features[0].shape}")

        features = np.concatenate(features, axis=1)

        print(f"After {type(features)} {features.shape}")


        if labels:
            y = data['label'].values
            return features, y
        else:
            return features

    @staticmethod
    def uncertainty_sampling(classifier, X, n_instances=10, **uncertainty_measure_kwargs):
        from modAL.utils.selection import shuffled_argmax
        uncertainty = MultiController.classifier_uncertainty(classifier, X)
        return shuffled_argmax(uncertainty, n_instances=n_instances)

    @staticmethod
    def classifier_uncertainty(classifier, X):
        # calculate uncertainty for each point provided
        classwise_uncertainty = classifier.estimator.predict_proba(X)

        # Ignore None labels to confirm and focus on actual labels
        prediction = classifier.predict(X)
        prediction = np.array([1 if p != 'None' else 0 for p in list(prediction)])

        # for each point, select the maximum uncertainty
        uncertainty = 1 - np.max(classwise_uncertainty, axis=1)

        return uncertainty * prediction  # A, uncertainty * predictionB

    def evaluate_model(self, features, labels):
        y_pred = self.alearn_loop['learner'].predict(features)
        print(classification_report(labels, y_pred))
        return format_metrics(classification_report(labels, y_pred, output_dict=True))

    def display_learner_metrics(self):

        st.caption("<hr>", unsafe_allow_html=True)
        st.caption("<h1 style=\"color:'coral'\"> Classifier metrics <h1>", unsafe_allow_html=True)
        st.caption("Metrics on positive labels")
        st.subheader("Train")

        if self.step_entry:
            eval_features, eval_labels = self.get_features(self.alearn_loop['data_test'], labels=True)
            metrics = self.evaluate_model(eval_features, eval_labels)
        else:
            metrics = self.alearn_loop['current_metrics']

        for label in self.labels:
            if label in metrics:
                st.subheader(label)
                if 'current_metrics' in self.alearn_loop and label in \
                        self.alearn_loop['current_metrics'] and self.step_entry:
                    display_metric_line(metrics[label],
                                        self.alearn_loop['current_metrics'][label])
                else:
                    display_metric_line(metrics[label])

        if 'current_metrics' not in self.alearn_loop:
            self.alearn_loop['current_metrics'] = None
        self.alearn_loop['current_metrics'] = metrics

        st.button('Annotate more samples', on_click=self.next_alearn_step)

        super(MultiController, self).display_learner_metrics()

    def annotate_new_samples(self):

        if 'learner' not in self.alearn_loop:
            st.error('Learner hasn\'t been trained yet')
        else:
            if 'sample_annotator' not in self.alearn_loop:
                self.alearn_loop['sample_annotator'] = {}
            if not self.annotator:

                steps = 2

                my_bar = st.progress((0 / steps), text="Creating pool features")

                pool_features = self.get_features(self.df, labels=False)

                my_bar.progress((1 / steps), text="Querying learner")

                self.alearn_loop['sample_annotator']['idxs'], _ = self.alearn_loop['learner'].query(pool_features)

                my_bar.progress((2 / steps), text="Start annotating")

                self.annotate(self.df, self.alearn_loop['sample_annotator']['idxs'])

            else:

                self.annotate(self.df, self.alearn_loop['sample_annotator']['idxs'])

    def retrain_model(self):

        steps = 2

        my_bar = st.progress((0 / steps), text="Creating features")
        train_features, labels = self.get_features(self.alearn_loop['data_train'], labels=True)

        my_bar.progress((1 / steps), text="Retraining learner")
        self.alearn_loop['learner'].fit(train_features, labels)

        my_bar.progress((2 / steps), text="Done")
        print('retraining', self.alearn_loop['step'])
        self.next_alearn_step()
        print('retrained', self.alearn_loop['step'])

    def export_model(self, export_folder='./exports'):

        export_path = os.path.join(export_folder, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        Path(export_path).mkdir(parents=True, exist_ok=True)

        model_name = os.path.join(export_path, 'model.pkl')
        pickle.dump(self.alearn_loop['learner'].estimator, open(model_name, 'wb'))

        for handler in self.handlers:
            vectorizer_name = os.path.join(export_path, f'vectorizer_{handler.col_name}.pkl')
            pickle.dump(handler.vectorizer, open(vectorizer_name, 'wb'))

        data_train_name = os.path.join(export_path, 'data_train.csv')
        self.alearn_loop['data_train'].to_csv(data_train_name)

        data_test_name = os.path.join(export_path, 'data_test.csv')
        self.alearn_loop['data_test'].to_csv(data_test_name)