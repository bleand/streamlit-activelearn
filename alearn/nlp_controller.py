from .controller import AlearnController
from .utils import check_column_types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report
from .utils import format_metrics, display_metric_line


class NLPController(AlearnController):

    def __init__(self, *args, **kwargs):
        self.vectorizer = None
        self.sampling = self.uncertainty_sampling
        super(NLPController, self).__init__(*args, **kwargs)

    def check_run(self):
        status = {'error': False}
        if check_column_types() != [self.dtype]:
            status['error'] = True
            status['message'] = f"Column types are {check_column_types()} but {[self.dtype]} was expected"
        return status

    def check_alearn_loop(self):
        if not self.alearn_loop:
            self.alearn_loop = {
                'step': 0,
                'seed_annotation': {},
            }

    def create_seeds(self):

        if not self.seeds:
            self.seeds = {}

        st.info('Select seeds')

        col1, col2, col3 = st.columns((8, 2, 2))
        with col1:
            st.text('Seed')
        with col2:
            st.text('Label')
        with col3:
            st.text('')

        for seed in self.seeds:
            with col1:
                st.text(seed)
            with col2:
                st.text(self.seeds[seed])
            with col3:
                st.button("🗑", key=f'delete_{seed}', on_click=self.delete_seed, args=(seed,))

        col1, col2 = st.columns((8, 2))
        with col1:
            st.text_input(label='', label_visibility='collapsed', key='new_seed')
        with col2:
            st.selectbox(label='', options=self.labels, key=f"new_seed_label",
                         label_visibility='collapsed')
        col1, col2 = st.columns((8, 2))
        with col2:
            st.button('Add seed', on_click=self.add_seed)  # , args=(st,))

        if set(self.seeds.values()) != set(self.labels):
            st.warning("At lease one seed of each label is needed")
        else:
            st.button('Proceed to next step', on_click=self.next_alearn_step)  # , args=(st,))

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

        if self.vectorizer:
            features = self.vectorizer.transform(data.review.values)
        else:
            self.vectorizer = TfidfVectorizer()
            features = self.vectorizer.fit_transform(data.review.values)

        if labels:
            y = data['label'].values
            return features, y
        else:
            return features

    def annotate_visual(self, df, ixs):

        my_bar = st.progress((self.annotator['pos'] / len(ixs)))

        data_col = [col for col, status in self.col_status.items() if status][0]
        sample = df.iloc[[ixs[self.annotator['pos']]]][[data_col]]
        st.markdown(sample[data_col].iloc[0])

        if 'learner' in self.alearn_loop:
            sample_pred = self.alearn_loop['learner'].estimator.predict(self.get_features(sample))
            
            st.info(f'Learner predicted this as {sample_pred}')

        cols = st.columns(tuple(1 for _ in self.labels))
        for ix, col in enumerate(cols):
            with col:
                st.button(self.labels[ix], key=f"apply_{self.labels[ix]}",
                          use_container_width=True, on_click=self.assign_annotation,
                          args=(ixs[self.annotator['pos']],
                                self.labels[ix],))

    def annotate_seeds(self):

        if not self.vectorizer:
            steps = 7

            my_bar = st.progress((0 / steps), text="Creating vectorizer")

            self.vectorizer = TfidfVectorizer(max_features=10000)

            data_col = [col for col, status in self.col_status.items() if status][0]
            self.vectorizer.fit(self.df[data_col].tolist())

            my_bar.progress((1 / steps), text="Collecting seeds")

            seeds = []
            for label in self.labels:
                seeds.append([seed for seed, seed_label in self.seeds.items() if seed_label == label])

            my_bar.progress((2 / steps), text="Vectorizing seeds")
            seed_vectors = []
            for seed in seeds:
                seed_vectors.append(self.vectorizer.transform(seed).todense())

            my_bar.progress((3 / steps), text="Vectorizing pool")
            pool_vectors = self.vectorizer.transform(self.df[data_col].tolist()).todense()
            NUM_SAMPLES = 2

            my_bar.progress((4 / steps), text="Calculating similarity")
            # Calculate cosine distance (opposite to cosine similarity)
            distances = []
            for vector in seed_vectors:
                distances.append(scipy.spatial.distance.cdist(vector, pool_vectors, metric='cosine'))

            my_bar.progress((5 / steps), text="Sorting distances")
            # Sort in ascending order and pick the first 10 of each
            closest = []
            for dist in distances:
                closest.append(np.argsort(dist)[:, :NUM_SAMPLES].flatten())

            my_bar.progress((6 / steps), text="Collecting idxs")

            self.alearn_loop['seed_annotation']['idxs'] = []
            for close in closest:
                self.alearn_loop['seed_annotation']['idxs'].extend(list(set(list(close))))

            my_bar.progress((7 / steps), text="Done")

            self.annotate(self.df, self.alearn_loop['seed_annotation']['idxs'])

        else:

            self.annotate(self.df, self.alearn_loop['seed_annotation']['idxs'])

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

    @staticmethod
    def uncertainty_sampling(classifier, X, n_instances=10, **uncertainty_measure_kwargs):
        from modAL.utils.selection import shuffled_argmax
        uncertainty = NLPController.classifier_uncertainty(classifier, X)
        return shuffled_argmax(uncertainty, n_instances=n_instances)

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

        super(NLPController, self).display_learner_metrics()

    def annotate_new_samples(self):

        if 'learner' not in self.alearn_loop:
            st.error('Learner hasn\'t been trained yet')
        else:
            if 'sample_annotator' not in self.alearn_loop:
                self.alearn_loop['sample_annotator'] = {}
            if not self.annotator:

                my_bar = st.progress((0 / 2), text="Creating pool features")

                pool_features = self.get_features(self.df, labels=False)

                my_bar.progress((1 / 2), text="Querying learner")

                self.alearn_loop['sample_annotator']['idxs'], _ = self.alearn_loop['learner'].query(pool_features)

                my_bar.progress((2 / 2), text="Start annotating")

                self.annotate(self.df, self.alearn_loop['sample_annotator']['idxs'])

            else:

                self.annotate(self.df, self.alearn_loop['sample_annotator']['idxs'])

    def retrain_model(self):

        my_bar = st.progress((0 / 2), text="Creating features")
        train_features, labels = self.get_features(self.alearn_loop['data_train'], labels=True)

        my_bar.progress((1 / 2), text="Retraining learner")
        self.alearn_loop['learner'].fit(train_features, labels)

        my_bar.progress((2 / 2), text="Done")
        print('retraining', self.alearn_loop['step'])
        self.next_alearn_step()
        print('retrained', self.alearn_loop['step'])