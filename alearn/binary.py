import streamlit as st

from .utils import check_column_types, check_alearn_loop, next_alearn_step, evaluate_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def get_features(df, labels=False, vectorizer=None):
    if vectorizer:
        features = vectorizer.transform(df.review.values)
    else:
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(df.review.values)

    if labels:
        y = df['label'].values
        return features, y, vectorizer
    else:
        return features, vectorizer


def add_seed():
    if st.session_state['new_seed'] not in st.session_state['seeds']:
        st.session_state['seeds'][st.session_state['new_seed']] = st.session_state['new_seed_label']
    st.session_state['new_seed'] = ''


def delete_seed(seed):
    del st.session_state['seeds'][seed]


def create_seeds():

    if 'seeds' not in st.session_state:
        st.session_state['seeds'] = {}

    st.info('Select seeds')
    col1, col2, col3 = st.columns((8, 2, 2))
    with col1:
        st.text('Seed')
    with col2:
        st.text('Label')
    with col3:
        st.text('')

    for seed in st.session_state['seeds']:
        with col1:
            st.text(seed)
        with col2:
            st.text(st.session_state['seeds'][seed])
        with col3:
            st.button("🗑", key=f'delete_{seed}', on_click=delete_seed, args=(seed,))

    col1, col2 = st.columns((8, 2))
    with col1:
        st.text_input(label='', label_visibility='collapsed', key='new_seed')
    with col2:
        st.selectbox(label='', options=st.session_state['labels'], key=f"new_seed_label",
                     label_visibility='collapsed')
    col1, col2 = st.columns((8, 2))
    with col2:
        st.button('Add seed', on_click=add_seed)#, args=(st,))

    if set(st.session_state['seeds'].values()) != set(st.session_state['labels']):
        st.warning("At lease one seed of each label is needed")
    else:
        st.button('Proceed to next step', on_click=next_alearn_step)#, args=(st,))


def annotate_seeds():

    if 'vectorizer' not in st.session_state:
        steps = 7

        my_bar = st.progress((0 / steps), text="Creating vectorizer")

        st.session_state['vectorizer'] = TfidfVectorizer(max_features=10000)

        data_col = [col for col, status in st.session_state['col_status'].items() if status][0]
        st.session_state['vectorizer'].fit(st.session_state['df'][data_col].tolist())

        my_bar.progress((1/steps), text="Collecting seeds")

        seeds = []
        for label in st.session_state['labels']:
            seeds.append([seed for seed, seed_label in st.session_state['seeds'].items() if seed_label == label])

        my_bar.progress((2 / steps), text="Vectorizing seeds")
        seed_vectors = []
        for seed in seeds:
            seed_vectors.append(st.session_state['vectorizer'].transform(seed).todense())

        my_bar.progress((3 / steps), text="Vectorizing pool")
        pool_vectors = st.session_state['vectorizer'].transform(st.session_state['df'][data_col].tolist()).todense()
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

        st.session_state['alearn_loop']['seed_annotation']['idxs'] = []
        for close in closest:
            st.session_state['alearn_loop']['seed_annotation']['idxs'].extend(list(set(list(close))))

        my_bar.progress((7 / steps), text="Done")

        annotate(st.session_state['df'], st.session_state['alearn_loop']['seed_annotation']['idxs'])

    else:

        annotate(st.session_state['df'], st.session_state['alearn_loop']['seed_annotation']['idxs'])


def annotate_new_samples():

    if 'learner' not in st.session_state['alearn_loop']:
        st.error('Learner hasn\'t been trained yet')
    else:
        if 'sample_annotator' not in st.session_state['alearn_loop']:
            st.session_state['alearn_loop']['sample_annotator'] = {}
        if 'annotator' not in st.session_state:

            my_bar = st.progress((0 / 2), text="Creating pool features")

            pool_features, st.session_state['vectorizer'] = get_features(st.session_state['df'],
                                                                        labels=False,
                                                                        vectorizer=st.session_state['vectorizer'])

            my_bar.progress((1 / 2), text="Querying learner")

            st.session_state['alearn_loop']['sample_annotator']['idxs'], _ = st.session_state['alearn_loop'][
                'learner'].query(pool_features)

            my_bar.progress((2 / 2), text="Start annotating")

            annotate(st.session_state['df'], st.session_state['alearn_loop']['sample_annotator']['idxs'])

        else:

            annotate(st.session_state['df'], st.session_state['alearn_loop']['sample_annotator']['idxs'])


def assign_annotation(ix, label):

    if 'annotations' not in st.session_state['annotator']:
        st.session_state['annotator']['annotations'] = {}
    if ix not in st.session_state['annotator']['annotations']:
        st.session_state['annotator']['annotations'][ix] = None
    st.session_state['annotator']['annotations'][ix] = label
    st.session_state['annotator']['pos'] += 1


def split_annotations(test_size=0.2):

    if len(st.session_state['annotator']['annotations']) > 0:
        annotations_df = pd.DataFrame(
                    st.session_state['annotator']['annotations'].items(), columns=['ix', 'label']
                )
        train_annotations, test_annotations = train_test_split(annotations_df, test_size=test_size)

        train_data = st.session_state['df'].iloc[train_annotations.ix.values]
        train_data['label'] = train_annotations.label.values

        test_data = st.session_state['df'].iloc[test_annotations.ix.values]
        test_data['label'] = test_annotations.label.values

        st.session_state['df'] = st.session_state['df'].iloc[
            ~st.session_state['df'].index.isin(annotations_df.ix.values)]

        st.session_state['df'].reset_index(inplace=True, drop=True)

        if 'data_train' not in st.session_state['alearn_loop']:
            st.session_state['alearn_loop']['data_train'] = train_data
        else:
            st.session_state['alearn_loop']['data_train'] = pd.concat([st.session_state['alearn_loop']['data_train'],
                                                                       train_data])

        st.session_state['alearn_loop']['data_train'].reset_index(inplace=True, drop=True)

        if 'data_test' not in st.session_state['alearn_loop']:
            st.session_state['alearn_loop']['data_test'] = test_data
        else:
            st.session_state['alearn_loop']['data_test'] = pd.concat([st.session_state['alearn_loop']['data_test'],
                                                                       test_data])

        st.session_state['alearn_loop']['data_test'].reset_index(inplace=True, drop=True)

        del st.session_state['annotator']


def annotate(df, ixs):

    if 'annotator' not in st.session_state:
        st.session_state['annotator'] = {}

    if 'pos' not in st.session_state['annotator']:
        st.session_state['annotator']['pos'] = 0

    if st.session_state['annotator']['pos'] >= len(ixs):
        #with st.expander('Annotations'):
        #    st.session_state['annotator']['annotations_df'] = pd.DataFrame(
        #        st.session_state['annotator']['annotations'].items(), columns=['ix', 'label']
        #    )
        #    st.dataframe(st.session_state['annotator']['annotations_df'])
        split_annotations()
        st.button('Proceed to next step', on_click=next_alearn_step)


    else:

        my_bar = st.progress((st.session_state['annotator']['pos'] / len(ixs)))

        data_col = [col for col, status in st.session_state['col_status'].items() if status][0]
        sample = df.iloc[[ixs[st.session_state['annotator']['pos']]]][[data_col]]
        st.markdown(sample[data_col].iloc[0])

        if 'learner' in st.session_state['alearn_loop']:
            sample_pred = st.session_state['alearn_loop']['learner'].estimator.predict(get_features(sample,
                                                                           vectorizer=st.session_state['vectorizer'])[0]
                                                              )
            st.info(f'Learner predicted this as {sample_pred}')

        cols = st.columns(tuple(1 for _ in st.session_state['labels']))
        for ix, col in enumerate(cols):
            with col:
                st.button(st.session_state['labels'][ix], key=f"apply_{st.session_state['labels'][ix]}",
                          use_container_width=True, on_click=assign_annotation,
                          args=(ixs[st.session_state['annotator']['pos']],
                                st.session_state['labels'][ix],))


def classifier_uncertainty(classifier, X):
    # calculate uncertainty for each point provided
    classwise_uncertainty = classifier.estimator.predict_proba(X)

    # Ignore None labels to confirm and focus on actual labels
    prediction = classifier.predict(X)
    prediction = np.array([1 if p != 'None' else 0 for p in list(prediction)])

    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)

    return uncertainty * prediction  # A, uncertainty * predictionB


def uncertainty_sampling(classifier, X, n_instances = 10,  **uncertainty_measure_kwargs):
    from modAL.utils.selection import shuffled_argmax
    uncertainty  = classifier_uncertainty(classifier, X)
    return shuffled_argmax(uncertainty, n_instances=n_instances)


def create_initial_learner():

    # pool_features, st.session_state['vectorizer'] = get_features(st.session_state['df'],
    #                                                             labels=False,
    #                                                             vectorizer=st.session_state['vectorizer'])

    my_bar = st.progress((0 / 3), text="Creating train features")

    train_features, labels, st.session_state['vectorizer'] = get_features(
        st.session_state['alearn_loop']['data_train'],
        labels=True,
        vectorizer=st.session_state['vectorizer'])

    my_bar.progress((1 / 3), text="Creating test features")

    #eval_features, eval_labels, st.session_state['vectorizer'] = get_features(
    #    st.session_state['alearn_loop']['data_test'],
    #    labels=True,
    #    vectorizer=st.session_state['vectorizer'])

    my_bar.progress((2 / 3), text="Creating learner")

    st.session_state['alearn_loop']['learner'] = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=uncertainty_sampling,
        X_training=train_features, y_training=labels
    )

    my_bar.progress((3 / 3), text="Done")

    next_alearn_step()


def display_metric_line(data, current_data=None):
    col1, col2, col3, col4 = st.columns(4)
    if current_data:
        with col1:
            st.metric("Precision", data['precision'], round(data['precision'] - current_data['precision'], 2))
        with col2:
            st.metric("Recall", data['recall'], round(data['recall'] - current_data['recall'], 2))
        with col3:
            st.metric("F1", data['f1-score'], round(data['f1-score'] - current_data['f1-score'], 2))
        with col4:
            st.metric("Support", data['support'], round(data['support'] - current_data['support'], 2))
    else:
        with col1:
            st.metric("Precision", data['precision'])
        with col2:
            st.metric("Recall", data['recall'])
        with col3:
            st.metric("F1", data['f1-score'])
        with col4:
            st.metric("Support", data['support'])


def display_learner_metrics():

    eval_features, eval_labels, st.session_state['vectorizer'] = get_features(
        st.session_state['alearn_loop']['data_test'],
        labels=True,
        vectorizer=st.session_state['vectorizer'])

    metrics = evaluate_model(st.session_state['alearn_loop']['learner'],
                             eval_features,
                             eval_labels)

    print(metrics)

    st.caption("<hr>", unsafe_allow_html=True)
    st.caption("<h1 style=\"color:'coral'\"> Classifier metrics <h1>", unsafe_allow_html=True)
    st.caption("Metrics on positive labels")
    st.subheader("Train")
    # if current_metrics:
    #    display_metric_line(metrics['classifier']['train']['1'], current_metrics['classifier']['train']['1'])
    # else:
    # display_metric_line(metrics['train']['1'])
    for label in st.session_state['labels']:
        if label in metrics:
            st.subheader(label)
            if 'current_metrics' in st.session_state['alearn_loop'] and label in st.session_state['alearn_loop']['current_metrics']:
                display_metric_line(metrics[label],
                                    st.session_state['alearn_loop']['current_metrics'][label])
            else:
                display_metric_line(metrics[label])

    if 'current_metrics' not in st.session_state['alearn_loop']:
        st.session_state['alearn_loop']['current_metrics'] = None
    st.session_state['alearn_loop']['current_metrics'] = metrics

    st.button('Annotate more samples', on_click=next_alearn_step)  # , args=(st,))


def retrain_model():

    my_bar = st.progress((0 / 2), text="Creating features")
    train_features, labels, st.session_state['vectorizer'] = get_features(
        st.session_state['alearn_loop']['data_train'],
        labels=True,
        vectorizer=st.session_state['vectorizer'])

    my_bar.progress((1 / 2), text="Retraining learner")
    st.session_state['alearn_loop']['learner'].fit(train_features, labels)

    my_bar.progress((2 / 2), text="Done")
    print('retraining', st.session_state['alearn_loop']['step'])
    next_alearn_step()
    print('retrained', st.session_state['alearn_loop']['step'])


def nlp_binary():

    if check_column_types() != ['TEXT']:

        st.error(f"Column types are {check_column_types()} but {['TEXT']} was expected")

    check_alearn_loop()

    if st.session_state['alearn_loop']['step'] == 0:

        create_seeds()

        st.info('Create initial seeds')

    if st.session_state['alearn_loop']['step'] == 1:

        annotate_seeds()

        st.info('Annotate seeds')

    if st.session_state['alearn_loop']['step'] == 2:

        create_initial_learner()

        st.info('Train initial learner')

    if st.session_state['alearn_loop']['step'] == 3:

        display_learner_metrics()

        st.info('Metrics')

    if st.session_state['alearn_loop']['step'] == 4:

        annotate_new_samples()

        st.info('More annotations samples')

    if st.session_state['alearn_loop']['step'] == 5:

        retrain_model()

        st.info('Re-train and test')

    # st.json(st.session_state)