import streamlit as st

from .utils import check_column_types, check_alearn_loop, next_alearn_step
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import numpy as np
import pandas as pd


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


def assign_annotation(ix, label):

    if 'annotations' not in st.session_state['annotator']:
        st.session_state['annotator']['annotations'] = {}
    if ix not in st.session_state['annotator']['annotations']:
        st.session_state['annotator']['annotations'][ix] = None
    st.session_state['annotator']['annotations'][ix] = label
    st.session_state['annotator']['pos'] += 1


def annotate(df, ixs):

    if 'annotator' not in st.session_state:
        st.session_state['annotator'] = {}

    if 'pos' not in st.session_state['annotator']:
        st.session_state['annotator']['pos'] = 0

    if st.session_state['annotator']['pos'] >= len(ixs):
        with st.expander('Annotations'):
            st.dataframe(pd.DataFrame(st.session_state['annotator']['annotations'].items(), columns=['ix', 'label']))
        st.button('Proceed to next step', on_click=next_alearn_step)
    else:
        my_bar = st.progress((st.session_state['annotator']['pos'] / len(ixs)))

        data_col = [col for col, status in st.session_state['col_status'].items() if status][0]
        st.markdown(df.iloc[ixs[st.session_state['annotator']['pos']]][data_col])

        cols = st.columns(tuple(1 for _ in st.session_state['labels']))
        for ix, col in enumerate(cols):
            with col:
                st.button(st.session_state['labels'][ix], key=f"apply_{st.session_state['labels'][ix]}",
                          use_container_width=True, on_click=assign_annotation,
                          args=(ixs[st.session_state['annotator']['pos']],
                                st.session_state['labels'][ix],))


def nlp_binary():

    if check_column_types() != ['TEXT']:
        st.error(f"Column types are {check_column_types()} but {['TEXT']} was expected")

    check_alearn_loop()

    if st.session_state['alearn_loop']['step'] == 0:

        create_seeds()

    if st.session_state['alearn_loop']['step'] == 1:

        annotate_seeds()

        st.info('Training initial learner')

    if st.session_state['alearn_loop']['step'] == 2:
        st.info('Annotate samples')

    if st.session_state['alearn_loop']['step'] == 3:
        st.info('Re-train and test')

    # st.json(st.session_state)