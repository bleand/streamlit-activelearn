import streamlit as st
import pandas as pd
from alearn.setup_functions import setup_labels
from alearn.utils import load_yaml

# Streamlit settings
st.set_page_config(layout="centered")

# Load configs
st.session_state['cfg'] = load_yaml('config.yaml')
COL_OPTIONS = st.session_state['cfg']['COL_OPTIONS']
MODEL_TYPES = st.session_state['cfg']['MODEL_TYPES']


def load_dataset():
    if 'dataset' in st.session_state:
        st.session_state['df'] = pd.read_csv(st.session_state['dataset'])
        st.session_state['col_status'] = {col: False for col in st.session_state['df']}
        st.session_state['col_type'] = {col: COL_OPTIONS[0] for col in st.session_state['df']}


def toggle(col):
    if st.session_state['col_status'][col]:
        st.session_state['col_status'][col] = False
    else:
        st.session_state['col_status'][col] = True


def change_selected_type(col):
    st.session_state['col_type'][col] = st.session_state[f"selected_type_{col}"]


def set_model_type():
    st.session_state['model_type'] = st.session_state['selected_model_type']


# """
# PAGE STARTS
# """

st.title("Active Learn")

if 'df' not in st.session_state:

    st.file_uploader("Upload a csv file", key='dataset', on_change=load_dataset)

else:

    st.markdown("""---""")
    if len(st.session_state['df'].columns) > 10:
        st.error('Too many columns in dataset')
    else:
        # """
        # Defining dataset columns
        # """
        st.header('Define columns')
        col1, col2, col3 = st.columns((8, 2, 2))
        with col1:
            st.text('Column')
        with col2:
            st.text('Type')
        with col3:
            st.text('Enable')
        for col in st.session_state['col_status']:
            col1, col2, col3 = st.columns((8, 2, 2))
            with col1:
                col_formatted = f'<p style="font-size: 24px;">{col}</p>'
                st.markdown(col_formatted, unsafe_allow_html=True)
            with col2:
                st.selectbox('', COL_OPTIONS, key=f"selected_type_{str(col)}",
                             label_visibility='collapsed', on_change=change_selected_type, args=(col,), index=0)
            with col3:
                if st.session_state['col_status'][col]:
                    st.button('ENABLE', key=f"enable_{str(col)}", on_click=toggle, args=(col,), type='primary')
                else:
                    st.button('ENABLE', key=f"enable_{str(col)}", on_click=toggle, args=(col,))

    # """
    # Selecting model type
    # """
    st.markdown("""---""")
    if len([1 for c in st.session_state['col_status'] if st.session_state['col_status'][c]]) == 0:
        st.error('Must enable at least one column')
    else:
        if 'model_type' not in st.session_state:
            st.session_state['model_type'] = MODEL_TYPES[0]
        st.header('Select model type')
        st.radio('Model type', options=MODEL_TYPES, key='selected_model_type', on_change=set_model_type, index=0)
        setup_labels()