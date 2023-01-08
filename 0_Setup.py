import streamlit as st
import pandas as pd


def load_dataset():
    if 'dataset' in st.session_state:
        st.session_state['df'] = pd.read_csv(st.session_state['dataset'])
        st.session_state['col_status'] = {col: False for col in st.session_state['df']}
        st.session_state['col_type'] = {col: None for col in st.session_state['df']}


def toggle(col):
    if st.session_state['col_status'][col]:
        st.session_state['col_status'][col] = False
    else:
        st.session_state['col_status'][col] = True


def add_label():
    if st.session_state['new_label'] not in st.session_state['labels']:
        st.session_state['labels'].append(st.session_state['new_label'])


def delete_label(label):
    st.session_state['labels'].remove(label)


st.title("Active Learn")

COL_OPTIONS = ['TEXT', 'NUM', 'CAT']
MODEL_TYPES = ['Binary', 'Multiclass', 'Multilabel', 'NER']

st.file_uploader("Upload a csv file", key='dataset', on_change=load_dataset)

if 'df' in st.session_state:
    st.markdown("""---""")
    if len(st.session_state['df'].columns) > 10:
        st.error('Too many columns in dataset')
    else:
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
                st.selectbox('', COL_OPTIONS, key=f"type_{str(col)}",
                             label_visibility='collapsed')
            with col3:
                if st.session_state['col_status'][col]:
                    st.button('ENABLE', key=f"enable_{str(col)}", on_click=toggle, args=(col,), type='primary')
                else:
                    st.button('ENABLE', key=f"enable_{str(col)}", on_click=toggle, args=(col,))
    st.markdown("""---""")
    if len([1 for c in st.session_state['col_status'] if st.session_state['col_status'][c]]) == 0:
        st.error('Must enable at least one column')
    else:
        st.header('Select model type')
        st.radio('Model type', options=MODEL_TYPES, key='model_type')

        if st.session_state['model_type'] != 'Binary':
            if 'labels' not in st.session_state:
                st.session_state['labels'] = []
            st.markdown("""---""")
            st.title("Create training labels")
            col1, col2 = st.columns((8, 2))
            with col1:
                st.text('Label name')
            with col2:
                st.text('Action')

            for label in st.session_state['labels']:
                col1, col2 = st.columns((8, 2))
                with col1:
                    st.text(label)
                with col2:
                    st.button("🗑", key=f'delete_{label}', on_click=delete_label, args=(label, ))

            col1, col2 = st.columns((8, 2))
            with col1:
                st.text_input('', label_visibility='collapsed', key='new_label', on_change=add_label)
            with col2:
                st.button('Add', on_click=add_label)

            if len(st.session_state['labels']) > 1 or \
                    (st.session_state['model_type'] == 'NER' and len(st.session_state['labels']) > 0):
                st.success('You can move to the next page to start training!')
            else:
                st.warning('Create labels')
        else:
            st.success('You can move to the next page to start training!')

# st.session_state