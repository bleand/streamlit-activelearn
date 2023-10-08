import streamlit as st


def add_label():
    if st.session_state['new_label'] not in st.session_state['labels']:
        st.session_state['labels'].append(st.session_state['new_label'])
    st.session_state['new_label'] = ''


def delete_label(label):
    st.session_state['labels'].remove(label)


def setup_labels():

    if 'labels' not in st.session_state:
        st.session_state['labels'] = []

    st.markdown("""---""")
    # """
    # Create training labels section
    # """
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
            st.button("🗑", key=f'delete_{label}', on_click=delete_label, args=(label,))

    col1, col2 = st.columns((8, 2))
    with col1:
        st.text_input('a', label_visibility='collapsed', key='new_label', on_change=add_label)

    # """
    # Validate labels
    # """

    if st.session_state['model_type'] == 'Binary' and len(st.session_state['labels']) != 2:
        st.warning('Binary requires 2 labels')
    elif st.session_state['model_type'] in ['Multiclass', 'Multilabel'] and len(st.session_state['labels']) <2:
        st.warning(f"{st.session_state['model_type']} requires at least 2 labels")
    elif len(st.session_state['labels']) > 1 or \
            (st.session_state['model_type'] == 'NER' and len(st.session_state['labels']) > 0):
        st.success('You can move to the next page to start training!')
    else:
        st.warning('Create labels')