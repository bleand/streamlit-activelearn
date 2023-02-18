import streamlit as st
from sklearn.metrics import classification_report

def check_column_types():
    return list(set([st.session_state['col_type'][col]
              for col in st.session_state['col_status']
              if st.session_state['col_status'][col]]))


def check_alearn_loop():
    if 'alearn_loop' not in st.session_state:
        st.session_state['alearn_loop'] = {
            'step': 0,
            'seed_annotation': {},
        }


def next_alearn_step():
    st.session_state['alearn_loop']['step'] += 1
    if st.session_state['alearn_loop']['step'] >= 6:
        st.session_state['alearn_loop']['step'] = 3
        st.experimental_rerun()


def format_metrics(report):
    report_out = {}

    for label in st.session_state['labels']:
        if label not in report:
            continue
        if label not in report_out:
            report_out[label] = {}
        for key, value in report[label].items():
            if key == "support":
                report_out[label][key] = round(report[label][key], 2)
            else:
                report_out[label][key] = round(report[label][key]*100, 2)
    report_out['accuracy'] = round(report['accuracy']*100, 2)
    return report_out


def evaluate_model(clf, features, labels):
    y_pred = clf.predict(features)
    print(classification_report(labels, y_pred))
    return format_metrics(classification_report(labels, y_pred, output_dict=True))
