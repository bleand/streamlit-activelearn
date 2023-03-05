from .nlp_controller import NLPController


def controller_from_settings(session):
    if 'TEXT' == 'TEXT': # Should be replaced with the dtype validation
        return NLPController(session)
