from .nlp_controller import NLPController
from .multi_controller import MultiController

def controller_from_settings(session):
    return MultiController(session)
    #if 'TEXT' != 'TEXT': # Should be replaced with the dtype validation
    #    return NLPController(session)
