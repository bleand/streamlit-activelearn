from .controller import AlearnController
from .handler_builder import handler_from_settings


class MultiController(AlearnController):

    def __init__(self, *args, **kwargs):
        session = args[0]
        self.handlers = self.init_handlers(session)
        print(self.handlers)
        super(MultiController, self).__init__(*args, **kwargs)

    @staticmethod
    def init_handlers(session):
        handlers = []
        for col in session['col_status']:
            if session['col_status'][col]:
                handlers.append(handler_from_settings(session['col_type'][col]))
        return handlers
