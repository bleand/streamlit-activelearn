from .text_handler import TextHandler


def handler_from_settings(dtype, col_name):
    if dtype == 'TEXT':
        return TextHandler(dtype, col_name)

