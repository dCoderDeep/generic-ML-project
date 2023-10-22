import logging
import sys

def error_message_detail(error, error_details:sys):
    e_type, e_object, e_traceback = sys.exc_info()   # e_traceback will tell on which file, which line the exception has occurred
    
    file_name = e_traceback.tb_frame.f_code.co_filename
    line_number = e_traceback.tb_lineno
    
    error_message = f"Error occurred in python script name [{file_name}] at line number [{line_number}], error message [{str(error)}]."
    return error_message
    
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details=error_detail)
        
    def __str__(self):
        return self.error_message