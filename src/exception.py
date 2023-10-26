import sys  # Import the sys module for accessing system-related information.
import logging  # Import the logging module for logging messages.

# Function to generate a detailed error message.
def error_message_detail(error, error_detail: sys):
    # Get information about the current exception.
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename of the code.
    
    # Create an error message with details.
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message

# Custom exception class.
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is converted to a string.