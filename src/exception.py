import sys
from logger import logging

# error details will basically will be in sys
def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # all information will be stored in this variable
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno, str(error)
    )

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message