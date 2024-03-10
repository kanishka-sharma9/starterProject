import sys
import logging
from src.logger import logging

def error_msg_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="Error ocurred in python script name[{0}] line number [{1}] err message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_msg

class customExeption(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg=error_msg_details(error_msg,error_detail)

    def __str__(self) -> str:
        return self.error_msg
    
