from app.schemas.error_codes import ErrorCode

class BusinessLogicError(Exception):
    
    def __init__(self, message: str, error_code: ErrorCode):
        self.message = message
        self.error_code = error_code
        super().__init__(message)