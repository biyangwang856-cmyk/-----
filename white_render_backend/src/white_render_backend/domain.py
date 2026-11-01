class EngineError(Exception):
    def __init__(self, code, message, *, status=422):
        self.code, self.message, self.status = code, message, status
        super().__init__(f"{code}: {message}")
    def as_dict(self):return {"code": self.code, "message": self.message}

def require(condition,code,message,*,status=422):
    if not condition:raise EngineError(code,str(message),status=status)
