

class OrchestratorInterface:

    engine =  None

    @classmethod
    def initialize(cls, engine_name:str):

        #right now just this, according to folders
        if engine_name == "asyncio_ray_engine":
            pass
