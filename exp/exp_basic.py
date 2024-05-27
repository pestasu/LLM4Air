from models import AutoTimes_Llama, AutoTimes_Gpt2, AutoTimes_Opt_1b


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # spatial-temporal
            'Airformer': airformer,
            'Gagnn': gagnn,
            # LLM
            'AutoTimes': autotimes,
            'TEST': test,
            'GPT4TS': gpt4ts,
            'aLLM4TS': allm4ts,
            # Ours
            'LLM4Air':llm4air,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
