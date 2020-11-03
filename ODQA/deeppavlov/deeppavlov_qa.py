!python -m deeppavlov install en_odqa_infer_wiki

from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

odqa = build_model(configs.odqa.en_odqa_infer_wiki, load_trained=True)
result = odqa(['Who is the current prime minister of India'])
print(result)