import re
from flask import Flask
from flask import request, jsonify

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BackendT5:

    def __init__(self, model_path='./t5'):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def rephrase(self, text,
                 do_sample=True,
                 max_length=256,
                 top_k=120,
                 top_p=0.98,
                 early_stopping=True,
                 num_return_sequences=5):
        paraphrased_texts = []

        text = "paraphrase:" + text
        encoding = self.tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        beam_outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_masks,
                                           do_sample=do_sample, max_length=max_length, top_k=top_k,
                                           top_p=top_p, early_stopping=early_stopping,
                                           num_return_sequences=num_return_sequences)

        for line in beam_outputs:
            paraphrased_texts.append(
                re.sub('[SEP]','',(re.sub('[CLS]','',self.tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)))))

        return paraphrased_texts



# app = Flask(__name__)
# backT5 = BackendT5()


# @app.route('/paraphrase', methods=['POST'])
# def paraphrase():
#     if request.method == 'POST':
#         text = request.args.get('text')
#         print("Got text:", text)
#         prephrased_text = backT5.rephrase(text=text)
#         print(prephrased_text)
#         return jsonify(prephrased_text)
# 

# if __name__ == '__main__': app.run(debug=True, host="0.0.0.0")
