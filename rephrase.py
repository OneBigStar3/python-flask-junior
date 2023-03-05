import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def changer(sentence, length, sentences):

	def set_seed(seed):
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	set_seed(42)


	tokenizer = T5Tokenizer.from_pretrained('t5-base')
	model = T5ForConditionalGeneration.from_pretrained('./t5_trained')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print ("device ",device)
	model = model.to(device)


	text =  "paraphrase: " + sentence + " </s>"


	max_len = length

	encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
	input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


	# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
	beam_outputs = model.generate(
		input_ids=input_ids, attention_mask=attention_masks,
		do_sample=True,
		max_length=length,
		top_k=120,
		top_p=0.98,
		early_stopping=True,
		num_return_sequences=sentences
	)


	maxi_len = 0
	final_outputs =[]
	for beam_output in beam_outputs:
		sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
		if sent.lower() != sentence.lower() and sent not in final_outputs and maxi_len<length:
			final_outputs.append(sent)
			maxi_len += 1

	return final_outputs

