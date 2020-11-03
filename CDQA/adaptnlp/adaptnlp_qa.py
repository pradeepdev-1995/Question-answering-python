from adaptnlp import EasyQuestionAnswering

## Example Query and Context 
query = "What has AdaptNLP"
context = "We have developed an open-source framework, AdaptNLP, that lowers the barrier to entry for practitioners to use these advanced capabilities. AdaptNLP is built atop two open-source libraries: Transformers (from Hugging Face) and Flair (from Zalando Research). AdaptNLP enables users to fine-tune language models for text classification, question answering, entity extraction, and part-of-speech tagging."
top_n = 5

## Load the QA module and run inference on results 
qa = EasyQuestionAnswering()
best_answer, best_n_answers = qa.predict_qa(query=query, context=context, n_best_size=top_n, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")

## Output top answer as well as top 5 answers
print(best_answer)
print(best_n_answers)