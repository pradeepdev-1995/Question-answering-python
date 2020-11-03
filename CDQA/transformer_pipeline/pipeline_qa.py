from transformers import pipeline
qNa= pipeline("question-answering")
paragraph = ''' The number of lives claimed by the Covid-19 coronavirus in India escalated sharply to 640 on Wednesday morning, with the total tally of positive cases rapidly nearing the 20,000 mark. The Indian Medical Association (IMA) called off the White Alert protest of doctors after the association was assured by Union Home Minister Amit Shah that they would be provided security by government. Meanwhile, a civil aviation ministry employee tested coronavirus positive today after which the B wing of the ministry was sealed and sanitisation procedure was initiated. '''
ans = qNa({'question': 'How much a total number of cases will India reach in the near future?',
           'context': f'{paragraph}'})
print(ans)