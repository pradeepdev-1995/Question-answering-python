import wikipedia as wiki
import pprint as pp

question = 'Who is the current US president'

results = wiki.search(question)
print("Wikipedia search results for our question:\n")
pp.pprint(results)

page = wiki.page(results[0])
text = page.content
print(f"\nThe {results[0]} Wikipedia article contains {len(text)} characters.")
