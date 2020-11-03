from txtai.embeddings import Embeddings
from txtai.extractor import Extractor

# Create embeddings model, backed by sentence-transformers & transformers
embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-mean-tokens"})

# Create extractor instance
extractor = Extractor(embeddings, "distilbert-base-cased-distilled-squad")

sections = ["Giants hit 3 HRs to down Dodgers",
            "Giants 5 Dodgers 4 final",
            "Dodgers drop Game 2 against the Giants, 5-4",
            "Blue Jays 2 Red Sox 1 final",
            "Red Sox lost to the Blue Jays, 2-1",
            "Blue Jays at Red Sox is over. Score: 2-1",
            "Phillies win over the Braves, 5-0",
            "Phillies 5 Braves 0 final",
            "Final: Braves lose to the Phillies in the series opener, 5-0",
            "Final score: Flyers 4 Lightning 1",
            "Flyers 4 Lightning 1 final",
            "Flyers win 4-1"]

# Add unique id to each section to assist with qa extraction
sections = [(uid, section) for uid, section in enumerate(sections)]

# Ad-hoc questions
question = "What hockey team won?"

print("----", question, "----")
print(extractor(sections, [(question, question, question, False)]))