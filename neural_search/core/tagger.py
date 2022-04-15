from transformers import pipeline

class NERTagger:

    def __init__(self,
                 model_name="wolfrage89/company_segment_ner",
                 entity_map={}):
        print("Loading NER model...")
        self.model = pipeline('ner', model_name)
        self.entity_map = {
            "B-ORG":"ORG",
            "B-SEG":"SEG",
            "B-SEGNUM":"SEGNUM"
        } if entity_map == {} else entity_map
        print("NER model loaded.")

    # TODO: Take real advantage of NER tags
    def predict(self, sentence):
        results = {}
        model_output = self.model(sentence)

        accumulate = ""
        current_class = None
        # start = 0
        # end = 0
        for item in model_output:
            if item['entity'].startswith("B"):
                if len(accumulate) >0:
                    # results.append({
                    #     current_class: {
                    #         "text": accumulate,
                    #         "start": start,
                    #         "end": end
                    #     }
                    # })
                    if current_class is not None:
                        results[current_class] = accumulate
                accumulate = item['word'].lstrip("Ġ")
                current_class = self.entity_map[item['entity']]
                # start=item['start']
                # end = item['end']
                
            else:
                if item['word'].startswith("Ġ"):
                    accumulate+=" "+item['word'].lstrip("Ġ")
                    
                else:
                    accumulate+=item['word']
                # end = item['end']

        # clear last cache
        if len(accumulate)>0:
            # results.append({
            #     current_class: {
            #         "text": accumulate,
            #         "start": start,
            #         "end": end
            #     }
            # })
            results[current_class] = accumulate

        return results
