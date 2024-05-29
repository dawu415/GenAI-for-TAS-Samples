from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk

from typing import List
import re
import yake

class KeywordsExtractor(DocTransformer):

    def __init__(self, top_n=20, max_ngram_size=4, deduplication_threshold=0.9, deduplication_algorithm="seqm", windowSize=1, language ="en", exclude_keywords: List[str]=[], exclude_subkeywords: bool = True, exclude_monograms:bool=True):
        self.kw_extractor = yake.KeywordExtractor(
            lan=language, 
            n=max_ngram_size, 
            dedupLim=deduplication_threshold, 
            dedupFunc=deduplication_algorithm, 
            windowsSize=windowSize, 
            top=top_n, 
            features=None)
        self.exclude_subkeywords = exclude_subkeywords
        self.exclude_keywords = exclude_keywords
        self.exclude_monograms = exclude_monograms

    def _remove_subwords_from_yake(self, keywords_scores):
        # Sort keywords by score in ascending order (lower score first)
        keywords_scores.sort(key=lambda x: x[1])
        
        # Extract only the keywords for easier processing
        keywords = [ks[0] for ks in keywords_scores]
        
        # Create a set to keep track of keywords to keep
        unique_keywords = set(keywords)
        
        # Iterate over the keywords
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                # If keyword[i] is a subword of keyword[j] or they overlap, remove the higher scored one
                if keywords[i] in keywords[j] or keywords[j] in keywords[i] or set(keywords[i].split()).intersection(set(keywords[j].split())):
                    unique_keywords.discard(keywords[j])
        
        # Prepare the final list of keywords and scores
        result = [ks[0] for ks in keywords_scores if ks[0] in unique_keywords]
        
        return result

    def _filter_keywords(self, keywords_list):
        new_kw_set = set()
        # Remove any keywords that are blank or starts with a non-alpha character
        # Convert all keywords to lower case and remove duplicates
        for kw in keywords_list:
            if kw not in self.exclude_keywords:
                if kw and kw[0].isalpha():
                    new_kw_set.add(kw.lower()) 
        
        return list(new_kw_set)

    def extract_keywords(self, content: str):
        # Extract words in backticks using regex
        highlighted_words = [ kw for kw in re.findall(r'`([^`]*)`', content.replace("\n"," ")) if ' ' not in kw or not kw ]

        # Extract words from markdown headers
        header_words = re.findall(r'^#+\s+(.*)', content, re.MULTILINE)

        keywords = self.kw_extractor.extract_keywords(content)

        if self.exclude_subkeywords:
            cleaned_kw = self._remove_subwords_from_yake(keywords)
        else:
            cleaned_kw = [kw[0] for kw in keywords]

        if self.exclude_monograms:
            cleaned_kw = [keyword for keyword in cleaned_kw if ' ' in keyword]
        
        final_keywords = self._filter_keywords(cleaned_kw + header_words + highlighted_words)

        return final_keywords

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        for text in texts:
            text.metadata.update({"keywords":self.extract_keywords(text.content) })

        return texts 

        
