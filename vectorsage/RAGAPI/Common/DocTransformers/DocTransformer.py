from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import re
import json

@dataclass_json
@dataclass
class DocChunk():
    content: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict,         
                                   metadata=config(
                                                    encoder=lambda metadataDict: json.dumps(metadataDict),
                                                    decoder=lambda metadataJsonStr: json.loads(metadataJsonStr)
                                                  )
                                        )

@dataclass_json 
@dataclass
class Document(DocChunk):
    def __init__(self, content: str, filename: str):
        super().__init__(content=content, metadata={"title": re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', ' ', filename)).strip()})
        self.filename = filename

class DocTransformer:
    pass

class DocTransformPipeline:
    pipeline: List[DocTransformer]

    def __init__(self, pipeline: List[DocTransformer]):
        self.pipeline = pipeline
    
    def run_documents(self, docs: List[Document]) -> List[Tuple[str,List[DocChunk]]]:
        output = []
        for doc in docs:
            parent_doc = doc
            transformed_doc = []
            for doc_transformer in self.pipeline:
                transformed_doc = doc_transformer(parent_doc, transformed_doc)

            output.append( (doc.filename, transformed_doc) )

        return output
    
    def run_docchunks(self, docChunks: List[DocChunk]) -> List[Tuple[str,List[DocChunk]]]:
        transformed_doc = docChunks
        for doc_transformer in self.pipeline:
            transformed_doc = doc_transformer(None, transformed_doc)

        return transformed_doc