import itertools
from Common.Data.Database import RAGDatabase
from Common.DocTransformers.DocTransformer import DocTransformPipeline, Document, DocChunk
from Common.DocTransformers.ChunkMerger import ChunkMerger
from Common.DocTransformers.ChunkReOrder import ChunkReOrder
from Common.DocTransformers.GatherStatistics import GatherStatistics
from Common.DocTransformers.KeywordsExtractor import KeywordsExtractor
from Common.DocTransformers.InstructorReRanker import InstructorReRanker
from Common.DocTransformers.TextChunker import ModelTokenizedTextChunker
from Common.DocTransformers.EmbeddingGenerator import EmbeddingGenerator, InstructorEmbeddingGenerator
from Common.OpenAIProviders.OpenAIEmbeddingProvider import OpenAIEmbeddingProvider
from Common.OpenAIProviders.OpenAILLMProvider import OpenAIProvider
from werkzeug.datastructures import FileStorage
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import string
import json
import numpy as np
import re


# Setup logging
logging.basicConfig(level=logging.INFO)

@dataclass
class RAGDataProvider:
    database: RAGDatabase
    oai_embed: OpenAIEmbeddingProvider
    oai_llm: OpenAIProvider
 
    max_results_to_retrieve: int = 30


    def chunk_run(self, 
                    markdown_files: List[FileStorage],
                    topic_display_name: str,
                    token_chunk_size: int= 128,
                    output_embeddings: bool=True
                    ):
        kb = self.database.get_knowledge_base(topic_display_name)
        topic_domain = kb[0].topic_domain
        embed_instruction = f"Represent this {topic_domain} document for retrieval:"

        try: 
            # Exclude any keywords that relate to the topic domain
            # We exclude these, since we already are working within that topic domain
            spaced_topic_name = topic_domain.replace("_"," ")
            excluded_keywords = [kw.lower() for kw in spaced_topic_name.split()]
            excluded_keywords.append(spaced_topic_name.lower())

            pipeline_actions = [
                ModelTokenizedTextChunker(model_tokenizer_path=self.oai_embed.model_name, token_chunk_size=token_chunk_size),
                KeywordsExtractor(top_n=20, max_ngram_size=4, exclude_keywords=excluded_keywords),
            ]

            if output_embeddings:
                if self.oai_embed.is_instructor_model:
                    pipeline_actions.append(InstructorEmbeddingGenerator(embedding_provider=self.oai_embed, 
                                                                         embedding_instruction=embed_instruction, 
                                                                         use_metadata_with_embeddings=True))
                else:
                    pipeline_actions.append(EmbeddingGenerator(embedding_provider=self.oai_embed,
                                                               use_metadata_with_embeddings=True))
            
            document_chunks = DocTransformPipeline(
                                            pipeline_actions
                                         ).run_documents(
                                             [
                                                 Document(content=file.read().decode('utf-8'),
                                                          filename=Path(file.filename).stem
                                                         ) 
                                                                            for file in markdown_files
                                             ])
             
            return document_chunks
        except Exception as e:
            raise BufferError( f"Error occurred while processing files: {e}")

    def chunk_insert_into_database(self, 
                                           markdown_files: List[FileStorage],
                                           topic_display_name: str,
                                           token_chunk_size: int= 128):
        kb = self.database.get_knowledge_base(topic_display_name)
        topic_domain = kb[0].topic_domain
        target_embedding_table_name = kb[0].schema_table_name
        embed_instruction = f"Represent this {topic_domain} document for retrieval:"
        
        try:
            # Exclude any keywords that relate to the topic domain
            spaced_topic_name = topic_domain.replace("_"," ")
            excluded_keywords = [kw.lower() for kw in spaced_topic_name.split()]
            excluded_keywords.append(spaced_topic_name.lower)
            
            pipeline_actions = [
                ModelTokenizedTextChunker(model_tokenizer_path=self.oai_embed.model_name, token_chunk_size=token_chunk_size),
                KeywordsExtractor(top_n=20, max_ngram_size=4, exclude_keywords=excluded_keywords),
            ]

            if self.oai_embed.is_instructor_model:
                pipeline_actions.append(InstructorEmbeddingGenerator(embedding_provider=self.oai_embed, 
                                                                     embedding_instruction=embed_instruction, 
                                                                     use_metadata_with_embeddings=True))
            else:
                pipeline_actions.append(EmbeddingGenerator(embedding_provider=self.oai_embed,
                                                           use_metadata_with_embeddings=True))
            document_chunks = DocTransformPipeline(
                                            pipeline_actions
                                         ).run_documents(
                                             [
                                                 Document(content=file.read().decode('utf-8'),
                                                          filename=Path(file.filename).stem
                                                         ) 
                                                                            for file in markdown_files
                                             ])
             
            self.database.insert_content_with_embeddings(
                                docChunksList=[c.to_dict() for _, docChunks in document_chunks for c in docChunks], 
                                schema_table_name=target_embedding_table_name
                                )
        except Exception as e:
            raise BufferError( f"Error occurred while processing files: {e}")
     

    def create_knowledgebase(self,
                             topic_display_name:str,
                             vector_size: int,
                             topic_domain: str,
                             prompt_template: str,
                             context_learning: List[Dict[str,Any]] = None):
        
        # Make table name singular. Don't use caps/punctuations for easier maintainance
        table_name = ''.join([word.lower().translate(str.maketrans('','',string.punctuation)) 
                                        for word in topic_display_name.split(' ')])
        
        # Make sure domain is just a single word.
        domain = '_'.join(topic_domain.split())

        knowledge_base = self.database.get_knowledge_base(topic_display_name=topic_display_name)

        message = ""
        if len(knowledge_base) == 0:
            self.database.create_knowledge_base(topic_display_name=topic_display_name,
                                                table_name=table_name,
                                                topic_domain=domain,
                                                vector_size=vector_size,
                                                prompt_template=prompt_template,
                                                context_learning=context_learning)
            message = f"Knowledge base {topic_display_name} created successfully."
        else:
            message = f"The Knowledge base {topic_display_name} already exists."
                
        return message

    def get_all_knowledgebases(self):
        return self.database.get_knowledge_base()
    
    def delete_knowledge_base(self, topic_display_name: str):
        self.database.delete_knowledge_base(topic_display_name=topic_display_name)

    def get_knowledge_base_context_learning(self, topic_display_name: str):
        return self.database.get_context_learning(topic_display_name=topic_display_name)

    def update_knowledge_base_context_learning(self, 
                                               topic_display_name: str, 
                                               new_context_learning: List[Dict[str,Any]]):
        self.database.update_context_learning(topic_display_name=topic_display_name,
                                              new_context_learning=new_context_learning)

    def update_prompt_template(self, 
                               topic_display_name: str,
                               new_prompt_template:str):
        
        self.database.update_prompt_template(topic_display_name=topic_display_name,
                                             prompt_template=new_prompt_template)

    def get_prompt_template(self, topic_display_name: str):
        knowledge_base = self.database.get_knowledge_base(topic_display_name=topic_display_name)

        prompt_template = None
        if knowledge_base: 
            prompt_template = knowledge_base[0].prompt_template

        return prompt_template

    def clear_knowledgebase_embeddings(self, topic_display_name:str):
        # Fetching the table name using the topic display name
        knowledge_base = self.database.get_knowledge_base(topic_display_name)
        if knowledge_base is None or len(knowledge_base) == 0:
            raise Exception(f"{topic_display_name} Knowledge base not found.")
        
        schema_table_name = knowledge_base[0].schema_table_name

        # Deleting all embeddings from the table
        deleted_count = self.database.delete_knowledge_base_embeddings(schema_table_name)
        
        return deleted_count
    
    @classmethod
    def get_default_prompt_template(cls):
#         prompt_template = """You are an expert assistant using only the provided context. Follow these rules:
# 1. Use only the context to answer the query. Do not infer or add information.
# 2. If the context does not contain the information needed to answer the query, respond with "I don't have enough information to answer this query."
# 3. Be as detailed and helpful as possible and always give structured responses.  Respond as an expert without prompting the user to check the context. Avoid references to the user's query or previous answers.
# 4. Include relevant document and image links if mentioned in the context.
# 5. Do not expand acronyms unless provided in the context.
# Process the following query and context:
# ---
# Query: {query}
# Context:
# {context}
# ---
# Respond appropriately based on the rules above in a professional document in Markdown format.Do not tell the user you are using a provided context.
# """
            return  """You are an expert assistant answering queries using only the provided context. Follow these guidelines:
1. Use only information from the context to answer the query. Ignore irrelevant details and provide complete sentences and information.
2. Do not infer, assume, or add information not explicitly mentioned in the context. Remain objective and avoid personal opinions or biases.
3. If the context lacks necessary information, respond with "I don't have enough information to answer this query."
4. Provide detailed, helpful, and well-structured responses without telling the user to refer, check or read the provided context or mentioning that you are using provided context.
5. Do not expand acronyms unless specified and provided in the context.
6. Minimize references to previous answers or queries.
7. Do not repeat the user's query in your response.
8. If available in the context, include relevant document and image links in your response.
Process the following query and context:
---
Query: {query}
Context:
{context}
---
Respond appropriately based on the guidelines above, without mentioning them to the user.
"""
    @classmethod
    def get_supported_prompt_template_variables(cls):
        return ["query",
                "context"]

    def gather_statistics(  self,
                            chunks: List[DocChunk],
                            query: str,
                            topic_display_name:str = ""
                            ):
    
        prompt_template =RAGDataProvider.get_default_prompt_template()
        if topic_display_name:
            # Get the knowledge base
            kb = self.database.get_knowledge_base(topic_display_name=topic_display_name)[0]
            prompt_template = kb.prompt_template

        statistics = DocTransformPipeline(
                                # Cheat with the model name for now...
                                [GatherStatistics(model_tokenizer_path ="mistralai/" + self.oai_llm.model_name,
                                                  query=query,
                                                  prompt_template=prompt_template,
                                                  supported_template_variables=RAGDataProvider.get_supported_prompt_template_variables())]
                                ).run_docchunks(chunks)        

        return statistics
            


    def get_embedding_content(  self,
                                query: str, 
                                topic_display_name: str,
                                ):
        # Get the knowledge base
        kb = self.database.get_knowledge_base(topic_display_name=topic_display_name)[0]

        # Convert user query into an embedding with the instruction
        instruction = f"Represent this {kb.topic_domain} question for retrieving supporting documents:"
        query_embedding = self.oai_embed.get_embeddings_with_instructions(instruction, query)

        # Exclude any keywords that relate to the topic domain
        spaced_topic_name = kb.topic_domain.replace("_"," ")
        excluded_keywords = [kw.lower() for kw in spaced_topic_name.split()]
        excluded_keywords.append(spaced_topic_name.lower())
        query_keywords = KeywordsExtractor(
                                            max_ngram_size=3,
                                            exclude_monograms=True,
                                            exclude_subkeywords=False,
                                            exclude_keywords=excluded_keywords
                                          ).extract_keywords(query)

        logging.info(f"Query: {query}")
        logging.info(f"Keyswords found for query: {query_keywords}")

        # Use the result to search for similar content from the vector db
        results = self.database.get_content_with_cosine_similarity(queryembedding=query_embedding, 
                                                                   schema_table_name=kb.schema_table_name,
                                                                   results_to_return=self.max_results_to_retrieve,
                                                                   search_terms=query_keywords
                                                                  )
        
        # Define a set of postprocessing actions to perform on our retrieved results.
        pipeline_actions = [
            ChunkMerger(),
            InstructorEmbeddingGenerator(embedding_provider=self.oai_embed,
                                         embedding_instruction="Represent the {topic_domain} passage for clustering:",
                                         use_metadata_with_embeddings=False),
            InstructorReRanker(embedding_provider=self.oai_embed, query=query, topic_domain=kb.topic_domain)
        ]        

        document_chunks = DocTransformPipeline(
                                pipeline_actions
                                ).run_docchunks(
                                    [
                                        DocChunk(content=r.content,
                                                 embedding=r.embedding, 
                                                 metadata={**r.metadata, "cosine_similarity": r.cosine_similarity}
                                                ) 
                                                            for r in results
                                    ])
        
        return document_chunks       

    def respond_to_user_query(  self,
                                query: str, 
                                topic_display_name: str,
                                override_context_learning: List[Dict[str, Any]] = None, 
                                lost_in_middle_reorder: bool = False,
                                stream: bool=False
                                ):
        # Get the knowledge base
        kb = self.database.get_knowledge_base(topic_display_name=topic_display_name)[0]

        # Convert user query into an embedding with the instruction
        instruction = f"Represent this {kb.topic_domain} question for retrieving supporting documents:"
        query_embedding = self.oai_embed.get_embeddings_with_instructions(instruction, query)

        # Exclude any keywords that relate to the topic domain
        spaced_topic_name = kb.topic_domain.replace("_"," ")
        excluded_keywords = [kw.lower() for kw in spaced_topic_name.split()]
        excluded_keywords.append(spaced_topic_name.lower())
        query_keywords = KeywordsExtractor(
                                            max_ngram_size=3,
                                            exclude_monograms=True,
                                            exclude_subkeywords=False,
                                            exclude_keywords=excluded_keywords
                                          ).extract_keywords(query)

        # Use the result to search for similar content from the vector db
        results = self.database.get_content_with_cosine_similarity(queryembedding=query_embedding, 
                                                                   schema_table_name=kb.schema_table_name,
                                                                   results_to_return=self.max_results_to_retrieve,
                                                                   search_terms=query_keywords
                                                                  )
        
        # Define a set of postprocessing actions to perform on our retrieved results.
        pipeline_actions = [
            ChunkMerger(),
            InstructorEmbeddingGenerator(embedding_provider=self.oai_embed,
                                         embedding_instruction="Represent the {topic_domain} passage for clustering:",
                                         use_metadata_with_embeddings=False),
            InstructorReRanker(embedding_provider=self.oai_embed, query=query, topic_domain=kb.topic_domain)
        ]        

        if lost_in_middle_reorder:
            pipeline_actions.append(ChunkReOrder())

        document_chunks = DocTransformPipeline(
                                pipeline_actions
                                ).run_docchunks(
                                    [
                                        DocChunk(content=r.content,
                                                 embedding=r.embedding, 
                                                 metadata={**r.metadata, "cosine_similarity": r.cosine_similarity}
                                                ) 
                                                            for r in results
                                    ])    
        # Use the found content and form a prompt
        context = " ".join([re.sub(r'\n+', ' ', kbchunk.content) for kbchunk in document_chunks])

        # Create prompt variables
        # These must match what is available at get_supported_prompt_template_variables
        # Could not see a clearer way to do this than below...
        # Otherwise introspection via vars() could be done, but that's confusing magic.
        prompt_variables = {
            "query": query,
            "context": context
        }

        # Verify the prompt variables
        unsupported_keys = [k for k in prompt_variables.keys() if k not in self.get_supported_prompt_template_variables()]
        if unsupported_keys:
            raise Exception(f"Unsupported keys in prompt variables: {unsupported_keys}")

        # Format the prompt template
        prompt = kb.prompt_template.format_map(prompt_variables)

        # Set the context learning from knowledge base, otherwise, override it with provided list
        context_learning = kb.context_learning
        if override_context_learning:
            context_learning = override_context_learning

        # Get the context learning and concatenate with the prompt to form a new prompt
        message = context_learning + [{"role": "user", "content": prompt}]

        response = None

        print(f"Is Streaming: {stream}")
        if stream:
            response = self._do_stream_chat(message)
        else:
            # Send the new prompt to an LLM to generate a response.
            response = self.oai_llm.chat_completion(message)
        
        return response

    # Separate out the stream chat method since we're using yield.
    def _do_stream_chat(self, message: List[Dict[str,Any]]):
        streaming_response = self.oai_llm.stream_chat_completion(message)
        
        def process_message_content(content: str):
            # Escape newlines to prevent frontend issues
            escaped_content = content.replace("\n", "\\n")
            return f"data: {json.dumps(escaped_content)}\n\n".encode('utf-8')
        
        for message in streaming_response:
            if self.oai_llm.is_using_legacy_chat_api:
                content = message.choices[0].text
            else:
                content = message.choices[0].delta.content
            
            if content is not None:
                yield process_message_content(content)