from Common.Data.Database import RAGDatabase
from Common.OpenAIProviders.OpenAIEmbeddingProvider import OpenAIEmbeddingProvider
from Common.OpenAIProviders.OpenAILLMProvider import OpenAILLMProvider
from Common.OpenAIProviders.OpenAILegacyAPILLMProvider import OpenAILegacyAPILLMProvider, HuggingfaceModelTokenAdaptor
import json
import os
import argparse
import logging
from flask import Flask, request, Response, stream_with_context
from flask_restx import Api, Resource, reqparse, fields, abort, inputs
from cfenv import AppEnv
from werkzeug.datastructures import FileStorage

from RAGDataProvider import RAGDataProvider

app = Flask(__name__)
api = Api(app, version='1.0', title='RAG API', description='API for RAG (Retrieval-Augmented Generation)')

# Setup logging
logging.basicConfig(level=logging.INFO)

RAG_PROVIDER: RAGDataProvider = None

#### API Routes
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('files', type=FileStorage, location='files', action="append", required=True, help='One or more Markdown Documents to upload')
upload_parser.add_argument('token_chunk_size', type=int, default=512, help='Token chunk size used for chunking')
upload_parser.add_argument('topic_display_name', required=True, help='Topic display name')
upload_parser.add_argument('dry_run_level', type=int, default=0, help='Dry run level. 0 - chunk and insert, 1 - text chunk with embeddings returned, 2 - text chunk return only')
@api.route('/upload_files')
class UploadFiles(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args(strict=True)
        files = args['files']
        token_chunk_size = args['token_chunk_size']
        topic_display_name = args['topic_display_name']
        dry_run_level = args['dry_run_level']

        try:
            if dry_run_level > 0:
                docs_chunk_list = RAG_PROVIDER.chunk_run(
                    markdown_files=files,
                    topic_display_name=topic_display_name,
                    token_chunk_size=int(token_chunk_size),
                    output_embeddings=True if dry_run_level == 1 else False
                )
                result = [{ filename: [chunk.to_dict() for chunk in doc_chunks]} for filename, doc_chunks in docs_chunk_list]
                return {'document_chunks': result}
            else:
                RAG_PROVIDER.chunk_insert_into_database(
                    markdown_files=files,
                    topic_display_name=topic_display_name,
                    token_chunk_size=int(token_chunk_size)
                )
            return {'message': 'Embeddings generated for input file and stored successfully'}
        except Exception as e:
            api.abort(500, str(e))

#########


kb_parser = reqparse.RequestParser()
kb_parser.add_argument('topic_display_name', required=True, help='Topic display name')
kb_parser.add_argument('vector_size', type=int, default=768, help='Vector size of the embedding model output.')
kb_parser.add_argument('topic_domain', required=True, help='Topic domain - Preferable a single world describing the subject domain, e.g. science, medicine etc.')
kb_parser.add_argument('prompt_template', default=RAGDataProvider.get_default_prompt_template(), help=f"A default prompt template to be used. Template variables are to be surrounded in curly braces with name, i.e., {{variable_name}}. Supported variable_names are [{', '.join(RAGDataProvider.get_supported_prompt_template_variables())}].")
kb_parser.add_argument('context_learning', default='[]', help='Context learning. Allows for multi-shot learning as part of prompt.')
@api.route('/create_knowledge_base')
class CreateKnowledgeBase(Resource):
    @api.expect(kb_parser)
    def post(self):
        args = kb_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']
        vector_size = args['vector_size']
        domain = args['topic_domain']
        context_learning_str = args['context_learning']
        prompt_template = args['prompt_template']

        try:
            context_learning = json.loads(context_learning_str)
            is_valid_context_learning = all(
                isinstance(item, dict) and 'role' in item and 'content' in item
                for item in context_learning
            )

            if not is_valid_context_learning:
                raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")
        except json.JSONDecodeError:
            api.abort(400, 'Malformed JSON in context_learning')

        try:
            message = RAG_PROVIDER.create_knowledgebase(
                topic_display_name=topic_display_name,
                vector_size=vector_size,
                topic_domain=domain,
                prompt_template=prompt_template,
                context_learning=context_learning
            )
            return {'message': message}
        except Exception as e:
            output = f"Knowledge base {topic_display_name} creation failed. "
            api.abort(500, output + str(e))

#######
@api.route('/list_knowledge_bases')
class ListKnowledgeBases(Resource):
    @api.param('topic_display_name_only','Return only topic display names', type=bool, default=False)
    def get(self):
        topic_display_name_only = request.args.get('topic_display_name_only', type=inputs.boolean, default=False)

        try:
            knowledge_bases = RAG_PROVIDER.get_all_knowledgebases()

            result = None
            if topic_display_name_only:
                result = [kb.topic_display_name for kb in knowledge_bases]
            else:
                result = [kb.to_dict() for kb in knowledge_bases]

            return {'knowledge_bases': result}
        except Exception as e:
            api.abort(500, str(e))

#########
kb_delete_parser = reqparse.RequestParser()
kb_delete_parser.add_argument('topic_display_name', required=True, help='Topic display name')
@api.route('/delete_knowledge_base')
class DeleteKnowledgeBase(Resource):
    @api.expect(kb_delete_parser)
    def post(self):
        args = kb_delete_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']

        try:
            RAG_PROVIDER.delete_knowledge_base(topic_display_name=topic_display_name)
            return {'message': f"Knowledge base {topic_display_name} deleted successfully"}
        except Exception as e:
            api.abort(500, str(e))

#########
get_context_parser = reqparse.RequestParser()
get_context_parser.add_argument('topic_display_name', required=True, help='Topic display name')
@api.route('/get_context_learning')
class GetContextLearning(Resource):
    @api.expect(get_context_parser)
    def get(self):
        args = get_context_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']

        try:
            context = RAG_PROVIDER.get_knowledge_base_context_learning(topic_display_name)
            if context is None:
                return {'message': 'No learning context found.'}, 404
            return context
        except Exception as e:
            api.abort(500, str(e))

#########
context_parser = reqparse.RequestParser()
context_parser.add_argument('topic_display_name', required=True, help='Topic display name')
context_parser.add_argument('context_learning', required=True, help='Context learning. Allows for multi-shot learning as part of prompt.')
@api.route('/update_context_learning')
class UpdateContextLearning(Resource):
    @api.expect(context_parser)
    def post(self):
        args = context_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']
        context_learning_str = args['context_learning']

        try:
            context_learning = json.loads(context_learning_str)

            is_valid_context_learning = all(
                isinstance(item, dict) and 'role' in item and 'content' in item
                for item in context_learning
            )

            if not is_valid_context_learning:
                raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")

        except json.JSONDecodeError:
            api.abort(400, 'Malformed JSON in context_learning')

        try:
            RAG_PROVIDER.update_knowledge_base_context_learning(
                topic_display_name=topic_display_name,
                new_context_learning=context_learning
            )
            return {'message': f"Learning context for {topic_display_name} updated successfully"}
        except Exception as e:
            api.abort(500, str(e))

#########
prompt_parser = reqparse.RequestParser()
prompt_parser.add_argument('topic_display_name', required=True, help='Topic display name')
prompt_parser.add_argument('prompt_template', default=RAGDataProvider.get_default_prompt_template(),help=f"A default prompt template to be used. Template variables are to be surrounded in curly braces with name, i.e., {{variable_name}}. Supported variable_names are [{', '.join(RAGDataProvider.get_supported_prompt_template_variables())}].")
@api.route('/update_prompt_template')
class UpdatePromptTemplate(Resource):
    @api.expect(prompt_parser)
    def post(self):
        args = prompt_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']
        prompt_template = args['prompt_template']

        try:
            RAG_PROVIDER.update_prompt_template(
                topic_display_name=topic_display_name,
                new_prompt_template=prompt_template
            )
            return {'message': f"Prompt template for {topic_display_name} updated successfully"}
        except Exception as e:
            logging.error(f"An error occurred while updating the prompt template: {e}")
            api.abort(500, str(e))

#########
template_parser = reqparse.RequestParser()
template_parser.add_argument('topic_display_name', required=True, help='Topic display name')
@api.route('/get_prompt_template')
class GetPromptTemplate(Resource):
    @api.expect(template_parser)
    def get(self):
        args = template_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']

        try:
            prompt_template = RAG_PROVIDER.get_prompt_template(topic_display_name)
            if prompt_template is None:
                api.abort(404, f"No prompt template found for topic display name: {topic_display_name}")
            return {'prompt_template': prompt_template}
        except Exception as e:
            logging.error(f"An error occurred while retrieving the prompt template: {e}")
            api.abort(500, str(e))

#########
kb_clear_parser = reqparse.RequestParser()
kb_clear_parser.add_argument('topic_display_name', required=True, help='Topic display name')
@api.route('/clear_embeddings')
class ClearEmbeddings(Resource):
    @api.expect(kb_clear_parser)
    def post(self):
        args = kb_clear_parser.parse_args(strict=True)
        topic_display_name = args['topic_display_name']

        try:
            deleted_count = RAG_PROVIDER.clear_knowledgebase_embeddings(topic_display_name=topic_display_name)
            return {'message': f'All embeddings cleared for {topic_display_name}.', 'deleted_rows': deleted_count}
        except Exception as e:
            api.abort(500, str(e))


#########
query_embed_content_parser = reqparse.RequestParser()
query_embed_content_parser.add_argument('query', required=True, help='The User query')
query_embed_content_parser.add_argument('topic_display_name', required=True, help='Topic display name')
query_embed_content_parser.add_argument('return_value_mode', required=False, type=int, default = 0, help='The values to return. 0 for all content+embedding, 1 for content, 2 for embedding')
@api.route('/get_embedding_content')
class GetEmbeddingContent(Resource):
    @api.expect(query_embed_content_parser)
    def post(self):
        args = query_embed_content_parser.parse_args(strict=True)
        query = args['query']
        topic_display_name = args['topic_display_name']
        return_value_mode = args['return_value_mode']

        try:
            results = RAG_PROVIDER.get_embedding_content(query=query, 
                                                         topic_display_name=topic_display_name)    
            
            statistics = RAG_PROVIDER.gather_statistics(chunks=results,
                                                        query=query, 
                                                        topic_display_name=topic_display_name)

            return_value = {}

            if return_value_mode == 0:
                return_value =  {'content': [ result.to_dict() for result in results] }
            elif return_value_mode == 1:
                return_value =  {'content': [ {k:v for k,v in result.to_dict().items() if k != "embedding"} for result in results] }
            else:
                return_value =  {'content': [ {k:v for k,v in result.to_dict().items() if k != "content"} for result in results] }
            
            return_value.update({'statistics': statistics})

            return return_value
        except Exception as e:
            api.abort(500, str(e)) 
        
#########
query_parser = reqparse.RequestParser()
query_parser.add_argument('query', required=True, help='The User query')
query_parser.add_argument('topic_display_name', required=True, help='Topic display name')
query_parser.add_argument('do_lost_in_middle_reorder', type=inputs.boolean, default=False, help='Perform lost in middle reordering')
query_parser.add_argument('override_context_learning', default=None, help='Override context learning. This will do a one time overwrite of any context learning that is stored with the knowledge base.')
query_parser.add_argument('stream', type=inputs.boolean, default=False, help='Initiate a streaming response')
@api.route('/respond_to_user_query')
class RespondToUserQuery(Resource):
    @api.expect(query_parser)
    def post(self):
        args = query_parser.parse_args(strict=True)
        query = args['query']
        topic_display_name = args['topic_display_name']
        do_lost_in_middle_reorder = args['do_lost_in_middle_reorder']
        override_context_learning_str = args['override_context_learning']
        stream = args['stream']

        context_learning = None
        if override_context_learning_str:
            logging.debug(override_context_learning_str)
            try:
                context_learning = json.loads(override_context_learning_str)

                if context_learning:
                    is_valid_context_learning = all(
                        isinstance(item, dict) and 'role' in item and 'content' in item
                        for item in context_learning
                    )
                    logging.debug(f"Is Valid json? {is_valid_context_learning}")
                    if not is_valid_context_learning:
                        logging.info("Context Invalid")
                        raise ValueError("Context Learning must be a list of dictionaries with each dictionary having keys, role and content with associated string content")
            except json.JSONDecodeError as e:
                logging.info(e)
                api.abort(400, f'Malformed JSON in context_learning: {str(e)}')

        def process_query():
            results = RAG_PROVIDER.respond_to_user_query(
                                                        query=query,
                                                        topic_display_name=topic_display_name,
                                                        override_context_learning=context_learning,
                                                        lost_in_middle_reorder=do_lost_in_middle_reorder,
                                                        stream=stream
                                                        )
            return results

        if stream:
            return Response(stream_with_context(process_query()), mimetype='text/event-stream')
        else:
            return process_query()

###########################
def initialize_and_start_service(args):
    """Initialize the application configuration from command line arguments and environment variables."""
    logging.basicConfig(level=logging.INFO)

    cf_env = AppEnv()

    # Database configuration
    database_url = args.database if args.database else cf_env.get_service(label='postgres').credentials['jdbcUrl']
    database = RAGDatabase(database_url)

    # API Client configuration
    api_base = args.api_base if args.api_base else \
                        cf_env.get_service(name='llm-config').credentials['api_base'] if cf_env.get_service(name='llm-config') and 'api_base' in cf_env.get_service(name='llm-config').credentials else \
                        cf_env.get_service(label='genai-service').credentials['api_base'] if cf_env.get_service(label='genai-service') else None
    
    api_key = args.api_key if args.api_key else \
                        cf_env.get_service(name='llm-config').credentials['api_key'] if cf_env.get_service(name='llm-config') and 'api_key' in cf_env.get_service(name='llm-config').credentials else \
                        cf_env.get_service(label='genai-service').credentials['api_key'] if cf_env.get_service(label='genai-service') else None
    
    if api_base is None or api_key is None:
        raise ValueError("API Base and API Key must be specified for the Text Generation Model")

    embedding_api_base = args.embedding_api_base if args.embedding_api_base else \
                        cf_env.get_service(name='embedding-service').credentials['api_base'] if cf_env.get_service(name='embedding-service') else \
                        cf_env.get_service(label='genai-service').credentials['api_base'] if cf_env.get_service(label='genai-service') else api_base
    
    embedding_api_key = args.embedding_api_key if args.embedding_api_key else \
                        cf_env.get_service(name='embedding-service').credentials['api_key'] if cf_env.get_service(name='embedding-service') else \
                        cf_env.get_service(label='genai-service').credentials['api_key'] if cf_env.get_service(label='genai-service') else api_key
    
    if embedding_api_base is None or embedding_api_key is None:
        raise ValueError("API Base and API Key must be specified for the Embedding Generation Model")
    
    use_legacy_openaiapi = bool(cf_env.get_service(name='llm-config').credentials['use_legacy_openaiapi']) if cf_env.get_service(name='llm-config') and 'use_legacy_openaiapi' in cf_env.get_service(name='llm-config').credentials else False
    hf_token = cf_env.get_service(name='llm-config').credentials['huggingface_token'] if cf_env.get_service(name='llm-config')  and 'huggingface_token' in cf_env.get_service(name='llm-config').credentials else ""

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Model and chunk sizes
    embed_model_name = args.embedding_model if args.embedding_model else os.environ.get("EMBED_MODEL", "hkunlp/instructor-xl")
    is_instructor_model = args.embed_model_is_instructor if args.embed_model_is_instructor else bool(os.environ.get("EMBED_MODEL_IS_INSTRUCTOR", "true").lower()=="true")        
    oaiEmbeddingProvider= OpenAIEmbeddingProvider(
                                            api_base=embedding_api_base,
                                            api_key=embedding_api_key,
                                            embed_model_name=embed_model_name,
                                            is_instructor_model=is_instructor_model
                                            )
    
    llm_model_name = args.llm_model if args.llm_model else os.environ.get("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    oai_llm = None
    if use_legacy_openaiapi:
        oai_llm = OpenAILegacyAPILLMProvider(api_base=api_base,
                                             api_key=api_key,
                                             llm_model_name=llm_model_name,
                                             temperature=0.0,
                                             model_token_adaptor=HuggingfaceModelTokenAdaptor(llm_model_name=llm_model_name, huggingface_token=hf_token)
                                            )
    else:
        oai_llm = OpenAILLMProvider(api_base=api_base,
                                    api_key=api_key,
                                    llm_model_name=llm_model_name,
                                    temperature=0.0
                                    )
                                    
    global RAG_PROVIDER
    RAG_PROVIDER = RAGDataProvider(
                                    database=database,
                                    oai_llm=oai_llm,
                                    oai_embed=oaiEmbeddingProvider
                                  )
    
    # Start the Flask application
    app.run(host=args.bind_ip, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-e", "--embedding_model", help="Model name for embeddings")
    parser.add_argument("-m", "--llm_model", help="Model name for text generation")
    parser.add_argument("-i", "--embed_model_is_instructor", type=bool, help="Model requires instruction")
    parser.add_argument("-s", "--api_base", help="Base URL for the OpenAI API. If not specified, search: user-provided-service named llm-config, otherwise GenAI Tile configuration will be used.")
    parser.add_argument("-a", "--api_key", help="API key for the OpenAI API. If not specified, search: user-provided-service named llm-config, otherwise GenAI Tile configuration will be used.")
    parser.add_argument("-g", "--embedding_api_base", help="Base URL for the OpenAI API for embedding generation.If not specified, search: user-provided-service named embedding-service, otherwise api_base will be used.")
    parser.add_argument("-y", "--embedding_api_key", help="API key for the OpenAI API for embedding generation. If not specified, search: user-provided-service named embedding-service, otherwise api_key will be used.")
    parser.add_argument("-d", "--database", help="Database connection string")
    parser.add_argument("-b", "--bind_ip", default="0.0.0.0", help="IP address to bind")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    initialize_and_start_service(args)