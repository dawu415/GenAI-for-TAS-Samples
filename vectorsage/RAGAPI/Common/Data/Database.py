from psycopg import connect, Connection
from psycopg.sql import SQL, Identifier, Literal, Placeholder
from psycopg.rows import class_row
from dataclasses import dataclass, fields, field, astuple, asdict
from dataclasses_json import dataclass_json, config
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Any, Type
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

### Helper Methods
def _parse_jdbc_connstring(jdbc_url):
    parsed_url = urlparse(jdbc_url[5:])  # Strip 'jdbc:'
    params = dict(param.split('=') for param in parsed_url.query.split('&'))
    # Construct psycopg connection string
    return f"dbname='{parsed_url.path[1:]}' host='{parsed_url.hostname}' port='{parsed_url.port}' user='{params['user']}' password='{params['password']}'"

### Data Classes
@dataclass_json
@dataclass
class KnowledgeBase:
    topic_display_name: str
    schema_table_name: str
    topic_domain: str
    prompt_template: str
    context_learning: List[Dict[str, Any]]
    id: int = None
    
    def __str__(self) -> str:
        return self.topic_display_name
    
    @classmethod
    def class_name(cls) -> str:
        return "knowledgebase"

    @classmethod
    def generate_create_table_sql(cls, schema: str):
        return SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        topic_display_name TEXT,
                        schema_table_name TEXT,
                        topic_domain TEXT,
                        prompt_template TEXT,
                        context_learning JSONB not null default '[]'::jsonb
                    );
                    """).format(Identifier(schema, KnowledgeBase.class_name()))

    @classmethod
    def generate_update_table_sql(cls, schema: str, table_name: str, existing_columns: List[str]):
        def get_column_type(field: Type) -> str:
            type_mapping = {
                str: "text",
                int: "integer",
                float: "float",
                Dict: "JSONB",
                List[Dict[str, Any]]: "JSONB not null default '[]'::jsonb"
            }
            return type_mapping.get(field, "text")
        
        new_columns = []
        for field in fields(cls):
            if field.name not in existing_columns:
                column_type = get_column_type(field.type)
                new_columns.append({'name': field.name, 'type': column_type})

        sql = ""
        if new_columns:
            sql = cls.generate_add_columns_sql(table_name, new_columns, schema)
        return sql 
    
    @classmethod
    def generate_add_columns_sql(cls, table_name: str, columns: List[dict], schema: str = None):
        add_columns_sql = SQL(", ").join(
            [SQL("ADD COLUMN {column_name} {column_type}").format(
                column_name=Identifier(column['name']),
                column_type=SQL(column['type'])) for column in columns]
        )
        return SQL("ALTER TABLE {schema_table} {add_columns}").format(
            schema_table=Identifier(schema, table_name) if schema else Identifier(table_name),
            add_columns=add_columns_sql
        )
    
    @classmethod
    def generate_insert_sql(cls, schema: str):
        included_fields = [f.name for f in fields(cls) if f.name != "id"]
        placeholders = SQL(", ").join([SQL("%s") for _ in included_fields])
        table_name = Identifier(schema, cls.class_name())
        field_identifiers = SQL(", ").join(map(Identifier, included_fields))
        return SQL("INSERT INTO {table_name} ({fields}) VALUES ({placeholders})").format(
            table_name=table_name,
            fields=field_identifiers,
            placeholders=placeholders
        )

    @classmethod
    def generate_delete_sql(cls, schema:str, topic_display_name: str):
        return SQL(
                    "DELETE FROM {table_name} WHERE topic_display_name = {topic_display_name}"
                ).format(table_name = Identifier(schema,  KnowledgeBase.class_name()),
                         topic_display_name = Literal(topic_display_name) 
                        )

    @classmethod
    def generate_get_knowledgebase_sql(cls,  schema:str, topic_display_name: str=None):
        # Determine the fields to select, excluding 'id'
        field_list = SQL(', ').join(
                            [Identifier(field.name) for field in fields(KnowledgeBase) if field.name != "id"]
                        )

        # Base table name using the class method for dynamic table names
        table_name = Identifier(schema, KnowledgeBase.class_name())

        # Conditional WHERE clause based on 'topic_display_name'
        where_clause = SQL(
                            " WHERE topic_display_name = {}"
                          ).format(
                              Literal(topic_display_name
                                      )) if topic_display_name else SQL("")

        return SQL("SELECT {} FROM {}{}").format(field_list, table_name, where_clause)
    
    @classmethod
    def generate_get_context_learning_sql(cls, schema: str, topic_display_name: str):
        table_name = Identifier(schema, KnowledgeBase.class_name())
        return SQL("SELECT context_learning FROM {table_name} WHERE topic_display_name = {topic_display_name}").format(
            table_name=table_name,
            topic_display_name=Literal(topic_display_name)
        )

    @classmethod
    def generate_update_context_learning_sql(cls, schema: str, topic_display_name: str, new_context: List[Dict[str,Any]]):
        table_name = Identifier(schema, KnowledgeBase.class_name())
        context_json = json.dumps(new_context)  # Ensure the new context is properly encoded to JSON
        return SQL("UPDATE {table_name} SET context_learning = {new_context} WHERE topic_display_name = {topic_display_name}").format(
            table_name=table_name,
            new_context=json.dumps(context_json),
            topic_display_name=Literal(topic_display_name)
        )

    @classmethod
    def generate_update_prompt_template_sql(cls, schema: str, topic_display_name: str, prompt_template: str):
        table_name = Identifier(schema, KnowledgeBase.class_name())
        return SQL("UPDATE {table_name} SET prompt_template = {prompt_template} WHERE topic_display_name = {topic_display_name}").format(
            table_name=table_name,
            prompt_template=Literal(prompt_template),
            topic_display_name=Literal(topic_display_name)
        )

@dataclass_json
@dataclass
class KnowledgeBaseEmbeddingResult:
    content: str
    embedding: List[float] = None
    id: int = None
    cosine_similarity: float = None
    metadata: dict = field(default_factory=dict,         
                                   metadata=config(
                                                    encoder=lambda metadataDict: json.dumps(metadataDict),
                                                    decoder=lambda metadataJsonStr: json.loads(metadataJsonStr)
                                                  )
                                        )

    @classmethod
    def generate_get_vector_sql_cosine_similarity(cls, table_name: str, schema: str = None, search_terms:List[str] =[], returned_results: int = 10):
        fields = ['id', 'content', 'embedding', 'metadata', '1 - (embedding <=> CAST({value} AS vector)) AS cosine_similarity']
        field_list = SQL(', ').join(
            [SQL(field).format(value=Placeholder()) if 'CAST' in field else Identifier(field) for field in fields]
            )
        
        # If there are search terms, use them. If the search terms still fail, 
        # the sql will fall back to just only using embeddings. 
        where_clause = SQL("")
        if search_terms:
            keyword_conditions = SQL(" OR ").join(
                [SQL("EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'keywords') elem WHERE elem ILIKE %s)") for _ in search_terms] +
                [SQL("metadata->>'title' ILIKE %s") for _ in search_terms]
            ) 

            where_clause = SQL(
                """
                WHERE (
                    CASE
                        WHEN {keyword_conditions} THEN TRUE
                        ELSE FALSE
                    END
                    OR NOT EXISTS (
                        SELECT 1
                        FROM {table_name} AS subquery
                        WHERE {keyword_conditions}
                    )
                )
                """
            ).format(
                keyword_conditions=keyword_conditions,
                table_name=Identifier(schema, table_name) if schema else Identifier(table_name)
            ) 

        return SQL(
            """
            SELECT {fields}
            FROM {table_name}
            {where_clause}
            ORDER BY cosine_similarity DESC
            LIMIT {limit};
            """
        ).format(
            fields=field_list,
            table_name=Identifier(schema, table_name) if schema else Identifier(table_name),
            where_clause=where_clause,
            limit=Literal(returned_results)
        )


@dataclass_json
@dataclass
class KnowledgeBaseEmbedding:
    content: str
    embedding: List[float] = None
    id: int = None
    metadata: dict = field(default_factory=dict,         
                                   metadata=config(
                                                    encoder=lambda metadataDict: json.dumps(metadataDict),
                                                    decoder=lambda metadataJsonStr: json.loads(metadataJsonStr)
                                                  )
                                        )

    @classmethod
    def generate_create_table_sql(cls, schema: str, table_name:str, vector_size: int):
        return SQL( """
                    CREATE TABLE IF NOT EXISTS {schema_table} (
                        id SERIAL PRIMARY KEY,
                        content text,
                        embedding vector({vector_size}),
                        metadata JSONB
                    );
                    """).format(schema_table=Identifier(schema, table_name),
                                vector_size=Literal(vector_size))

    @classmethod
    def generate_drop_table_sql(cls, table_name:str, schema: str = None):
        return SQL(
                "DROP TABLE {schema_table};"
                ).format(
                    schema_table=Identifier(schema, table_name))
    
    @classmethod
    def generate_insert_vector_sql(cls, table_name: str, schema: str = None):
        included_fields = [f.name for f in fields(cls) if f.name != "id"]
        placeholders = SQL(", ").join([SQL(f"%({f})s") for f in included_fields])
        field_identifiers = SQL(", ").join(map(Identifier, included_fields))

        return SQL("INSERT INTO {table_name} ({fields}) VALUES ({placeholders})").format(
            table_name=Identifier(schema, table_name) if schema else Identifier(table_name),
            fields=field_identifiers,
            placeholders=placeholders
        )
    
    @classmethod
    def generate_delete_all_sql(cls, table_name: str, schema: str = None):
        return SQL("DELETE FROM {}").format(Identifier(schema, table_name) if schema else Identifier(table_name))

    @classmethod
    def generate_add_columns_sql(cls, table_name: str, columns: List[dict], schema: str = None):
        add_columns_sql = SQL(", ").join(
            [SQL("ADD COLUMN {column_name} {column_type}").format(
                column_name=Identifier(column['name']),
                column_type=SQL(column['type'])) for column in columns]
        )
        return SQL("ALTER TABLE {schema_table} {add_columns}").format(
            schema_table=Identifier(schema, table_name) if schema else Identifier(table_name),
            add_columns=add_columns_sql
        )
    
    @classmethod
    def generate_update_table_sql(cls, schema: str, table_name: str, existing_columns: List[str], vector_size: int = 768):
        def get_column_type(field: Type) -> str:
            type_mapping = {
                str: "text",
                int: "integer",
                float: "float",
                Dict: "JSONB",
                List[float]: "vector(%s)" % vector_size 
            }
            return type_mapping.get(field, "text")
        
        new_columns = []
        for field in fields(cls):
            if field.name not in existing_columns:
                column_type = get_column_type(field.type)
                new_columns.append({'name': field.name, 'type': column_type})

        sql = ""
        if new_columns:
            sql = cls.generate_add_columns_sql(table_name, new_columns, schema)
        return sql 

### Database Class
class RAGDatabase:
    def __init__(self, jdbc_connection_string):
        self.default_schema = "cf_rag"
        self.connection_string = _parse_jdbc_connstring(jdbc_connection_string)
        self._setup()

    def connect(self) -> Connection:
        return connect(self.connection_string)
    
    def _setup(self):
        try:
            with self.connect() as conn, conn.cursor() as cur:
                # Ensure vector extension is enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # We're putting our tables in the cf_rag schema
                cur.execute(
                            SQL("CREATE SCHEMA IF NOT EXISTS {};"
                            ).format(Identifier(self.default_schema)))

                # Create a knowledge base table that will hold the information of
                # knowledge bases that will point to a table that contains the 
                # embeddings and associated content
                cur.execute(KnowledgeBase.generate_create_table_sql(self.default_schema))
                conn.commit()

                # Update the Knowledgebase table - in case we have new updates
                self._update_knowledgebase_table(self.default_schema)
                # Update the Knowledgebase embedding tables - in case we have new updates.
                self._update_all_knowledgebase_embedding_tables()


        except Exception as e:
            logging.error(f"An error occurred while setting up the database on startup: {e}")
            raise e
    
    def create_knowledge_base(self, topic_display_name:str, table_name: str, topic_domain: str, prompt_template:str, context_learning:List[Dict[str, Any]]=None, vector_size:int = 768):
        try:
           # Generate the SQL statement and return the target row data that Knowledgebase contains
            knowledgebase_insertsql = KnowledgeBase.generate_insert_sql(self.default_schema)
            
            if context_learning == None:
                context_learning = []
            
            # Generate Knowledge Embedding Creation SQL
            kbembed_table_creation_sql = KnowledgeBaseEmbedding.generate_create_table_sql(self.default_schema, table_name, vector_size)
            knowledgebase_insertsql_with_data = (knowledgebase_insertsql,
                                                [topic_display_name, f"{self.default_schema}.{table_name}", topic_domain, prompt_template, json.dumps(context_learning)]
                                                )

            with self.connect() as conn, conn.cursor() as cur:
                # Create the table for storing embeddings
                cur.execute(kbembed_table_creation_sql)
                
                # Insert the knowledge base into the knowledge base table
                cur.execute(*knowledgebase_insertsql_with_data)
                conn.commit()
        except Exception as e:
            logging.error(f"An error occurred while creating knowledgebase: {e}")
            raise e

    def delete_knowledge_base(self, topic_display_name: str):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase contains
            kb = self.get_knowledge_base(topic_display_name=topic_display_name)
            schema_table_name = kb[0].schema_table_name

            kbembed_dropsql = KnowledgeBaseEmbedding.generate_drop_table_sql(table_name=schema_table_name.split('.')[-1],
                                                            schema=schema_table_name.split('.')[0] if '.' in schema_table_name else None)

            knowledgebase_deletesql = KnowledgeBase.generate_delete_sql(self.default_schema, topic_display_name)  
            with self.connect() as conn, conn.cursor() as cur:
                # Delete the Knowledge Base Embeddings
                cur.execute(kbembed_dropsql)
                # Delete the entry from the knowledge base
                cur.execute(knowledgebase_deletesql)
                conn.commit()
        except Exception as e:
            logging.error(f"An error occurred while deleting knowledgebase: {e}")
            raise e
        
    def _update_all_knowledgebase_embedding_tables(self):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase contains
            kblist = self.get_knowledge_base()

            for kb in kblist:
                self._update_knowledgebase_embedding_table(kb)

        except Exception as e:
            logging.error(f"An error occurred while updating all knowledgebases: {e}")
            raise e

    def _update_knowledgebase_embedding_table(self, kb: KnowledgeBase):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase embedding contains
            schema_table_name = kb.schema_table_name
            table_name = table_name=schema_table_name.split('.')[-1]
            schema = schema_table_name.split('.')[0] if '.' in schema_table_name else "public"

            with self.connect() as conn, conn.cursor() as cur:
                cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s;
                        """, (schema, table_name))
                
                existing_columns = {row[0] for row in cur.fetchall()}  

                kbembed_updatesql = KnowledgeBaseEmbedding.generate_update_table_sql(
                                                    table_name=schema_table_name.split('.')[-1],
                                                    schema=schema_table_name.split('.')[0] if '.' in schema_table_name else "public",
                                                    existing_columns=existing_columns
                                                    )
                if kbembed_updatesql:
                    cur.execute(kbembed_updatesql)
                    conn.commit()

        except Exception as e:
            logging.error(f"An error occurred while updating knowledgebase embedding table: {e}")
            raise e

    def _update_knowledgebase_table(self, schema: str):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase contains
            table_name= KnowledgeBase.class_name()
            schema = schema if schema else "public"

            with self.connect() as conn, conn.cursor() as cur:
                cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s;
                        """, (schema, table_name))
                
                existing_columns = {row[0] for row in cur.fetchall()}  

                kb_updatesql = KnowledgeBase.generate_update_table_sql(
                                                    table_name=table_name,
                                                    schema=schema,
                                                    existing_columns=existing_columns
                                                    )
                if kb_updatesql:
                    cur.execute(kb_updatesql)
                    conn.commit()

        except Exception as e:
            logging.error(f"An error occurred while updating knowledgebase table: {e}")
            raise e

    def get_knowledge_base(self, topic_display_name: str=None):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase contains
            # If topic_display_name is not specified, it will retrieve all rows.
            knowledgebase_selectsql = KnowledgeBase.generate_get_knowledgebase_sql(self.default_schema, topic_display_name)
            
            def parse_context_learning(record:KnowledgeBase):
                if isinstance(record.context_learning, str):
                    record.context_learning = json.loads(record.context_learning)
                return record

            with self.connect() as conn, conn.cursor(row_factory=class_row(KnowledgeBase)) as cur:
                kb = cur.execute(knowledgebase_selectsql).fetchall()
                kb = [parse_context_learning(record) for record in kb] # convert the context_learning to json
            return kb
        except Exception as e:
            logging.error(f"An error occurred while listing knowledgebase: {e}")
            raise e
        
    def delete_knowledge_base_embeddings(self, schema_table_name: str):
        try:
            # Generate the SQL statement for deleting all entries in the table
            delete_all_sql = KnowledgeBaseEmbedding.generate_delete_all_sql(
                                                table_name=schema_table_name.split('.')[-1],
                                                schema=schema_table_name.split('.')[0] if '.' in schema_table_name else None)
            
            with self.connect() as conn, conn.cursor() as cur:
                # Execute the delete command
                cur.execute(delete_all_sql)
                deleted_rows = cur.rowcount  # Get the number of rows affected
                conn.commit()  # Commit changes to ensure the delete operation is saved
            
            return deleted_rows
        except Exception as e:  # It's a good practice to catch more specific exceptions if possible
            logging.error(f"An error occurred while deleting contents of {schema_table_name}: {e}")
            raise e
    
    def insert_content_with_embeddings(self, docChunksList: List[Dict], schema_table_name: str):
        try:
            # Generate the SQL statement and insert row data 
            insert_vector_sql = KnowledgeBaseEmbedding.generate_insert_vector_sql(
                                                table_name=schema_table_name.split('.')[-1],
                                                schema=schema_table_name.split('.')[0] if '.' in schema_table_name else None
                                            )  
            with self.connect() as conn, conn.cursor() as cur:
                # Create the table for storing embeddings
                cur.executemany(insert_vector_sql, docChunksList)
                conn.commit()
        except Exception as e:
            logging.error(f"An error occurred while inserting content with embeddings: {e}")
            raise e
        
    def get_content_with_cosine_similarity(self, queryembedding: List[float], schema_table_name: str, search_terms: List[str] = [], results_to_return: int = 10):
        try:
            # Generate the SQL statement and return the target row data that KnowledgebaseEmbeddings 
            # has closest to the queryembedding using cosine similarity built into the Db
            content_selectsql = KnowledgeBaseEmbeddingResult.generate_get_vector_sql_cosine_similarity(
                                                   table_name=schema_table_name.split('.')[-1],
                                                   schema=schema_table_name.split('.')[0] if '.' in schema_table_name else None,
                                                   search_terms=search_terms,
                                                   returned_results=results_to_return
                                                )
            
            params = [queryembedding]
            if search_terms:
                kw_search_terms = [f'%{term}%' for term in search_terms]
                title_search_terms = [f'%{term}%' for term in search_terms]
                params += kw_search_terms + title_search_terms + kw_search_terms + title_search_terms # Adding twice for keyword_conditions and fallback

            kb = None
            with self.connect() as conn, conn.cursor(row_factory=class_row(KnowledgeBaseEmbeddingResult)) as cur:
                kb = cur.execute(content_selectsql, tuple(params)).fetchall()
            
            return kb
        except Exception as e:
            logging.error(f"An error occurred while getting Knowledgebase Embedding Content: {e}")
            raise e
        

    def get_context_learning(self, topic_display_name: str):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase Learning Context
            context_selectsql = KnowledgeBase.generate_get_context_learning_sql(self.default_schema, topic_display_name)
            
            with self.connect() as conn, conn.cursor() as cur:
                context_learning = cur.execute(context_selectsql).fetchone()

                if isinstance(context_learning, list):
                    context_learning = context_learning[0]
                
                if context_learning == None:
                    context_learning = []
                elif isinstance(context_learning, str):
                    context_learning = json.loads(context_learning)
   
            return context_learning
        except Exception as e:
            logging.error(f"An error occurred while getting Knowledgebase Learning Context: {e}")
            raise e
        
    def update_context_learning(self, topic_display_name: str, new_context_learning: List[Dict[str, Any]]):
        try:
            # Generate the SQL statement and return the target row data that Knowledgebase Learning Context
            context_updatesql = KnowledgeBase.generate_update_context_learning_sql(self.default_schema, 
                                                                                   topic_display_name,
                                                                                   new_context_learning)
            
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute(context_updatesql)
                conn.commit()
        except Exception as e:
            logging.error(f"An error occurred while getting Knowledgebase Learning Context: {e}")
            raise e 
    
    def update_prompt_template(self, topic_display_name: str, prompt_template: str):
        try:
            # Generate the SQL statement to update the prompt template
            update_sql = KnowledgeBase.generate_update_prompt_template_sql(
                schema=self.default_schema,
                topic_display_name=topic_display_name,
                prompt_template=prompt_template
            )
            
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute(update_sql)
                conn.commit()
        except Exception as e:
            logging.error(f"An error occurred while updating the prompt template: {e}")
            raise e
