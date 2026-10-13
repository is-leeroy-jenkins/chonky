'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    Provides centralized runtime configuration for the Chonky Streamlit application.

    Purpose:
        Defines environment readers, application paths, logging database settings, provider
        credentials, model option lists, tab names, and session-state defaults used across the
        loading, processing, analysis, tokenization, embedding, and vector-database workflows.
        The module keeps import-time configuration safe by returning caller-provided defaults
        when optional environment variables are missing or malformed.
  </summary>
  ******************************************************************************************
'''
import os
from pathlib import Path

# -------------- APP-LEVEL UTILITIES -------------

def throw_if( name: str, value: object ) -> None:
	"""Validate a required argument or configuration value.

	Purpose:
		Provides a shared guard for required runtime values used by configuration helpers
		and source modules. The function raises a deterministic exception when a caller
		passes a falsy value for a required argument or setting name.

	Args:
		name: Name of the argument or configuration value being validated.
		value: Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is falsy.
	"""
	if not value:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def get_bool( name: str, default: bool = False ) -> bool:
	"""Read a Boolean environment variable.

	Purpose:
		Converts optional environment-variable text into a Boolean value used by Chonky
		configuration. Missing variables return the caller-provided default, while common
		truthy strings are normalized to ``True`` and all other defined values are treated
		as ``False``.

	Args:
		name: Environment variable name.
		default: Boolean value returned when the variable is missing or parsing fails.

	Returns:
		bool: Parsed Boolean value or the supplied default.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value is None else value.strip( ).lower( ) in (
				'1',
				'true',
				'yes',
				'y',
				'on'
		)
	except Exception:
		return default

def get_int( name: str, default: int ) -> int:
	"""Read an integer environment variable.

	Purpose:
		Parses optional integer configuration while preserving safe import behavior for
		local development, Streamlit deployment, and documentation generation. Missing,
		empty, or malformed values return the supplied default rather than interrupting
		module import.

	Args:
		name: Environment variable name.
		default: Integer value returned when the variable is missing or invalid.

	Returns:
		int: Parsed integer value or the supplied default.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else int( str( value ).strip( ) )
	except Exception:
		return default

def get_float( name: str, default: float ) -> float:
	"""Read a floating-point environment variable.

	Purpose:
		Parses optional numeric configuration for Chonky without making startup depend on
		perfect environment state. Missing, empty, or malformed values return the supplied
		default so downstream modules can import consistently.

	Args:
		name: Environment variable name.
		default: Floating-point value returned when the variable is missing or invalid.

	Returns:
		float: Parsed floating-point value or the supplied default.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else float( str( value ).strip( ) )
	except Exception:
		return default

def get_path( name: str, default: Path ) -> Path:
	"""Read a path environment variable.

	Purpose:
		Resolves optional filesystem configuration from the environment for Chonky paths
		such as logging locations and data directories. Missing or invalid variables fall
		back to the resolved default path to keep imports and MkDocs builds stable.

	Args:
		name: Environment variable name.
		default: Default path returned when the variable is missing or invalid.

	Returns:
		Path: Resolved environment path or resolved default path.
	"""
	try:
		throw_if( 'name', name )
		throw_if( 'default', default )
		value = os.getenv( name )
		return Path( value ).resolve( ) if value else default.resolve( )
	except Exception:
		return default.resolve( )

def get_text( name: str, default: str ) -> str:
	"""Read a text environment variable.

	Purpose:
		Centralizes optional text configuration for paths, provider settings, and runtime
		values. Missing or empty environment variables return the supplied default so the
		application and generated documentation can load without complete credentials.

	Args:
		name: Environment variable name.
		default: Default text returned when the variable is missing or empty.

	Returns:
		str: Environment value or supplied default text.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else str( value )
	except Exception:
		return default

# ------ PATHS ----------------------------------
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
ROOT_DIR = Path( __file__ ).resolve( ).parent
LOG_DIR: Path = get_path( 'LOG_DIR', ROOT_DIR / 'logging' )
LOG_PATH: str = get_text( 'LOG_PATH', str( LOG_DIR / 'Exceptions.db' ) )
LOG_FILE: str = get_text( 'LOG_FILE', 'Exceptions' )
ICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/chonky.png'
DB = str(get_path('CHONKY_DB_PATH', ROOT_DIR / 'stores' / 'sqlite' / 'data.db'))

# ------ CREDENTIALS ----------------------------------
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )

# ------ API KEYS ----------------------------------
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT_ID' )
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )
LANGSMITH_API_KEYS = os.getenv( 'LANGSMITH_API_KEYS' )
PINECONE_API_KEY = os.getenv( 'PINECONE_API_KEY' )
VERTEX_API_KEY = os.getenv( 'VERTEX_API_KEY' )
CHROMA_API_KEY = os.getenv( 'CHROMA_API_KEY' )
CHROMA_TENANT = os.getenv( 'CHROMA_TENANT' ) or os.getenv( 'CHROMA_TENET_ID' )
CHROMA_TENET_ID = os.getenv( 'CHROMA_TENET_ID' ) or CHROMA_TENANT
CHROMA_DATABASE = os.getenv( 'CHROMA_DATABASE' )

# ------- CONSTANTS ------------------------------
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:30px 0 30px 0;'></div>"

TABS = [ 'Loading', 'Processing', 'Analysis', 'Tokenization',
         'Embeddings', 'Database' ]

REQUIRED_CORPORA = [ 'brown', 'gutenberg', 'reuters', 'webtext', 'inaugural', 'state_union',
		'punkt', 'stopwords', ]

PROVIDERS = [ 'Foundry', 'OpenAI', 'Gemini', 'Groq' ]

GPT_MODELS = [ 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002' ]

GEMINI_MODELS = [ 'text-embedding-004', 'text-multilingual-embedding-002' ]

GROK_MODELS = [ 'nomic-embed-text-v1.5', 'text-embedding-3-small',
                'text-embedding-3-large', 'text-embedding-ada-002' ]

SESSION_STATE_DEFAULTS = {
		# -----------------------------
		# Ingestion
		# -----------------------------
		'documents': None,
		'raw_documents': None,
		'active_loader': None,
		'source_file_name': '',
		'document_name': '',
		'collection_name': '',
		# -----------------------------
		# Input
		# -----------------------------
		'raw_text': None,
		'raw_tokens': None,
		'raw_text_view': None,
		# -----------------------------
		# Processing
		# -----------------------------
		'parser': None,
		'processed_text': None,
		'processed_text_view': None,
		# -----------------------------
		# Performance
		# -----------------------------
		'start_time': None,
		'end_time': None,
		'total_time': None,
		# -----------------------------
		# Tokenization / Vocabulary
		# -----------------------------
		'tokens': None,
		'vocabulary': None,
		'token_counts': None,
		'df_synsets': None,
		# -----------------------------
		# SQLite / Excel
		# -----------------------------
		'active_table': None,
		# -----------------------------
		# Chunking
		# -----------------------------
		'lines': None,
		'chunks': None,
		'chunk_modes': None,
		'chunked_documents': None,
		# -----------------------------
		# Embeddings
		# -----------------------------
		'embedder': None,
		'embeddings': None,
		'embedding_provider': None,
		'embedding_model': None,
		'embedding_source': None,
		'embedding_documents': None,
		'df_embedding_input': None,
		'df_embedding_output': None,
		# -----------------------------
		# Retrieval / Search
		# -----------------------------
		'search_results': None,
		# -----------------------------
		# DataFrames
		# -----------------------------
		'df_frequency': None,
		'df_tables': None,
		'df_schema': None,
		'df_preview': None,
		'df_count': None,
		'df_chunks': None,
		# -----------------------------
		# Data
		# -----------------------------
		'data_connection': None,
		# -----------------------------
		# Sidebar / API Keys
		# -----------------------------
		'api_keys': {
				'openai': None,
				'groq': None,
				'google': None,
				'pinecone': None,
				'chroma': None,
				'google_credentials_path': None,
		},
		'chroma_api_key': '',
		'chroma_tenant': '',
		'chroma_database': '',
		'pinecone_index_name': '',
		'vector_store_provider': 'SQLiteVec',
		'vector_store_cloud_pinecone': False,
		# -----------------------------
		# XML Loader (explicit contract)
		# -----------------------------
		'xml_loader': None,
		'xml_documents': None,
		'xml_split_documents': None,
		'xml_tree_loaded': None,
		'xml_namespaces': None,
		'xml_xpath_results': None,
		# -----------------------------
		# WordNet Caches
		# -----------------------------
		'wordnet_synsets_sig': None,
		'df_wordnet_synsets': None,
		'df_wordnet_lemmas': None,
}

BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )

CHUNKABLE_LOADERS = {
		'TextLoader': [ 'chars', 'tokens' ],
		'CsvLoader': [ 'chars' ],
		'PdfLoader': [ 'chars' ],
		'ExcelLoader': [ 'chars' ],
		'WordLoader': [ 'chars' ],
		'MarkdownLoader': [ 'chars' ],
		'HtmlLoader': [ 'chars' ],
		'JsonLoader': [ 'chars' ],
		'PowerPointLoader': [ 'chars' ],
}

# ------------- API DEFINITIONS ------------------

ARXIV = r'''arXiv is a free distribution service and an open-access archive for nearly 2.4 million
		scholarly articles in the fields of physics, mathematics, computer science, quantitative
		biology, quantitative finance, statistics, electrical engineering and systems science, and
		economics. Materials on this site are not peer-reviewed by arXiv.
		
		https://docs.langchain.com/oss/python/integrations/retrievers/arxiv
'''

GOOGLE_DRIVE = r'''Google Drive is a file storage and synchronization service developed by Google
		
		https://docs.langchain.com/oss/python/integrations/retrievers/google_drive
'''

WIKIPEDIA = r'''A multilingual free online encyclopedia written and maintained by a community of
		volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing
		system called MediaWiki. Wikipedia is the largest and most-read reference work in history.
		
		https://docs.langchain.com/oss/python/integrations/retrievers/wikipedia
'''

PUBMED = r'''The National Center for Biotechnology Information, National Library of Medicine
		comprises more than 35 million citations for biomedical literature from MEDLINE,
		life science journals, and online books. Citations may include links to full text content
		from PubMed Central and publisher web sites.
		
		Key Features and Components:
		MEDLINE: The largest component, containing curated journal citations indexed with
		MeSH (Medical Subject Headings). PubMed Central (PMC): A  full-text archive of biomedical
		and life sciences journal literature, including peer-reviewed articles and preprints.
		Bookshelf: An archive of books, reports, and documents related to health and life sciences.
		Accessibility: PubMed does not host full text directly for all items but provides
		links to full-text articles through PMC or publisher websites
		
		https://docs.langchain.com/oss/python/integrations/retrievers/pubmed
'''
# ---------------- FOUNDRY CODE WORKSPACE ----------------
# Set the exact alias shown in the imported model's Models panel snippet.
FOUNDRY_EMBEDDING_MODEL = get_text('CHONKY_FOUNDRY_EMBEDDING_MODEL', '')
FOUNDRY_EMBED_BATCH_SIZE = max(1, get_int('CHONKY_FOUNDRY_EMBED_BATCH_SIZE', 32))
FOUNDRY_NLTK_DATA_DIR = get_text('CHONKY_NLTK_DATA_DIR', '')
VECTOR_DB_PATH = str(get_path('CHONKY_VECTOR_DB_PATH', ROOT_DIR / 'stores' / 'sqlite' / 'vectors.db'))
PINECONE_INDEX_NAME = get_text('PINECONE_INDEX_NAME', '')
