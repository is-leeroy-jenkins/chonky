'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                embedders.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="embedders.py" company="Terry D. Eppler">

	     embedders.py
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
    Provides hosted embedding-provider wrappers for the Chonky document-processing workflow.

    Purpose:
        Defines OpenAI, Gemini, and Grok-compatible embedding classes used by the Tensor
        Embeddings tab and downstream vector-database workflow. The module validates input text,
        prepares provider-specific request payloads, stores raw provider responses on class
        instances for diagnostics, returns embedding vectors in a consistent list-based shape,
        and records wrapped provider or validation failures through the application logger.
  </summary>
  ******************************************************************************************
'''
from groq import Groq
from requests.models import Response

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import (EmbedContentConfig, HttpOptions)
if __package__:
	from . import config as cfg
else:
	import config as cfg
from typing import List, Optional, Any, Union, Dict
import tiktoken
import os
if __package__:
	from .boogr import Error, Logger
else:
	from boogr import Error, Logger
import requests

def throw_if( name: str, value: object ):
	"""Validate a required runtime argument.

	Purpose:
		Provides a lightweight guard for provider wrapper methods that require non-empty
		inputs before calling hosted embedding APIs. The function raises a deterministic
		``ValueError`` when a required value is missing.

	Args:
		name: Name of the argument being validated.
		value: Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is ``None``.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class GPT( ):
	"""OpenAI embedding provider wrapper.

	Purpose:
		Creates single and batch text embeddings through OpenAI embedding models for Chonky's
		semantic-analysis and vector-database pipeline. The class stores the active API key,
		client, model, response, and generated vectors on the instance so the Streamlit workflow
		can inspect provider state after an embedding operation.

	Attributes:
		web_options: Optional web option values retained for provider compatibility.
		input: Current text input or batch input sent to the provider.
		client: Initialized OpenAI client used for embedding requests.
		prompt: Prompt state retained for provider compatibility.
		response_format: Provider response-format state retained for compatibility.
		response: Raw OpenAI embedding response from the last request.
		embedding: Single embedding vector returned by the last single-text request.
		encoding_format: Output encoding format requested from the provider.
		dimensions: Optional vector dimensionality value retained for provider compatibility.
	"""
	
	web_options: Optional[ List[ str ] ]
	input: Optional[ List[ str ] ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	input: Optional[ List[ str ] ]
	response: Optional[ Response ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	
	def __init__( self, api_key: Optional[str] = None ):
		"""Initialize the OpenAI embedding wrapper.

		Purpose:
			Creates the OpenAI client from Chonky configuration and initializes request, response,
			model, encoding, and embedding state used by later single-text and batch embedding
			methods.
		"""
		super( ).__init__( )
		self.api_key = api_key or cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=self.api_key )
		self.encoding_format = None
		self.input = None
		self.model = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported OpenAI embedding models.

		Purpose:
			Exposes the OpenAI embedding model names used by Chonky UI controls and provider
			selection logic.

		Returns:
			List[str]: Supported OpenAI embedding model names.
		"""
		return [ 'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Return supported OpenAI embedding encoding formats.

		Purpose:
			Exposes provider-supported encoding format options for embedding responses so the UI
			and calling code can request float vectors or base64-encoded vectors.

		Returns:
			List[str]: Supported encoding format names.
		"""
		return [ 'float',
		         'base64' ]
	
	def create( self, text: str, model: str = 'text-embedding-3-small', format: str = 'float' ) -> \
			List[ float ]:
		"""Create a single OpenAI embedding vector.

		Purpose:
			Validates a single text input, stores the request model and encoding format on the
			instance, calls the OpenAI embeddings endpoint, captures the raw response, and returns
			the first embedding vector for downstream similarity search and vector persistence.

		Args:
			text: Text value to embed.
			model: OpenAI embedding model name.
			format: Embedding response encoding format.

		Returns:
			List[float]: Embedding vector for the supplied text.

		Raises:
			Error: Raised when validation or OpenAI embedding generation fails.
		"""
		try:
			throw_if( 'text', text )
			self.input = text
			self.model = model
			self.encoding_format = format
			self.response = self.client.embeddings.create( input=self.input, model=self.model,
				encoding_format=self.encoding_format )
			self.embedding = self.response.data[ 0 ].embedding
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'GPT'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			Logger( ).write( exception )
			raise exception
	
	def embed( self, texts: List[ str ], model: str = 'text-embedding-3-small' ) -> List[
		List[ float ] ]:
		"""Create batch OpenAI embedding vectors.

		Purpose:
			Validates and cleans a batch of text inputs, sends non-empty strings to the OpenAI
			embeddings endpoint in provider-safe batches, and returns vectors aligned to the
			cleaned batch for Chonky's embedding diagnostics and vector-database persistence.

		Args:
			texts: Text values to embed.
			model: OpenAI embedding model name.

		Returns:
			List[List[float]]: Embedding vectors for the cleaned text inputs.

		Raises:
			Error: Raised when validation or OpenAI batch embedding generation fails.
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			max_batch_size: int = 2048
			all_vectors: List[ List[ float ] ] = [ ]
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				self.response = self.client.embeddings.create(
					model=self.model,
					input=batch
				)
				all_vectors.extend( [ item.embedding for item in self.response.data ] )
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'GPT'
			exception.method = 'embed( self, text: str, model: str ) -> List[ List[ float ] ]'
			Logger( ).write( exception )
			raise exception
	
	def count_tokens( self, text: str, coding: str ) -> int | None:
		"""Count tokens for text using a tiktoken encoding.

		Purpose:
			Measures provider-tokenized text length for embedding readiness diagnostics and UI
			metrics. The method validates the text and encoding name, resolves the tiktoken
			encoding, and returns the number of token identifiers produced from the input text.

		Args:
			text: Text value to tokenize.
			coding: tiktoken encoding name.

		Returns:
			int | None: Token count for the supplied text.

		Raises:
			Error: Raised when validation or token encoding fails.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'coding', coding )
			_encoding = tiktoken.get_encoding( coding )
			_tokens = len( _encoding.encode( text ) )
			return _tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return public wrapper members.

		Purpose:
			Provides a stable list of public attributes and methods exposed by the OpenAI
			embedding wrapper for introspection, diagnostics, and documentation generation.

		Returns:
			List[str] | None: Public member names exposed by the wrapper.
		"""
		return [ 'create',
		         'api_key',
		         'client',
		         'model',
		         'count_tokens',
		         'path',
		         'model_options', ]

class Gemini( ):
	"""Google GenAI embedding provider wrapper.

	Purpose:
		Creates single and batch embeddings through the Google GenAI SDK for Chonky's hosted
		embedding workflow. The class stores Google project, location, client, task type,
		configuration, response, and vector state on the instance so embedding generation and
		diagnostics can share a consistent provider contract.

	Attributes:
		number: Optional numeric state retained for provider compatibility.
		api_key: Google or Vertex API key from Chonky configuration.
		project: Google Cloud project identifier from configuration.
		location: Google Cloud location from configuration.
		prompt: Prompt state retained for provider compatibility.
		credentials: Google application credentials path from configuration.
		model: Active Gemini embedding model name.
		response_format: Response format state retained for provider compatibility.
		dimensions: Requested embedding dimensionality.
		client: Initialized Google GenAI client.
		response: Raw provider response from the last embedding request.
		embedding: Single embedding vector from the last single-text request.
		embeddings: Batch embedding vectors from the last batch request.
		encoding_format: Encoding format state retained for provider compatibility.
		use_vertex: Vertex AI client flag.
		task_type: Gemini embedding task type.
		http_options: Google GenAI HTTP options.
		embedding_config: Google embedding request configuration.
		contents: Provider content values retained for compatibility.
		input_text: Current text input or batch input sent to the provider.
		file_path: Source file path state retained for compatibility.
		response_modalities: Response modality state retained for compatibility.
	"""
	number: Optional[ int ]
	api_key: Optional[ str ]
	project: Optional[ str ]
	location: Optional[ str ]
	prompt: Optional[ str ]
	credentials: Optional[ str ]
	model: Optional[ str ]
	response_format: Optional[ str ]
	dimensions: Optional[ int ]
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	use_vertex: Optional[ bool ]
	task_type: Optional[ str ]
	http_options: Optional[ HttpOptions ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ List[ str ] ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, version: str = 'v1alpha', use_ai: bool = False, dimensions: int = 768, api_key: Optional[str] = None ):
		"""Initialize the Gemini embedding wrapper.

		Purpose:
			Creates Google GenAI client state from Chonky configuration, stores API-version and
			Vertex selection settings, and initializes embedding response fields used by single
			and batch Gemini embedding methods.

		Args:
			version: Google GenAI API version used by HTTP options.
			use_ai: Flag indicating whether to initialize the client for Vertex AI.
			dimensions: Default requested embedding dimensionality.
		"""
		super( ).__init__( )
		self.api_key = api_key or cfg.GEMINI_API_KEY or cfg.GOOGLE_API_KEY or cfg.VERTEX_API_KEY
		self.project = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.location = cfg.GOOGLE_CLOUD_LOCATION
		self.credentials = cfg.GOOGLE_APPLICATION_CREDENTIALS
		self.model = None
		self.api_version = version
		self.use_vertex = use_ai
		self.dimensions = dimensions
		self.credentials = None
		self.http_options = HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=self.http_options )
		self.embedding = None
		self.embeddings = None
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.response_modalities = None
		self.embedding_config = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Return supported Gemini embedding models.

		Purpose:
			Exposes Gemini embedding model options for Chonky UI controls and provider selection.

		Returns:
			List[str] | None: Supported Gemini embedding model names.
		"""
		return [ 'gemini-embedding-001', 'text-multilingual-embedding-002' ]
	
	@property
	def task_options( self ) -> List[ str ] | None:
		"""Return supported Gemini embedding task types.

		Purpose:
			Exposes Gemini task-type values used to configure retrieval, document, and semantic
			similarity embedding requests.

		Returns:
			List[str] | None: Supported Gemini embedding task types.
		"""
		return [ 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY' ]
	
	def generate( self, text: str, model: str = 'gemini-embedding-001', dimensions: int = 768 ) -> \
			Optional[ List[ float ] ]:
		"""Create a single Gemini embedding vector.

		Purpose:
			Validates a single text input, configures the active Gemini model and dimensionality,
			calls the Google GenAI embedding endpoint, stores the provider response, and returns
			the first embedding vector for downstream Chonky semantic retrieval workflows.

		Args:
			text: Text value to embed.
			model: Gemini embedding model name.
			dimensions: Requested output dimensionality.

		Returns:
			Optional[List[float]]: Embedding vector for the supplied text.

		Raises:
			Error: Raised when validation or Gemini embedding generation fails.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.model = model
			self.dimensions = dimensions
			self.embedding_config = EmbedContentConfig( task_type=self.task_type )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			self.embedding = self.response.embeddings[ 0 ].values
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Gemini'
			exception.method = 'generate( self, text, model ) -> List[ float ]'
			Logger( ).write( exception )
			raise exception
	
	def embed( self, texts: List[ str ], task: str = 'RETRIEVAL_DOCUMENT',
			model: str = 'gemini-embedding-001', dimensions: int = 768 ) -> List[ List[ float ] ]:
		"""Create batch Gemini embedding vectors.

		Purpose:
			Validates and cleans a batch of text values, configures Gemini task and dimensionality
			settings, submits provider-safe batches to Google GenAI, and returns vectors for
			Chonky embedding diagnostics and vector persistence.

		Args:
			texts: Text values to embed.
			task: Gemini embedding task type.
			model: Gemini embedding model name.
			dimensions: Requested output dimensionality.

		Returns:
			List[List[float]]: Embedding vectors for the cleaned text inputs.

		Raises:
			Error: Raised when validation or Gemini batch embedding generation fails.
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			self.dimensions = dimensions
			self.task_type = task
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			all_vectors: List[ List[ float ] ] = [ ]
			max_batch_size: int = 250
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				
				self.embedding_config = EmbedContentConfig(
					task_type=self.task_type,
					output_dimensionality=self.dimensions
				)
				
				self.response = self.client.models.embed_content(
					model=self.model,
					contents=batch,
					config=self.embedding_config
				)
				
				all_vectors.extend(
					[ embedding.values for embedding in self.response.embeddings ]
				)
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Gemini'
			exception.method = 'embed( self, texts, task, model, dimensions ) -> List[ List[ float ] ]'
			Logger( ).write( exception )
			raise exception

class Grok( ):
	"""Grok-compatible embedding provider wrapper.

	Purpose:
		Provides the embedding-provider contract used by Chonky for Grok/Groq-labeled hosted
		embedding workflows. The class prepares authorization headers, request payloads, model
		state, raw responses, and returned vector lists for single-text and batch embedding
		requests.

	Attributes:
		content_type: HTTP content type used for embedding requests.
		api_key: Groq API key from Chonky configuration.
		authorization: Bearer authorization header value.
		prompt: Prompt state retained for provider compatibility.
		model: Active embedding model name.
		response_format: Response-format state retained for provider compatibility.
		contents: Text contents retained for batch provider compatibility.
		http_options: HTTP option state retained for provider compatibility.
		embedding: Embedding vector returned by the last single-text request.
		endpoint: Embedding endpoint URL used for requests.
	"""
	content_type: Optional[ str ]
	api_key: Optional[ str ]
	authorization: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	response_format: Optional[ Union[ str, Dict[ str, str ] ] ]
	contents: Optional[ List[ str ] ]
	http_options: Optional[ Dict[ str, Any ] ]
	embedding: Optional[ float ]
	endpoint: Optional[ str ]
	
	def __init__( self, model: str = 'text-embedding-3-small' ):
		"""Initialize the Grok-compatible embedding wrapper.

		Purpose:
			Loads provider configuration from Chonky settings, prepares authorization headers,
			creates the Groq client object, and initializes request, response, payload, encoding,
			and embedding state used by hosted embedding calls.

		Args:
			model: Default embedding model name.
		"""
		self.api_key = cfg.GROQ_API_KEY
		self.model = model
		self.content_type = 'application/json'
		self.authorization = f'Bearer {self.api_key}'
		self.headers = { 'Content-Type': self.content_type, 'Authorization': self.authorization }
		self.endpoint = os.getenv('CHONKY_GROQ_EMBEDDING_ENDPOINT', '').strip()
		self.client = Groq( api_key=self.api_key )
		self.contents = [ ]
		self.payload = None
		self.encoding_format = 'float'
		self.input_text = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Grok-compatible embedding models.

		Purpose:
			Exposes embedding model names for Chonky UI controls and provider selection where
			Grok/Groq-compatible embedding requests are available.

		Returns:
			List[str]: Supported embedding model names.
		"""
		return [ 'nomic-embed-text-v1.5',
		         'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Return supported Grok-compatible encoding formats.

		Purpose:
			Exposes embedding response encoding formats for Chonky UI controls and request
			configuration.

		Returns:
			List[str]: Supported encoding format names.
		"""
		return [ 'float', 'base64' ]
	
	def create( self, text: str, model: str = 'text-embedding-3-small', format: str = 'float' ) -> \
			List[ float ] | None:
		"""Create a single Grok-compatible embedding vector.

		Purpose:
			Validates a single text input, prepares an embedding request payload, submits the
			request to the configured embedding endpoint, stores the raw JSON response, and returns
			the first vector when the provider reports success.

		Args:
			text: Text value to embed.
			model: Embedding model name.
			format: Embedding response encoding format.

		Returns:
			List[float] | None: Embedding vector when the provider succeeds; otherwise ``None``.

		Raises:
			Error: Raised when validation, request construction, or provider execution fails.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model;
			self.encoding_format = format
			if not self.endpoint:
				raise RuntimeError('Groq embedding endpoint is not configured; use a verified provider endpoint.')
			self.payload = \
				{
						'input': self.input_text,
						'model': self.model,
						'encoding_format': self.encoding_format
				}
			
			_response = requests.post( url=self.endpoint, headers=self.headers, json=self.payload )
			if _response.status_code == 200:
				self.response = _response.json( )
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
				return self.embedding
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'embeddings'
			exception.cause = 'Grok'
			exception.method = 'create( self, text, model, format )'
			Logger( ).write( exception )
			raise exception
	
	def embed( self, texts: List[ str ], model: str = 'text-embedding-3-small' ) -> List[
		List[ float ] ]:
		"""Create batch Grok-compatible embedding vectors.

		Purpose:
			Validates and cleans a batch of text inputs, prepares provider authorization headers,
			submits provider-safe embedding batches to the configured endpoint, and returns vector
			lists for Chonky embedding diagnostics and vector-database persistence.

		Args:
			texts: Text values to embed.
			model: Embedding model name.

		Returns:
			List[List[float]]: Embedding vectors for the cleaned text inputs.

		Raises:
			Error: Raised when validation or batch provider execution fails.
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			self.model = model
			self.headers = \
				{
						'Authorization': f'Bearer {self.api_key}',
						'Content-Type': 'application/json',
				}
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			max_batch_size: int = 2048
			all_vectors: List[ List[ float ] ] = [ ]
			
			for i in range( 0, len( cleaned ), max_batch_size ):
				batch: List[ str ] = cleaned[ i:i + max_batch_size ]
				if not self.endpoint:
					raise RuntimeError('Groq embedding endpoint is not configured; use a verified provider endpoint.')
				self.payload = { 'model': model, 'input': batch }
				
				response = requests.post(
					url=self.endpoint,
					headers=self.headers,
					json=self.payload,
					timeout=30,
				)
				response.raise_for_status( )
				data = response.json( )
				all_vectors.extend( [ item[ 'embedding' ] for item in data[ 'data' ] ] )
			
			return all_vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'embeddings'
			exception.cause = 'Grok'
			exception.method = 'embed( self, texts, model )'
			Logger( ).write( exception )
			raise exception

class Foundry:
	"""Generate embeddings using an imported Foundry model (no external provider key)."""

	def __init__(self, model: Optional[str] = None, batch_size: Optional[int] = None):
		self.model = model if model is not None else cfg.FOUNDRY_EMBEDDING_MODEL
		self.batch_size = batch_size if batch_size is not None else cfg.FOUNDRY_EMBED_BATCH_SIZE
		if not isinstance(self.model, str) or not self.model.strip():
			raise ValueError('Import a Foundry embedding model and set CHONKY_FOUNDRY_EMBEDDING_MODEL.')
		if not isinstance(self.batch_size, int) or self.batch_size < 1:
			raise ValueError('Foundry embedding batch size must be positive.')

	def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
		import math
		from palantir_models.models import GenericEmbeddingModel
		from language_model_service_api.languagemodelservice_api_embeddings_v3 import GenericEmbeddingsRequest
		if not isinstance(texts, list) or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
			raise ValueError('Foundry embedding input must be a nonempty list of nonempty text strings.')
		alias = model or self.model
		client = GenericEmbeddingModel.get(alias)
		result = []
		dimension = None
		for start in range(0, len(texts), self.batch_size):
			batch = texts[start:start+self.batch_size]
			response = client.create_embeddings(GenericEmbeddingsRequest(inputs=batch))
			vectors = response.embeddings
			if len(vectors) != len(batch):
				raise ValueError('Foundry returned an embedding count different from the requested batch.')
			for vector in vectors:
				values = [float(x) for x in vector]
				if not values or not all(math.isfinite(x) for x in values):
					raise ValueError('Foundry returned an empty or non-finite embedding.')
				if dimension is None:
					dimension = len(values)
				elif len(values) != dimension:
					raise ValueError('Foundry returned inconsistent embedding dimensions.')
				result.append(values)
		return result
