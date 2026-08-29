'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                loca.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loca.py" company="Terry D. Eppler">

	     loca.py
	     Copyright ©  2023  Terry Eppler

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
    Provides local GGUF embedding wrappers for the Chonky embedding workflow.

    Purpose:
        Defines a llama-cpp-python backed embedding base class and concrete local model
        wrappers used to create single-text and batch embeddings without a hosted provider.
        The module validates configured model names and GGUF model paths, lazily loads local
        embedding clients, stores raw embedding responses for diagnostics, returns list-based
        embedding vectors for downstream vector persistence, and records wrapped failures
        through the application logger.
  </summary>
  ******************************************************************************************
'''
import os
from typing import List, Optional, Any
from llama_cpp import Llama
from boogr import Error, Logger

def throw_if( name: str, value: object ):
	"""Validate a required runtime value.

	Purpose:
		Provides a lightweight guard for local embedding wrapper methods that require model
		configuration, model paths, or text input before creating llama-cpp embedding requests.

	Args:
		name: Name of the argument or instance value being validated.
		value: Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is ``None``.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Loca( ):
	"""Base wrapper for local GGUF embedding models.

	Purpose:
		Loads local GGUF embedding models through llama-cpp-python and exposes a provider-like
		contract for Chonky's embedding and vector-database workflows. The class stores the
		logical model name, GGUF model path, loaded llama.cpp client, raw response, and returned
		embedding vectors on the instance for diagnostics and downstream processing.

	Attributes:
		model: Logical model name used to identify the local embedding wrapper.
		model_path: Filesystem path to the GGUF model file.
		client: Loaded llama-cpp-python client configured for embedding generation.
		response: Raw embedding response from the last llama.cpp request.
		embedding: Single embedding vector returned by the last single-text request.
		embeddings: Batch embedding vectors returned by the last batch request.
	"""
	
	model: Optional[ str ]
	model_path: Optional[ str ]
	client: Optional[ Llama ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	
	def __init__( self, model: Optional[ str ] = None, model_path: Optional[ str ] = None ):
		"""Initialize the local embedding wrapper.

		Purpose:
			Stores the logical model name and GGUF path supplied by concrete wrapper classes,
			then initializes client, response, and vector state used by lazy model loading and
			embedding generation.

		Args:
			model: Logical name assigned to the local embedding model.
			model_path: Filesystem path to the local GGUF embedding model.
		"""
		super( ).__init__( )
		self.model = model
		self.model_path = model_path
		self.client = None
		self.response = None
		self.embedding = None
		self.embeddings = None
	
	def load( self ) -> None:
		"""Load the configured local GGUF embedding model.

		Purpose:
			Validates the logical model name and GGUF model path, verifies the model file exists,
			and lazily initializes a llama-cpp-python client configured for embedding generation.
			If the client is already loaded, the method returns without reloading the model.

		Raises:
			Error: Raised when model configuration, file validation, or llama.cpp loading fails.
		"""
		try:
			throw_if( 'model', self.model )
			throw_if( 'model_path', self.model_path )
			
			if self.client is not None:
				return
			
			if not os.path.exists( self.model_path ):
				raise FileNotFoundError( f'Local GGUF model not found: {self.model_path}' )
			
			self.client = Llama(
				model_path=self.model_path,
				embedding=True,
				verbose=False
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'load( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str ) -> List[ float ]:
		"""Create a single local embedding vector.

		Purpose:
			Validates and normalizes a single text value, lazily loads the configured GGUF model,
			calls llama.cpp embedding generation, stores the raw response, and returns the first
			resolved embedding vector for downstream Chonky vector workflows.

		Args:
			text: Text value to embed.

		Returns:
			List[float]: Embedding vector for the supplied text, or an empty list for blank text.

		Raises:
			Error: Raised when validation, model loading, or local embedding generation fails.
		"""
		try:
			throw_if( 'text', text )
			
			if not isinstance( text, str ):
				raise TypeError( 'Argument "text" must be a string.' )
			
			value: str = text.strip( )
			if not value:
				return [ ]
			
			self.load( )
			self.response = self.client.create_embedding( value )
			
			if isinstance( self.response, dict ) and 'data' in self.response:
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
			else:
				self.embedding = self.response
			
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'create( self, text: str ) -> List[ float ]'
			Logger( ).write( exception )
			raise exception
	
	def embed( self, texts: List[ str ] ) -> List[ List[ float ] ]:
		"""Create batch local embedding vectors.

		Purpose:
			Validates a list of text inputs, removes blank values, lazily loads the configured
			GGUF model, calls llama.cpp batch embedding generation, stores the raw response, and
			returns embedding vectors for Chonky diagnostics and vector persistence.

		Args:
			texts: Text values to embed.

		Returns:
			List[List[float]]: Embedding vectors for cleaned text inputs, or an empty list when
			no non-empty text values are supplied.

		Raises:
			Error: Raised when validation, model loading, or batch embedding generation fails.
		"""
		try:
			throw_if( 'texts', texts )
			
			if not isinstance( texts, list ):
				raise TypeError( 'Argument "texts" must be a list of strings.' )
			
			cleaned: List[ str ] = [ ]
			for text in texts:
				if isinstance( text, str ):
					value = text.strip( )
					if value:
						cleaned.append( value )
			
			if not cleaned:
				return [ ]
			
			self.load( )
			self.response = self.client.create_embedding( cleaned )
			
			if isinstance( self.response, dict ) and 'data' in self.response:
				self.embeddings = [ item[ 'embedding' ] for item in self.response[ 'data' ] ]
			else:
				self.embeddings = self.response
			
			return self.embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'embedders'
			exception.cause = 'Loca'
			exception.method = 'embed( self, texts: List[ str ] ) -> List[ List[ float ] ]'
			Logger( ).write( exception )
			raise exception

class Booger( Loca ):
	"""Local BGE small English embedding wrapper.

	Purpose:
		Configures the shared local embedding base class for the local BGE small English GGUF
		model stored under Chonky's ``models/boogr`` directory. The wrapper supplies the logical
		model name and model path while inheriting loading and embedding behavior from ``Loca``.
	"""
	
	def __init__( self ):
		"""Initialize the Booger local embedding model wrapper.

		Purpose:
			Assigns the local BGE small English model name and GGUF path used by the inherited
			``Loca`` loading, single-text embedding, and batch embedding methods.
		"""
		super( ).__init__( model='boogr-small-en-v1.5', model_path=os.path.abspath(
			os.path.join( 'models', 'boogr', 'boogr-small-en-v1.5-q8_0.gguf' ) ) )

class Nomnom( Loca ):
	"""Local Nomic embedding wrapper.

	Purpose:
		Configures the shared local embedding base class for the local Nomic GGUF model stored
		under Chonky's ``models/nomi`` directory. The wrapper supplies the logical model name
		and model path while inheriting loading and embedding behavior from ``Loca``.
	"""
	
	def __init__( self ):
		"""Initialize the Nomnom local embedding model wrapper.

		Purpose:
			Assigns the local Nomic model name and GGUF path used by the inherited ``Loca``
			loading, single-text embedding, and batch embedding methods.
		"""
		super( ).__init__( model='nomnom-embed-text-v1.5',
			model_path=os.path.abspath(
				os.path.join( 'models', 'nomi', 'nomnom-embed-text-v1.5.Q4_K_M.gguf' ) ) )

class Bobo( Loca ):
	"""Local Mixedbread embedding wrapper.

	Purpose:
		Configures the shared local embedding base class for the local Mixedbread GGUF model
		stored under Chonky's ``models/bobo`` directory. The wrapper supplies the logical model
		name and model path while inheriting loading and embedding behavior from ``Loca``.
	"""
	
	def __init__( self ):
		"""Initialize the Bobo local embedding model wrapper.

		Purpose:
			Assigns the local Mixedbread model name and GGUF path used by the inherited ``Loca``
			loading, single-text embedding, and batch embedding methods.
		"""
		super( ).__init__( model='bobo-embed-large-v1', model_path=os.path.abspath(
			os.path.join( 'models', 'bobo', 'bobo-embed-large-v1-f16.gguf' ) ) )