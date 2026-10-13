'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                loaders.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loaders.py" company="Terry D. Eppler">

	     loaders.py
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
    Provides document-loader wrappers for the Chonky ingestion workflow.

    Purpose:
        Defines local, web, repository, cloud, email, notebook, XML, PDF, spreadsheet,
        presentation, and public-data loader wrappers that normalize source content into
        LangChain Document objects. The module centralizes path validation, metadata
        assignment, chunking helpers, provider-specific loader construction, and logged
        exception handling for Chonky's loading stage.
  </summary>
  ******************************************************************************************
'''
import arxiv
import docx2txt

if __package__:
	from .boogr import Error, Logger
else:
	from boogr import Error, Logger
if __package__:
	from . import config as cfg
else:
	import config as cfg
import glob
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
	CSVLoader,
	Docx2txtLoader,
	PyPDFLoader,
	JSONLoader,
	GithubFileLoader,
	UnstructuredExcelLoader,
	RecursiveUrlLoader,
	WebBaseLoader,
	ArxivLoader,
	WikipediaLoader,
	UnstructuredEmailLoader,
	SharePointLoader,
	GoogleDriveLoader,
	UnstructuredPowerPointLoader,
	OutlookMessageLoader,
	OneDriveLoader,
	UnstructuredXMLLoader,
	PubMedLoader,
	OpenCityDataLoader,
	NotebookLoader,
	S3FileLoader,
)

from langchain_google_community import (GCSFileLoader, SpeechToTextLoader)
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_google_community import GCSDirectoryLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import os
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
from lxml import etree

def throw_if( name: str, value: Any ) -> None:
	"""Throw if.
	
	Purpose:
		Executes the throw_if workflow for the loader wrapper while preserving loader state for
		downstream Chonky stages.
	
	Args:
		name: Runtime value used by the operation.
		value: Runtime value used by the operation.
	
	Returns:
		None: Result produced by the operation.
	"""
	if value is None:
		raise ValueError( f"Argument '{name}' cannot be empty!" )

class Loader( ):
	"""Loader document loader wrapper.
	
	Purpose:
		Provides shared file validation, path resolution, and document-splitting utilities used by
		concrete Chonky loader wrappers. The class stores loader configuration, loaded documents,
		splitting settings, and source metadata on the instance so the Streamlit application can
		pass normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		documents: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		pattern: Runtime state used by the loader wrapper.
		expanded: Runtime state used by the loader wrapper.
		candidates: Runtime state used by the loader wrapper.
		resolved: Runtime state used by the loader wrapper.
		loader: Runtime state used by the loader wrapper.
		splitter: Runtime state used by the loader wrapper.
		chunk_size: Runtime state used by the loader wrapper.
		overlap_amount: Runtime state used by the loader wrapper.
	"""
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ BaseLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def verify_exists( self, path: str ) -> str | None:
		"""Verify exists.
		
		Purpose:
			Validates that a supplied filesystem path exists before loader execution.
		
		Args:
			path: Runtime value used by the operation.
		
		Returns:
			str | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			if not os.path.isfile( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			else:
				self.file_path = path
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
		"""Resolve paths.
		
		Purpose:
			Resolves a file path or glob pattern into concrete filesystem paths.
		
		Args:
			pattern: Runtime value used by the operation.
		
		Returns:
			List[str] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'pattern', pattern )
			self.candidates.append( pattern )
			for p in self.candidates:
				if os.path.isfile( p ):
					self.resolved.append( p )
				else:
					for m in glob.glob( p ):
						if os.path.isfile( m ):
							self.resolved.append( m )
			
			if not self.resolved:
				raise FileNotFoundError( f'No files matched or existed for input: {pattern}' )
			return sorted( set( self.resolved ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = 'resolve_paths( self, pattern: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def load_documents( self, path: str, encoding: Optional[ str ],
			csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Executes the load_documents workflow for the Loader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			path: Runtime value used by the operation.
			encoding: Runtime value used by the operation.
			csv_args: Runtime value used by the operation.
			source_column: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = BaseLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			Logger( ).write( exception )
			raise exception
	
	def split_documents( self, docs: List[ Document ], chunk: int = 1000, overlap: int = 200 ) -> \
			List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits supplied Document objects with the configured recursive text splitter.
		
		Args:
			docs: Runtime value used by the operation.
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
				model_name='gpt-4o',
				chunk_size=self.chunk_size, overlap=self.overlap_amount )
			return self.splitter.split_documents( documents=self.documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, **kwargs ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception

class TextLoader( Loader ):
	"""TextLoader document loader wrapper.
	
	Purpose:
		Loads plain-text files into LangChain Document objects and prepares loaded text for
		character or token splitting. The class stores loader configuration, loaded documents,
		splitting settings, and source metadata on the instance so the Streamlit application can
		pass normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		splitter: Runtime state used by the loader wrapper.
		raw_text: Runtime state used by the loader wrapper.
		separator: Runtime state used by the loader wrapper.
		length_function: Runtime state used by the loader wrapper.
	"""
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	raw_text: Optional[ str ]
	separator: Optional[ str ]
	length_function: Optional[ object ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.raw_text = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.separator = "\n\n"
		self.length_function = len
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'raw_text',
				'separator',
				'length_function',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split_tokens',
				'split_chars',
		]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			filepath: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			
			with open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ) as handle:
				self.raw_text = handle.read( )
			
			self.documents = [
					Document(
						page_content=self.raw_text if isinstance( self.raw_text, str ) else '',
						metadata={
								'source': os.path.basename( self.file_path ),
								'loader': 'TextLoader',
								'path': self.file_path,
						}
					)
			]
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'load( self, filepath: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_tokens( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split tokens.
		
		Purpose:
			Executes the split_tokens workflow for the TextLoader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			size: Runtime value used by the operation.
			amount: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
				encoding_name='cl100k_base',
				chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount
			)
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'tokens'
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = ('split_tokens( self, size: int=1000, amount: int=200 ) -> List[ '
			                    'Document ] | None')
			Logger( ).write( exception )
			raise exception
	
	def split_chars( self, size: int = 1000, amount: int = 200,
			seps: str = "\n\n" ) -> List[ Document ] | None:
		"""Split chars.
		
		Purpose:
			Executes the split_chars workflow for the TextLoader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			size: Runtime value used by the operation.
			amount: Runtime value used by the operation.
			seps: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.separator = seps
			self.splitter = CharacterTextSplitter(
				separator=self.separator,
				chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount,
				length_function=self.length_function
			)
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'chars'
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = (
					'split_chars( self, size: int=1000, amount: int=200, '
					'seps: str="\\n\\n" ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class CsvLoader( Loader ):
	"""CsvLoader document loader wrapper.
	
	Purpose:
		Loads delimited CSV content into LangChain Document objects for downstream Chonky
		processing. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		splitter: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		quote_char: Runtime state used by the loader wrapper.
		csv_args: Runtime state used by the loader wrapper.
		columns: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	file_path: Optional[ str ]
	quote_char: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	columns: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.columns = None
		self.csv_args = None
		self.documents = None
		self.quote_char = '"'
		self.pattern = ','
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'delimiter',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'csv_args',
				'columns',
		]
	
	def load( self, filepath: str, columns: Optional[ List[ str ] ] = None,
			delimiter: str = ',', quotechar: str = '"' ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			filepath: Runtime value used by the operation.
			columns: Runtime value used by the operation.
			delimiter: Runtime value used by the operation.
			quotechar: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			
			self.file_path = self.verify_exists( filepath )
			self.columns = columns
			self.pattern = delimiter if isinstance( delimiter, str ) and delimiter else ','
			self.quote_char = quotechar if isinstance( quotechar, str ) and quotechar else '"'
			self.csv_args = { 'delimiter': self.pattern, 'quotechar': self.quote_char, }
			
			if isinstance( self.columns, list ) and self.columns:
				self.csv_args[ 'fieldnames' ] = self.columns
			
			self.loader = CSVLoader( file_path=self.file_path, csv_args=self.csv_args,
				content_columns=self.columns, )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = (
					'load( self, filepath: str, columns: Optional[ List[ str ] ]=None, '
					'delimiter: str=",", quotechar: str=\'"\' ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			size: Runtime value used by the operation.
			amount: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = (
					'split( self, size: int=1000, amount: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class XmlLoader( Loader ):
	"""XmlLoader document loader wrapper.
	
	Purpose:
		Loads XML through semantic document parsing and structured lxml tree parsing for XPath
		workflows. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		loader: Runtime state used by the loader wrapper.
		splitter: Runtime state used by the loader wrapper.
		chunk_size: Runtime state used by the loader wrapper.
		overlap_amount: Runtime state used by the loader wrapper.
		xml_tree: Runtime state used by the loader wrapper.
		xml_root: Runtime state used by the loader wrapper.
		xml_namespaces: Runtime state used by the loader wrapper.
	"""
	
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	loader: Optional[ UnstructuredXMLLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	xml_tree: Optional[ etree._ElementTree ]
	xml_root: Optional[ etree._Element ]
	xml_namespaces: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.loader = None
		self.splitter = None
		self.chunk_size = None
		self.overlap_amount = None
		self.xml_tree = None
		self.xml_root = None
		self.xml_namespaces = None
	
	def __dir__( self ) -> List[ str ]:
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		
		Returns:
			List[str]: Result produced by the operation.
		"""
		return [
				"loader",
				"documents",
				"splitter",
				"file_path",
				"expanded",
				"candidates",
				"resolved",
				"chunk_size",
				"overlap_amount",
				"xml_tree",
				"xml_root",
				"xml_namespaces",
				"verify_exists",
				"resolve_paths",
				"split_documents",
				"load",
				"split",
				"load_tree",
				"get_elements",
		]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			filepath: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			self.file_path = self.verify_exists( filepath )
			self.loader = UnstructuredXMLLoader( file_path=self.file_path, mode="elements" )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "load(self, filepath: str)"
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			size: Runtime value used by the operation.
			amount: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( "No documents loaded via load()." )
			self.chunk_size = size
			self.overlap_amount = amount
			split_docs = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return split_docs
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "split(self, size: int = 1000, amount: int = 200)"
			Logger( ).write( exception )
			raise exception
	
	def load_tree( self, filepath: str ) -> etree._ElementTree | None:
		"""Load tree.
		
		Purpose:
			Parses XML into an lxml ElementTree and stores tree, root, and namespace state.
		
		Args:
			filepath: Runtime value used by the operation.
		
		Returns:
			etree._ElementTree | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			self.file_path = self.verify_exists( filepath )
			parser = etree.XMLParser( recover=True, remove_comments=True, remove_blank_text=True )
			self.xml_tree = etree.parse( self.file_path, parser )
			self.xml_root = self.xml_tree.getroot( )
			self.xml_namespaces = {
					prefix if prefix is not None else "default": uri
					for prefix, uri in (self.xml_root.nsmap or { }).items( )
			}
			
			return self.xml_tree
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "load_tree(self, filepath: str)"
			Logger( ).write( exception )
			raise exception
	
	def get_elements( self, xpath: str ) -> List[ etree._Element ] | None:
		"""Get elements.
		
		Purpose:
			Runs an XPath expression against the loaded XML tree and returns matching elements.
		
		Args:
			xpath: Runtime value used by the operation.
		
		Returns:
			List[etree._Element] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.xml_root is None:
				raise ValueError( "XML tree not loaded. Call load_tree() first." )
			elements = self.xml_root.xpath( xpath, namespaces=self.xml_namespaces )
			return list( elements )
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "get_elements(self, xpath: str)"
			Logger( ).write( exception )
			raise exception

class WebLoader( Loader ):
	"""WebLoader document loader wrapper.
	
	Purpose:
		Loads one or more web pages into LangChain Document objects through direct page loading or
		recursive crawling. The class stores loader configuration, loaded documents, splitting
		settings, and source metadata on the instance so the Streamlit application can pass
		normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		url: Runtime state used by the loader wrapper.
		web_paths: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		max_depth: Runtime state used by the loader wrapper.
		timeout: Runtime state used by the loader wrapper.
		ignore: Runtime state used by the loader wrapper.
		with_progress: Runtime state used by the loader wrapper.
		recursive: Runtime state used by the loader wrapper.
		prevent_outside: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, recursive: bool = False, max_depth: int = 2,
			prevent_outside: bool = True, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> None:
		"""Initialize the wrapper.
		
		Purpose:
			Initializes instance state used by the loader wrapper without changing the loader
			execution contract.
		
		Args:
			recursive: Runtime value used by the operation.
			max_depth: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.url = None
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'url',
				'web_paths',
				'max_depth',
				'timeout',
				'ignore',
				'with_progress',
				'recursive',
				'prevent_outside',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_pages',
				'split',
		]
	
	def load( self, urls: str | List[ str ], depth: int = 2, timeout: int = 10,
			ignore: bool = True, progress: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			urls: Runtime value used by the operation.
			depth: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.recursive:
				return self.load_recursive(
					urls=urls,
					depth=depth,
					timeout=timeout,
					ignore=ignore,
					prevent_outside=prevent_outside
				)
			
			return self.load_pages(
				urls=urls,
				timeout=timeout,
				ignore=ignore,
				progress=progress
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'load( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, progress: bool=True, '
					'prevent_outside: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> List[ Document ] | None:
		"""Load pages.
		
		Purpose:
			Loads one or more web pages into LangChain Document objects.
		
		Args:
			urls: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			
			self.loader = WebBaseLoader(
				web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore,
				requests_kwargs={ 'timeout': self.timeout }
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'load_pages( self, urls: str | List[ str ], timeout: int=10, '
					'ignore: bool=True, progress: bool=True ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int = 2,
			timeout: int = 10, ignore: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load recursive.
		
		Purpose:
			Recursively crawls a starting URL and loads discovered pages into Document objects.
		
		Args:
			urls: Runtime value used by the operation.
			depth: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			
			self.loader = RecursiveUrlLoader(
				url=self.url,
				max_depth=self.max_depth,
				timeout=self.timeout,
				continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PdfLoader( Loader ):
	"""PdfLoader document loader wrapper.
	
	Purpose:
		Loads PDF files into LangChain Document objects with configurable page mode, extraction
		mode, and optional image parsing. The class stores loader configuration, loaded documents,
		splitting settings, and source metadata on the instance so the Streamlit application can
		pass normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		mode: Runtime state used by the loader wrapper.
		extraction: Runtime state used by the loader wrapper.
		include_images: Runtime state used by the loader wrapper.
		image_format: Runtime state used by the loader wrapper.
		custom_delimiter: Runtime state used by the loader wrapper.
		image_parser: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	extraction: Optional[ str ]
	include_images: Optional[ bool ]
	image_format: Optional[ str ]
	custom_delimiter: Optional[ str ]
	image_parser: Optional[ RapidOCRBlobParser ]
	
	def __init__( self, size: int = 1000, overlap: int = 150,
			has_tables: bool = True, include: bool = True ) -> None:
		"""Initialize the wrapper.
		
		Purpose:
			Initializes instance state used by the loader wrapper without changing the loader
			execution contract.
		
		Args:
			size: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
			has_tables: Runtime value used by the operation.
			include: Runtime value used by the operation.
		"""
		super( ).__init__( )
		self.enable_tables = has_tables
		self.include_images = include
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = size
		self.overlap_amount = overlap
		self.loader = None
		self.mode = None
		self.extraction = None
		self.image_format = None
		self.custom_delimiter = None
		self.image_parser = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
 API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'extraction',
				'include_images',
				'image_format',
				'custom_delimiter',
				'image_parser',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
				'extraction_options',
				'image_options',
		]
	
	@property
	def mode_options( self ):
		"""Mode options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'page', 'single' ]
	
	@property
	def extraction_options( self ):
		"""Extraction options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'plain', 'layout' ]
	
	@property
	def image_options( self ):
		"""Image options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'html-img', 'markdown-img', 'text-img' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		""" normalize mode.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			mode: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'elements':
			return 'page'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def _normalize_extraction( self, extract: str ) -> str:
		""" normalize extraction.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			extract: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = extract.strip( ).lower( ) if isinstance( extract, str ) else 'plain'
		
		if value == 'ocr':
			return 'layout'
		
		if value not in self.extraction_options:
			return 'plain'
		
		return value
	
	def _normalize_image_format( self, format: str ) -> str:
		""" normalize image format.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			format: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = format.strip( ).lower( ) if isinstance( format, str ) else 'markdown-img'
		
		if value == 'text':
			return 'markdown-img'
		
		if value not in self.image_options:
			return 'markdown-img'
		
		return value
	
	def load( self, filepath: str, mode: str = 'single', extract: str = 'plain',
			include: bool = False, format: str = 'markdown-img' ) -> List[ Document ]:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			filepath: Runtime value used by the operation.
			mode: Runtime value used by the operation.
			extract: Runtime value used by the operation.
			include: Runtime value used by the operation.
			format: Runtime value used by the operation.
		
		Returns:
			List[Document]: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', filepath )
			
			self.file_path = self.verify_exists( filepath )
			self.mode = self._normalize_mode( mode )
			self.extraction = self._normalize_extraction( extract )
			self.include_images = include
			self.image_format = self._normalize_image_format( format )
			
			if self.include_images:
				self.image_parser = RapidOCRBlobParser( )
				self.loader = PyPDFLoader(
					file_path=self.file_path,
					mode=self.mode,
					extraction_mode=self.extraction,
					extract_images=self.include_images,
					images_inner_format=self.image_format,
					images_parser=self.image_parser
				)
			else:
				self.loader = PyPDFLoader(
					file_path=self.file_path,
					mode=self.mode,
					extraction_mode=self.extraction
				)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = (
					'load( self, filepath: str, mode: str="single", '
					'extract: str="plain", include: bool=False, '
					'format: str="markdown-img" ) -> List[ Document ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ExcelLoader( Loader ):
	"""ExcelLoader document loader wrapper.
	
	Purpose:
		Loads Excel spreadsheets into LangChain Document objects using LangChain unstructured
		spreadsheet loading. The class stores loader configuration, loaded documents, splitting
		settings, and source metadata on the instance so the Streamlit application can pass
		normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		mode: Runtime state used by the loader wrapper.
		has_headers: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
		self.has_headers = True
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'has_headers',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
		]
	
	@property
	def mode_options( self ):
		"""Mode options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		""" normalize mode.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			mode: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single',
			has_headers: bool = True ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
			mode: Runtime value used by the operation.
			has_headers: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.has_headers = has_headers
			
			self.loader = UnstructuredExcelLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'load( self, path: str, mode: str="single", '
					'has_headers: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class WordLoader( Loader ):
	"""WordLoader document loader wrapper.
	
	Purpose:
		Loads Word documents into LangChain Document objects using document text extraction
		wrappers. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_splits = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class MarkdownLoader( Loader ):
	"""MarkdownLoader document loader wrapper.
	
	Purpose:
		Loads Markdown files into LangChain Document objects for downstream cleanup and semantic
		processing. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		mode: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
				API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
		]
	
	@property
	def mode_options( self ):
		"""Mode options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		""" normalize mode.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			mode: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
			mode: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredMarkdownLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return _documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class HtmlLoader( Loader ):
	"""HtmlLoader document loader wrapper.
	
	Purpose:
		Loads HTML files into LangChain Document objects while preserving loader metadata for
		downstream processing. The class stores loader configuration, loaded documents, splitting
		settings, and source metadata on the instance so the Streamlit application can pass
		normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HtmlLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class JsonLoader( Loader ):
	"""JsonLoader document loader wrapper.
	
	Purpose:
		Loads JSON and JSON Lines content into LangChain Document objects using jq-style
		extraction settings. The class stores loader configuration, loaded documents, splitting
		settings, and source metadata on the instance so the Streamlit application can pass
		normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		jq_schema: Runtime state used by the loader wrapper.
		content_key: Runtime state used by the loader wrapper.
		text_content: Runtime state used by the loader wrapper.
		json_lines: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ JSONLoader ]
	file_path: Optional[ str ]
	jq_schema: Optional[ str ]
	content_key: Optional[ str ]
	text_content: Optional[ bool ]
	json_lines: Optional[ bool ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.jq_schema = '.'
		self.content_key = None
		self.text_content = True
		self.json_lines = False
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'jq_schema',
				'content_key',
				'text_content',
				'json_lines',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, filepath: str, jq_schema: str = '.',
			content_key: Optional[ str ] = None, is_text: bool = True,
			is_lines: bool = False ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			filepath: Runtime value used by the operation.
			jq_schema: Runtime value used by the operation.
			content_key: Runtime value used by the operation.
			is_text: Runtime value used by the operation.
			is_lines: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.jq_schema = jq_schema if isinstance( jq_schema,
				str ) and jq_schema.strip( ) else '.'
			self.content_key = (content_key.strip( )
			                    if isinstance( content_key,
				str ) and content_key.strip( ) else None)
			self.text_content = bool( is_text )
			self.json_lines = bool( is_lines )
			kwargs = {
					'file_path': self.file_path,
					'jq_schema': self.jq_schema,
					'text_content': self.text_content,
					'json_lines': self.json_lines,
			}
			
			if self.content_key:
				kwargs[ 'content_key' ] = self.content_key
			
			self.loader = JSONLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = (
					'load( self, filepath: str, jq_schema: str=".", '
					'content_key: Optional[ str ]=None, is_text: bool=True, '
					'is_lines: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ArXivLoader( Loader ):
	"""ArXivLoader document loader wrapper.
	
	Purpose:
		Loads arXiv search results and papers into LangChain Document objects for research-text
		workflows. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		max_documents: Runtime state used by the loader wrapper.
		max_characters: Runtime state used by the loader wrapper.
		include_metadata: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ ArxivLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_metadata = False
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
		         API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_documents',
		         'max_characters',
		         'include_metadata',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, query: str, max_chars: int = 1000 ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			query: Runtime value used by the operation.
			max_chars: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_characters = max_chars
			self.loader = ArxivLoader( query=self.query,
				doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class WikiLoader( Loader ):
	"""WikiLoader document loader wrapper.
	
	Purpose:
		Loads Wikipedia content into LangChain Document objects for reference-corpus processing.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		max_documents: Runtime state used by the loader wrapper.
		max_characters: Runtime state used by the loader wrapper.
		include_all: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ WikipediaLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all = False
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
				API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'max_documents',
				'max_characters',
				'include_all',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, query: str, max_docs: int = 25, max_chars: int = 4000,
			include_all: bool = False ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			query: Runtime value used by the operation.
			max_docs: Runtime value used by the operation.
			max_chars: Runtime value used by the operation.
			include_all: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'query', query )
			
			self.query = query
			self.max_documents = max_docs
			self.max_characters = max_chars
			self.include_all = include_all
			
			self.loader = WikipediaLoader(
				query=self.query,
				load_max_docs=self.max_documents,
				load_all_available_meta=self.include_all,
				doc_content_chars_max=self.max_characters
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = (
					'load( self, query: str, max_docs: int=25, max_chars: int=4000, '
					'include_all: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GithubLoader( Loader ):
	"""GithubLoader document loader wrapper.
	
	Purpose:
		Loads GitHub repository files into LangChain Document objects for source-documentation and
		semantic-search workflows. The class stores loader configuration, loaded documents,
		splitting settings, and source metadata on the instance so the Streamlit application can
		pass normalized content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		repo: Runtime state used by the loader wrapper.
		branch: Runtime state used by the loader wrapper.
		access_token: Runtime state used by the loader wrapper.
		github_url: Runtime state used by the loader wrapper.
		file_filter: Runtime state used by the loader wrapper.
		pattern: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ GithubFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	repo: Optional[ str ]
	branch: Optional[ str ]
	access_token: Optional[ str ]
	github_url: Optional[ str ]
	file_filter: Optional[ object ]
	pattern: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.github_url = None
		self.repo = None
		self.branch = None
		self.access_token = None
		self.file_filter = None
		self.pattern = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'repo',
				'branch',
				'access_token',
				'github_url',
				'file_filter',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, url: str, repo: str, branch: str, filetype: str = '.md',
			access_token: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			url: Runtime value used by the operation.
			repo: Runtime value used by the operation.
			branch: Runtime value used by the operation.
			filetype: Runtime value used by the operation.
			access_token: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'url', url )
			throw_if( 'repo', repo )
			throw_if( 'branch', branch )
			
			self.github_url = url
			self.repo = repo
			self.branch = branch
			self.access_token = access_token.strip( ) if isinstance( access_token,
				str ) and access_token.strip( ) else None
			self.pattern = filetype.strip( ) if isinstance( filetype,
				str ) and filetype.strip( ) else '.md'
			self.file_filter = lambda file_path: file_path.endswith( self.pattern )
			
			kwargs = {
					'repo': self.repo,
					'branch': self.branch,
					'github_api_url': self.github_url,
					'file_filter': self.file_filter,
			}
			
			if self.access_token:
				kwargs[ 'access_token' ] = self.access_token
			
			self.loader = GithubFileLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = (
					'load( self, url: str, repo: str, branch: str, '
					'filetype: str=".md", access_token: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PowerPointLoader( Loader ):
	"""PowerPointLoader document loader wrapper.
	
	Purpose:
		Loads PowerPoint presentation files into LangChain Document objects for slide-text
		processing. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		mode: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'query',
				'mode',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_multiple',
				'split',
		]
	
	def _normalize_mode( self, mode: str ) -> str:
		""" normalize mode.
		
		Purpose:
			Normalizes a UI or legacy option value into a supported loader option before execution.
		
		Args:
			mode: Runtime value used by the operation.
		
		Returns:
			str: Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'multiple':
			return 'elements'
		
		if value not in [ 'single', 'elements' ]:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
			mode: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredPowerPointLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		"""Load multiple.
		
		Purpose:
			Executes the load_multiple workflow for the PowerPointLoader wrapper while preserving
			loader state for downstream Chonky stages.
		
		Args:
			path: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			return self.load( path, mode='elements' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load_multiple( self, path: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class OutlookLoader( Loader ):
	"""OutlookLoader document loader wrapper.
	
	Purpose:
		Loads Outlook message files into LangChain Document objects for email-content processing.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		max_documents: Runtime state used by the loader wrapper.
		max_characters: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ OutlookMessageLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_charactes',
		         'max_documents',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class WebCrawler( Loader ):
	"""WebCrawler document loader wrapper.
	
	Purpose:
		Crawls web pages into LangChain Document objects for recursive document ingestion. The
		class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		url: Runtime state used by the loader wrapper.
		web_paths: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		max_depth: Runtime state used by the loader wrapper.
		timeout: Runtime state used by the loader wrapper.
		ignore: Runtime state used by the loader wrapper.
		with_progress: Runtime state used by the loader wrapper.
		recursive: Runtime state used by the loader wrapper.
		prevent_outside: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, url: str, recursive: bool = False, max_depth: int = 2,
			prevent_outside: bool = True, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> None:
		"""Initialize the wrapper.
		
		Purpose:
			Initializes instance state used by the loader wrapper without changing the loader
			execution contract.
		
		Args:
			url: Runtime value used by the operation.
			recursive: Runtime value used by the operation.
			max_depth: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.url = url
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
		self.loader = RecursiveUrlLoader( url=self.url, max_depth=self.max_depth,
			timeout=self.timeout, continue_on_failure=self.ignore,
			prevent_outside=self.prevent_outside )
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
			API documentation.
		"""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'url',
				'web_paths',
				'max_depth',
				'timeout',
				'ignore',
				'with_progress',
				'recursive',
				'prevent_outside',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_pages',
				'split',
		]
	
	def load( self, urls: str | List[ str ], depth: int = 2, timeout: int = 10,
			ignore: bool = True, progress: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			urls: Runtime value used by the operation.
			depth: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.recursive:
				return self.load_recursive(
					urls=urls,
					depth=depth,
					timeout=timeout,
					ignore=ignore,
					prevent_outside=prevent_outside
				)
			
			return self.load_pages(
				urls=urls,
				timeout=timeout,
				ignore=ignore,
				progress=progress
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, progress: bool=True, '
					'prevent_outside: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> List[ Document ] | None:
		"""Load pages.
		
		Purpose:
			Loads one or more web pages into LangChain Document objects.
		
		Args:
			urls: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			progress: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			
			self.loader = WebBaseLoader(
				web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore,
				requests_kwargs={ 'timeout': self.timeout }
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load_pages( self, urls: str | List[ str ], timeout: int=10, '
					'ignore: bool=True, progress: bool=True ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int = 2,
			timeout: int = 10, ignore: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load recursive.
		
		Purpose:
			Recursively crawls a starting URL and loads discovered pages into Document objects.
		
		Args:
			urls: Runtime value used by the operation.
			depth: Runtime value used by the operation.
			timeout: Runtime value used by the operation.
			ignore: Runtime value used by the operation.
			prevent_outside: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'urls', urls )
			
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			
			self.loader = RecursiveUrlLoader(
				url=self.url,
				max_depth=self.max_depth,
				timeout=self.timeout,
				continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class SpfxLoader( Loader ):
	"""SpfxLoader document loader wrapper.
	
	Purpose:
		Loads SharePoint content into LangChain Document objects for connected-document workflows.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		library_id: Runtime state used by the loader wrapper.
		subsite_id: Runtime state used by the loader wrapper.
		folder_id: Runtime state used by the loader wrapper.
		object_ids: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		with_token: Runtime state used by the loader wrapper.
		is_recursive: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ SharePointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	library_id: Optional[ str ]
	subsite_id: Optional[ str ]
	folder_id: Optional[ str ]
	object_ids: Optional[ List[ str ] ]
	query: Optional[ str ]
	with_token: Optional[ bool ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.folder_id = None
		self.library_id = None
		self.subsite_id = None
		self.object_ids = [ ]
		self.with_token = None
		self.is_recursive = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
		         API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'folder_id',
		         'library_id',
		         'subsite_id',
		         'object_id',
		         'with_token',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, library_id: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			library_id: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'library_id', library_id )
			self.library_id = library_id
			self.is_recursive = True
			self.with_token = True
			self.loader = SharePointLoader( document_library_id=self.library_id,
				recursive=self.is_recursive, auth_with_token=self.with_token )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		"""Load folder.
		
		Purpose:
			Executes the load_folder workflow for the SpfxLoader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			library_id: Runtime value used by the operation.
			folder_id: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'library_id', library_id )
			throw_if( 'folder_id', folder_id )
			self.library_id = library_id
			self.folder_id = folder_id
			self.loader = SharePointLoader( document_library_id=self.library_id,
				folder_id=self.folder_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class OneDriveDocLoader( Loader ):
	"""OneDriveDocLoader document loader wrapper.
	
	Purpose:
		Loads OneDrive documents into LangChain Document objects for connected-file workflows. The
		class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		client_id: Runtime state used by the loader wrapper.
		drive_id: Runtime state used by the loader wrapper.
		client_secret: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.drive_id = None
		self.client_id = None
		self.client_secret = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
		         API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'drive_id',
		         'client_id',
		         'client_secret',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ):
		"""File options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'pdf', 'doc', 'docx', 'txt' ]
	
	def load( self, id: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			id: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		"""Load folder.
		
		Purpose:
			Executes the load_folder workflow for the OneDriveDocLoader wrapper while preserving
			loader state for downstream Chonky stages.
		
		Args:
			id: Runtime value used by the operation.
			path: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class GoogleLoader( Loader ):
	"""GoogleLoader document loader wrapper.
	
	Purpose:
		Loads Google Drive content into LangChain Document objects for cloud-document ingestion.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		file_id: Runtime state used by the loader wrapper.
		folder_id: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		is_recursive: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ GoogleDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	file_id: Optional[ str ]
	folder_id: Optional[ str ]
	query: Optional[ str ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.file_id = None
		self.folder_id = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_recursive = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
		         API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'folder_id',
		         'file_id',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ):
		"""File options.
		
		Purpose:
			Exposes supported option values for Streamlit controls and documentation output.
		"""
		return [ 'document',
		         'sheet',
		         'pdf' ]
	
	def load_file( self, file_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Load file.
		
		Purpose:
			Executes the load_file workflow for the GoogleLoader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			file_id: Runtime value used by the operation.
			recursive: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'recursive', recursive )
			self.file_id = file_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( file_ids=[ self.file_id ],
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_File( self, file_id: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, folder_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Load folder.
		
		Purpose:
			Executes the load_folder workflow for the GoogleLoader wrapper while preserving loader
			state for downstream Chonky stages.
		
		Args:
			folder_id: Runtime value used by the operation.
			recursive: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'folder_id', folder_id )
			self.folder_id = folder_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( folder_id=self.folder_id,
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class EmailLoader( Loader ):
	"""EmailLoader document loader wrapper.
	
	Purpose:
		Loads email message files into LangChain Document objects for message-text processing. The
		class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		has_attachments: Runtime state used by the loader wrapper.
		mode: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""  dir  .
		
		Purpose:
			Returns the public member names exposed by the wrapper for introspection and generated
		         API documentation.
		"""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'has_attachments',
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, mode: str = 'single', attachments: bool = True ) -> List[
		Document ]:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
			mode: Runtime value used by the operation.
			attachments: Runtime value used by the operation.
		
		Returns:
			List[Document]: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.has_attachments = attachments
			self.loader = UnstructuredEmailLoader( file_path=self.file_path, mode=self.mode,
				process_attachments=self.has_attachments )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('split( self, chunk: int=1000, overlap: int=200 ) -> List[ '
			                    'Document ]')
			Logger( ).write( exception )
			raise exception

class PubMedSearchLoader( Loader ):
	"""PubMedSearchLoader document loader wrapper.
	
	Purpose:
		Loads PubMed search results into LangChain Document objects for biomedical literature
		workflows. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		query: Runtime state used by the loader wrapper.
		max_docs: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ PubMedLoader ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_docs: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.query = None
		self.max_docs = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'query',
				'max_docs',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, query: str, max_docs: int = 5 ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			query: Runtime value used by the operation.
			max_docs: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_docs = max_docs
			self.loader = PubMedLoader( query=self.query, load_max_docs=self.max_docs )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
					'load( self, query: str, max_docs: int=5 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class OpenCityLoader( Loader ):
	"""OpenCityLoader document loader wrapper.
	
	Purpose:
		Loads open-city dataset content into LangChain Document objects for public-data workflows.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		city_id: Runtime state used by the loader wrapper.
		dataset_id: Runtime state used by the loader wrapper.
		limit: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ OpenCityDataLoader ]
	documents: Optional[ List[ Document ] ]
	city_id: Optional[ str ]
	dataset_id: Optional[ str ]
	limit: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.city_id = None
		self.dataset_id = None
		self.limit = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'city_id',
				'dataset_id',
				'limit',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, city_id: str, dataset_id: str, limit: int = 100 ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			city_id: Runtime value used by the operation.
			dataset_id: Runtime value used by the operation.
			limit: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'city_id', city_id )
			throw_if( 'dataset_id', dataset_id )
			self.city_id = city_id
			self.dataset_id = dataset_id
			self.limit = limit
			self.loader = OpenCityDataLoader(
				city_id=self.city_id,
				dataset_id=self.dataset_id,
				limit=self.limit
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = (
					'load( self, city_id: str, dataset_id: str, limit: int=100 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class JupyterNotebookLoader( Loader ):
	"""JupyterNotebookLoader document loader wrapper.
	
	Purpose:
		Loads Jupyter Notebook cells into LangChain Document objects for notebook-text processing.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		include_outputs: Runtime state used by the loader wrapper.
		max_output_length: Runtime state used by the loader wrapper.
		remove_newline: Runtime state used by the loader wrapper.
		traceback: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ NotebookLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	include_outputs: Optional[ bool ]
	max_output_length: Optional[ int ]
	remove_newline: Optional[ bool ]
	traceback: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.file_path = None
		self.include_outputs = None
		self.max_output_length = None
		self.remove_newline = None
		self.traceback = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'file_path',
				'include_outputs',
				'max_output_length',
				'remove_newline',
				'traceback',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, path: str, include_outputs: bool = False, max_output_length: int = 10,
			remove_newline: bool = False, traceback: bool = False ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			path: Runtime value used by the operation.
			include_outputs: Runtime value used by the operation.
			max_output_length: Runtime value used by the operation.
			remove_newline: Runtime value used by the operation.
			traceback: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.include_outputs = include_outputs
			self.max_output_length = max_output_length
			self.remove_newline = remove_newline
			self.traceback = traceback
			
			self.loader = NotebookLoader(
				self.file_path,
				include_outputs=self.include_outputs,
				max_output_length=self.max_output_length,
				remove_newline=self.remove_newline,
				traceback=self.traceback
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = (
					'load( self, path: str, include_outputs: bool=False, '
					'max_output_length: int=10, remove_newline: bool=False, '
					'traceback: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GoogleCloudFileLoader( Loader ):
	"""GoogleCloudFileLoader document loader wrapper.
	
	Purpose:
		Loads Google Cloud Storage files into LangChain Document objects for cloud-file ingestion.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		project_name: Runtime state used by the loader wrapper.
		bucket: Runtime state used by the loader wrapper.
		blob: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ GCSFileLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	blob: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.blob = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'project_name',
				'bucket',
				'blob',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_name: str, bucket: str, blob: str ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			project_name: Runtime value used by the operation.
			bucket: Runtime value used by the operation.
			blob: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			throw_if( 'blob', blob )
			self.project_name = project_name
			self.bucket = bucket
			self.blob = blob
			self.loader = GCSFileLoader( project_name=self.project_name, bucket=self.bucket,
				blob=self.blob )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
					'load( self, project_name: str, bucket: str, blob: str ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class AwsFileLoader( Loader ):
	"""AwsFileLoader document loader wrapper.
	
	Purpose:
		Loads AWS S3 files into LangChain Document objects for cloud-file ingestion. The class
		stores loader configuration, loaded documents, splitting settings, and source metadata on
		the instance so the Streamlit application can pass normalized content into processing,
		embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		bucket: Runtime state used by the loader wrapper.
		key: Runtime state used by the loader wrapper.
		aws_access_key_id: Runtime state used by the loader wrapper.
		aws_secret_access_key: Runtime state used by the loader wrapper.
		aws_session_token: Runtime state used by the loader wrapper.
		region_name: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ S3FileLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	key: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.key = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'bucket',
				'key',
				'aws_access_key_id',
				'aws_secret_access_key',
				'aws_session_token',
				'region_name',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, bucket: str, key: str, aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None, aws_session_token: Optional[
				str ] = None,
			region_name: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			bucket: Runtime value used by the operation.
			key: Runtime value used by the operation.
			aws_access_key_id: Runtime value used by the operation.
			aws_secret_access_key: Runtime value used by the operation.
			aws_session_token: Runtime value used by the operation.
			region_name: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'key', key )
			
			self.bucket = bucket
			self.key = key
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			
			kwargs: Dict[ str, Any ] = { }
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			
			self.loader = S3FileLoader(
				self.bucket,
				self.key,
				**kwargs
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = (
					'load( self, bucket: str, key: str, '
					'aws_access_key_id: Optional[ str ]=None, '
					'aws_secret_access_key: Optional[ str ]=None, '
					'aws_session_token: Optional[ str ]=None, '
					'region_name: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GoogleSpeechToTextLoader( Loader ):
	"""GoogleSpeechToTextLoader document loader wrapper.
	
	Purpose:
		Loads speech-to-text output through Google-backed document loading workflows. The class
		stores loader configuration, loaded documents, splitting settings, and source metadata on
		the instance so the Streamlit application can pass normalized content into processing,
		embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		project_id: Runtime state used by the loader wrapper.
		file_path: Runtime state used by the loader wrapper.
		config: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ SpeechToTextLoader ]
	documents: Optional[ List[ Document ] ]
	project_id: Optional[ str ]
	file_path: Optional[ str ]
	config: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_id = None
		self.file_path = None
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'project_id',
				'file_path',
				'config',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_id: str, file_path: str,
			config: Optional[ Dict[ str, Any ] ] = None ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			project_id: Runtime value used by the operation.
			file_path: Runtime value used by the operation.
			config: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'project_id', project_id )
			throw_if( 'file_path', file_path )
			
			self.project_id = project_id
			self.file_path = file_path
			self.config = config
			
			if self.config:
				self.loader = SpeechToTextLoader( project_id=self.project_id, file_path=self.file_path,
					config=self.config )
			else:
				self.loader = SpeechToTextLoader( project_id=self.project_id,
					file_path=self.file_path )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = 'load( self, *args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = 'split( self, *args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception

class GoogleBucketLoader( Loader ):
	"""GoogleBucketLoader document loader wrapper.
	
	Purpose:
		Loads Google Cloud Storage bucket contents into LangChain Document objects for batch cloud
		ingestion. The class stores loader configuration, loaded documents, splitting settings,
		and source metadata on the instance so the Streamlit application can pass normalized
		content into processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		project_name: Runtime state used by the loader wrapper.
		bucket: Runtime state used by the loader wrapper.
		prefix: Runtime state used by the loader wrapper.
		continue_on_failure: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ GCSDirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	continue_on_failure: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.prefix = None
		self.continue_on_failure = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'project_name',
				'bucket',
				'prefix',
				'continue_on_failure',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_name: str, bucket: str, prefix: Optional[ str ] = None,
			continue_on_failure: bool = False ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			project_name: Runtime value used by the operation.
			bucket: Runtime value used by the operation.
			prefix: Runtime value used by the operation.
			continue_on_failure: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			self.project_name = project_name
			self.bucket = bucket
			self.prefix = prefix
			self.continue_on_failure = continue_on_failure
			kwargs: Dict[ str, Any ] = {
					'project_name': self.project_name,
					'bucket': self.bucket,
					'continue_on_failure': self.continue_on_failure,
			}
			
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			
			self.loader = GCSDirectoryLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = 'load( self, *args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class AwsBucketLoader( Loader ):
	"""AwsBucketLoader document loader wrapper.
	
	Purpose:
		Loads AWS S3 bucket contents into LangChain Document objects for batch cloud ingestion.
		The class stores loader configuration, loaded documents, splitting settings, and source
		metadata on the instance so the Streamlit application can pass normalized content into
		processing, embedding, and retrieval stages.
	
	Attributes:
		loader: Runtime state used by the loader wrapper.
		documents: Runtime state used by the loader wrapper.
		bucket: Runtime state used by the loader wrapper.
		prefix: Runtime state used by the loader wrapper.
		aws_access_key_id: Runtime state used by the loader wrapper.
		aws_secret_access_key: Runtime state used by the loader wrapper.
		aws_session_token: Runtime state used by the loader wrapper.
		region_name: Runtime state used by the loader wrapper.
		endpoint_url: Runtime state used by the loader wrapper.
	"""
	loader: Optional[ S3DirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	endpoint_url: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.prefix = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
		self.endpoint_url = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'bucket',
				'prefix',
				'aws_access_key_id',
				'aws_secret_access_key',
				'aws_session_token',
				'region_name',
				'endpoint_url',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load(
			self,
			bucket: str,
			prefix: Optional[ str ] = None,
			aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None,
			aws_session_token: Optional[ str ] = None,
			region_name: Optional[ str ] = None,
			endpoint_url: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load.
		
		Purpose:
			Loads source content into LangChain Document objects and stores the active loader
			response on the instance.
		
		Args:
			bucket: Runtime value used by the operation.
			prefix: Runtime value used by the operation.
			aws_access_key_id: Runtime value used by the operation.
			aws_secret_access_key: Runtime value used by the operation.
			aws_session_token: Runtime value used by the operation.
			region_name: Runtime value used by the operation.
			endpoint_url: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket = bucket
			self.prefix = prefix
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			self.endpoint_url = endpoint_url
			
			kwargs: Dict[ str, Any ] = { }
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			if self.endpoint_url:
				kwargs[ 'endpoint_url' ] = self.endpoint_url
			
			self.loader = S3DirectoryLoader(
				self.bucket,
				**kwargs
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = (
					'load( self, bucket: str, prefix: Optional[ str ]=None, '
					'aws_access_key_id: Optional[ str ]=None, '
					'aws_secret_access_key: Optional[ str ]=None, '
					'aws_session_token: Optional[ str ]=None, '
					'region_name: Optional[ str ]=None, '
					'endpoint_url: Optional[ str ]=None ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split.
		
		Purpose:
			Splits loaded Document objects into smaller chunks and stores chunk settings on the
			instance.
		
		Args:
			chunk: Runtime value used by the operation.
			overlap: Runtime value used by the operation.
		
		Returns:
			List[Document] | None: Result produced by the operation.
		
		Raises:
			Error: Raised when validation, loading, parsing, or document splitting fails.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
