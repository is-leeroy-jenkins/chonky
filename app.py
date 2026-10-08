'''
  ******************************************************************************************
      Assembly:                Chonky
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

	     app.py
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
    Provides the Streamlit application shell for the Chonky document-processing workflow.

    Purpose:
        Coordinates Chonky's staged interface for document loading, text processing,
        semantic analysis, tokenization diagnostics, hosted and local embedding generation,
        vector persistence, and similarity retrieval. The module initializes Streamlit
        session state, configures the page, wires loader, processor, and embedder wrappers
        into tabbed UI sections, and preserves shared workflow outputs for downstream tabs.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
import sys
import types
import altair
import base64
from collections import Counter
from lxml import etree

import config as cfg
import collections
import functools
import itertools
import json
import sqlite3
import math
import nltk
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import re
import sqlite3
import sqlite_vec
import sys
import statistics
import streamlit as st
import time
import tempfile
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings, )
from langchain_community.vectorstores import SQLiteVec
from processors import Processor, TextParser, NltkParser, WordParser, PdfParser
from loaders import (TextLoader, CsvLoader, PdfLoader, ExcelLoader, WordLoader, MarkdownLoader,
                     HtmlLoader, JsonLoader, PowerPointLoader, WikiLoader, GithubLoader, WebLoader,
                     ArXivLoader, XmlLoader, OutlookLoader, WebCrawler, EmailLoader,
                     JupyterNotebookLoader, PubMedSearchLoader, OpenCityLoader,
                     GoogleCloudFileLoader, AwsFileLoader, GoogleBucketLoader, AwsBucketLoader,
                     OneDriveDocLoader, SpfxLoader, )

from embedders import GPT, Grok, Gemini

try:
	import textstat
	
	TEXTSTAT_AVAILABLE = True
except Exception:
	TEXTSTAT_AVAILABLE = False
	textstat = None

from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize

# ======================================================================================
# Session State Initialization
# ======================================================================================

if 'openai_api_key' not in st.session_state:
	st.session_state[ 'openai_api_key' ] = ''

if 'gemini_api_key' not in st.session_state:
	st.session_state[ 'gemini_api_key' ] = ''

if 'groq_api_key' not in st.session_state:
	st.session_state[ 'groq_api_key' ] = ''

if 'google_api_key' not in st.session_state:
	st.session_state[ 'google_api_key' ] = ''

if 'pinecone_api_key' not in st.session_state:
	st.session_state[ 'pinecone_api_key' ] = ''

if 'chroma_api_key' not in st.session_state:
	st.session_state[ 'chroma_api_key' ] = ''

if 'chroma_tenant' not in st.session_state:
	st.session_state[ 'chroma_tenant' ] = ''

if 'chroma_database' not in st.session_state:
	st.session_state[ 'chroma_database' ] = ''

if 'google_application_credentials' not in st.session_state:
	st.session_state[ 'google_application_credentials' ] = ''

for key, default in cfg.SESSION_STATE_DEFAULTS.items( ):
	if key not in st.session_state:
		st.session_state[ key ] = default

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )

# ======================================================================================
# UTILITIES
# ======================================================================================

def style_subheaders( ) -> None:
	"""Style Streamlit subheaders.

	Purpose:
		Injects the CSS used to keep Chonky's secondary headings visually aligned with the
		application accent color. The helper affects Streamlit markdown containers and chat
		message headings without changing application state or downstream workflow data.
	"""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

def encode_image_base64( path: str ) -> str:
	"""Encode an image file as base64 text.

	Purpose:
		Reads an image from disk and converts its bytes into a base64 string for Streamlit
		HTML or markdown rendering where inline image content is required.

	Args:
		path: Filesystem path to the image file.

	Returns:
		str: Base64-encoded image content.
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def clear_if_active( loader_name: str ) -> None:
	"""
	Purpose:
		Clears loader-specific session state when the supplied loader is currently active.
		The reset removes loaded documents, processed data, chunk records, diagnostic data,
		embeddings, and dataframe state without affecting an unrelated active loader.

	Args:
		loader_name: Name of the loader whose state should be cleared.

	Returns:
		None.
	"""
	if st.session_state.get( 'active_loader' ) == loader_name:
		st.session_state.documents = None
		st.session_state.raw_documents = None
		st.session_state.raw_text = None
		st.session_state.processed_text = None
		st.session_state.displayed_text = ''
		st.session_state.processed_text_display = ''
		st.session_state.active_loader = None
		st.session_state.source_file_name = ''
		st.session_state.document_name = ''
		st.session_state.collection_name = ''
		st.session_state.tokens = None
		st.session_state.vocabulary = None
		st.session_state.token_counts = None
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.df_chunk_records = None
		st.session_state.chunk_size = None
		st.session_state.chunk_overlap = None
		st.session_state.chunk_mode_value = None
		st.session_state.sentences = None
		st.session_state.lines = None
		st.session_state.df_sentence_tokens = None
		st.session_state.embeddings = None
		st.session_state.embedding_model = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_token_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None
		st.session_state.df_chunks = None
		
		st.session_state.pdf_pages = None

def derive_document_identity( source_file_name: str ) -> tuple[ str, str ]:
	"""Derive canonical document and vector collection names.

	Purpose:
		Preserves the uploaded document stem exactly for local TXT and CSV downloads while
		deriving one lowercase snake-case identifier for Chroma collections and Pinecone
		namespaces. The normalized identifier is generated once and then retained in session state.

	Args:
		source_file_name: Original uploaded filename or filesystem basename.

	Returns:
		tuple[str, str]: Original-case document stem and lowercase snake-case collection name.
	"""
	cfg.throw_if( 'source_file_name', source_file_name )

	file_name = Path( str( source_file_name ).strip( ) ).name
	document_name = Path( file_name ).stem.strip( )

	if not document_name:
		return '', ''

	collection_name = re.sub( r'[^A-Za-z0-9]+', '_', document_name ).strip( '_' ).lower( )

	if not collection_name:
		collection_name = 'document'
	elif len( collection_name ) < 3:
		collection_name = f'doc_{collection_name}'

	return document_name, collection_name

def sync_document_identity( ) -> None:
	"""Synchronize uploaded document identity with the active loader.

	Purpose:
		Captures the original uploaded filename after a local document loader succeeds. The
		identity becomes the single downstream source for processed-text filenames, chunk CSV
		filenames, Chroma collection names, and Pinecone namespaces.

	Returns:
		None: This function updates Streamlit session state.
	"""
	loader_upload_keys = {
		'TextLoader': 'txt_upload',
		'CsvLoader': 'csv_upload',
		'XmlLoader': 'xml_file_uploader',
		'WordLoader': 'word_upload',
		'PdfLoader': 'pdf_upload',
		'PowerPointLoader': 'pptx_upload',
		'JupyterNotebookLoader': 'ipynb_upload',
		'ExcelLoader': 'excel_upload',
		'MarkdownLoader': 'md_upload',
		'HtmlLoader': 'html_upload',
		'JsonLoader': 'json_upload',
		'OutlookLoader': 'outlook_upload',
		'EmailLoader': 'email_upload',
	}

	active_loader = st.session_state.get( 'active_loader' )
	upload_key = loader_upload_keys.get( active_loader )

	if not upload_key:
		st.session_state.source_file_name = ''
		st.session_state.document_name = ''
		st.session_state.collection_name = ''
		return

	source_file_name = ''
	documents = st.session_state.get( 'documents' ) or [ ]

	for document in documents:
		metadata = getattr( document, 'metadata', None )

		if not isinstance( metadata, dict ) or metadata.get( 'loader' ) != active_loader:
			continue

		source = metadata.get( 'source' )
		if isinstance( source, str ) and source.strip( ):
			source_file_name = Path( source ).name
			break

	if not source_file_name:
		uploaded_value = st.session_state.get( upload_key )

		if isinstance( uploaded_value, (list, tuple) ) and uploaded_value:
			first_uploaded = uploaded_value[ 0 ]
			source_file_name = getattr( first_uploaded, 'name', '' )
		elif uploaded_value is not None:
			source_file_name = getattr( uploaded_value, 'name', '' )

	if not source_file_name:
		return

	document_name, collection_name = derive_document_identity( source_file_name )

	if not document_name or not collection_name:
		return

	st.session_state.source_file_name = Path( source_file_name ).name
	st.session_state.document_name = document_name
	st.session_state.collection_name = collection_name

def validate_remote_vector_state( ) -> tuple[ bool, str ]:
	"""Validate canonical chunk and embedding alignment for cloud persistence.

	Purpose:
		Prevents Chroma or Pinecone writes unless the currently generated embeddings were created
		from the canonical chunk list and remain positionally aligned with those chunks.

	Returns:
		tuple[bool, str]: Validation result and an explanatory message when validation fails.
	"""
	chunked_documents = st.session_state.get( 'chunked_documents' )
	embedding_texts = st.session_state.get( 'embedding_texts' )
	embeddings = st.session_state.get( 'embeddings' )
	embedding_source = st.session_state.get( 'embedding_source' )

	if embedding_source != 'Chunked Documents':
		return False, 'Generate embeddings from Chunked Documents before cloud persistence.'

	if st.session_state.get( 'embedding_is_stale', False ):
		return False, 'The current embeddings are stale. Regenerate them before persistence.'

	if not isinstance( chunked_documents, list ) or not chunked_documents:
		return False, 'Canonical chunk data is unavailable.'

	if not isinstance( embedding_texts, list ) or not embedding_texts:
		return False, 'Embedding source text is unavailable.'

	if not isinstance( embeddings, list ) or not embeddings:
		return False, 'Embedding vectors are unavailable.'

	if len( chunked_documents ) != len( embedding_texts ) or len( embedding_texts ) != len( embeddings ):
		return False, 'Chunk text and embedding counts are not aligned.'

	for chunk_text, embedding_text in zip( chunked_documents, embedding_texts ):
		if chunk_text != embedding_text:
			return False, 'Embedding source text no longer matches the canonical chunk order.'

	vector_array = np.asarray( embeddings, dtype=float )
	if vector_array.ndim != 2 or vector_array.shape[ 0 ] != len( embedding_texts ):
		return False, 'Embedding vectors do not form a valid two-dimensional matrix.'

	if vector_array.shape[ 1 ] < 1 or not np.isfinite( vector_array ).all( ):
		return False, 'Embedding vectors contain an invalid dimension or non-finite values.'

	return True, ''

def build_vector_metadata( index: int, chunk_text: str ) -> dict:
	"""Build shared metadata for a persisted chunk vector.

	Purpose:
		Creates provider-neutral metadata used by both Chroma and Pinecone so persisted vectors
		retain document provenance, chunk configuration, and embedding-model details.

	Args:
		index: Zero-based chunk index associated with the vector.
		chunk_text: Canonical chunk text associated with the vector.

	Returns:
		dict: Metadata values suitable for Chroma and Pinecone records.
	"""
	cfg.throw_if( 'chunk_text', chunk_text )
	
	return { 'document_name': str( st.session_state.get( 'document_name', '' ) ),
			'source_file_name': str( st.session_state.get( 'source_file_name', '' ) ),
			'collection_name': str( st.session_state.get( 'collection_name', '' ) ),
			'chunk_id': int( index + 1 ), 'chunk_index': int( index ), 'chunk_text': chunk_text,
			'embedding_provider': str( st.session_state.get( 'embedding_provider', '' ) ),
			'embedding_model': str( st.session_state.get( 'embedding_model', '' ) ),
			'vector_dimension': int( st.session_state.get( 'embedding_vector_dimension', 0 ) or 0 ),
			'chunk_mode': str( st.session_state.get( 'chunk_mode_value', '' ) ),
			'chunk_size': int( st.session_state.get( 'chunk_size', 0 ) or 0 ),
			'chunk_overlap': int( st.session_state.get( 'chunk_overlap', 0 ) or 0 ),
			'source_signature': str( st.session_state.get( 'embedding_source_signature', '' ) ), }

def generate_query_embedding( query_text: str ) -> list[ float ]:
	"""Generate one query embedding using the active embedding provider contract.

	Purpose:
		Creates a search vector using the same provider, model, task, and requested dimensions
		that produced the persisted document embeddings.

	Args:
		query_text: Query text submitted for semantic retrieval.

	Returns:
		list[float]: One query vector aligned with the persisted vector dimension.
	"""
	cfg.throw_if( 'query_text', query_text )

	provider = st.session_state.get( 'embedding_provider' )
	model = st.session_state.get( 'embedding_model' )

	if provider == 'OpenAI':
		raw_vectors = GPT( ).embed( [ query_text ], model=model )
	elif provider == 'Gemini':
		task = st.session_state.get( 'embedding_task' )
		dimensions = int( st.session_state.get( 'embedding_dimensions', 0 ) or 0 )
		raw_vectors = Gemini( ).embed( [ query_text ], task=task, model=model,
			dimensions=dimensions )
	else:
		raise ValueError( f'Unsupported embedding provider: {provider}' )

	vector_array = np.asarray( raw_vectors, dtype=float )
	if vector_array.ndim == 1:
		vector_array = vector_array.reshape( 1, -1 )

	if vector_array.ndim != 2 or vector_array.shape[ 0 ] != 1:
		raise ValueError( 'The query embedding provider returned an invalid vector.' )

	return vector_array[ 0 ].tolist( )

def metric_with_tooltip( label: str, value: str, tooltip: str ):
	"""Render a Streamlit metric with optional hover guidance.

	Purpose:
		Displays a metric value beside an informational tooltip icon for Chonky analysis panels.
		The helper keeps metric layout consistent while suppressing tooltip icons for core text
		length metrics that do not require additional explanation.

	Args:
		label: Metric label displayed by Streamlit.
		value: Metric value displayed by Streamlit.
		tooltip: Tooltip text rendered in the hover title attribute.
	"""
	col_metric, col_info = st.columns( [ 0.5, 0.5 ] )
	
	with col_metric:
		st.metric( label, value )
	
	with col_info:
		if label not in [ 'Characters', 'Tokens', 'Unique Tokens', 'Avg Length' ]:
			st.markdown( f"""
	            <span style="
	                cursor: help;
	                font-size: 0.85rem;
	                color:#888;
	                vertical-align: super;
	            " title="{tooltip}">ℹ️ </span>
	            """, unsafe_allow_html=True, )

def normalize_embeddings( emb_array: np.ndarray ) -> np.ndarray:
	"""Normalize an embedding array to two dimensions.

	Purpose:
		Converts a single one-dimensional embedding vector into a row-shaped array so Chonky
		embedding diagnostics, dimensionality reduction, and vector persistence paths can operate
		on a consistent matrix-like structure.

	Args:
		emb_array: NumPy embedding array to normalize.

	Returns:
		np.ndarray: Original array or reshaped two-dimensional embedding array.
	"""
	if isinstance( emb_array, np.ndarray ) and emb_array.ndim == 1:
		return emb_array.reshape( 1, -1 )
	return emb_array

def rebuild_raw_text_from_documents( ) -> str | None:
	"""Rebuild raw text from loaded documents.

	Purpose:
		Reconstructs the shared raw-text buffer from the current Streamlit ``documents`` state
		after
		loader-specific clear operations. The helper preserves downstream processing compatibility
		by joining non-empty ``page_content`` values from remaining LangChain documents.

	Returns:
		str | None: Rebuilt raw text, or ``None`` when no usable document text remains.
	"""
	docs = st.session_state.get( "documents" ) or [ ]
	if not docs:
		return None
	text = '\n\n'.join( d.page_content for d in docs if
		hasattr( d, 'page_content' ) and isinstance( d.page_content,
			str ) and d.page_content.strip( ) )
	return text if text.strip( ) else None

def invalidate_embedding_state( clear_chunks: bool = False ) -> None:
	"""Invalidate generated embedding state.
	
	Purpose:
		Removes vectors, immutable source-text provenance, provider metadata, model metadata,
		diagnostic state, and exported embedding records after an upstream source change.
		Optionally removes canonical chunks and chunk metadata when processed text changes so
		chunks cannot remain associated with a superseded corpus.
	
	Args:
		clear_chunks: Indicates whether canonical chunk data must also be removed.
	
	Returns:
		None: This function updates Streamlit session state.
	"""
	st.session_state.embeddings = None
	st.session_state.embedding_documents = None
	st.session_state.embedding_texts = [ ]
	st.session_state.df_embedding_output = pd.DataFrame( )
	st.session_state.embedding_provider = None
	st.session_state.embedding_model = None
	st.session_state.embedding_task = None
	st.session_state.embedding_dimensions = None
	st.session_state.embedding_vector_dimension = None
	st.session_state.embedding_source_signature = None
	st.session_state.embedding_is_stale = False
	
	if clear_chunks:
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.df_chunk_records = None
		st.session_state.df_chunks = None
		st.session_state.chunk_size = None
		st.session_state.chunk_overlap = None
		st.session_state.chunk_mode_value = None

def chunk_characters( text: str, size: int, overlap: int ) -> List[ str ]:
	"""Chunk text by character windows.

	Purpose:
		Splits source text into overlapping character-based chunks for semantic analysis,
		embedding preparation, and display workflows. The function clamps chunk size and overlap
		to safe values before producing non-empty chunk strings.

	Args:
		text: Source text to chunk.
		size: Maximum number of characters in each chunk.
		overlap: Number of overlapping characters between adjacent chunks.

	Returns:
		List[str]: Character-based chunk strings.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	size = int( size )
	overlap = int( overlap )
	size = max( 1, size )
	overlap = max( 0, min( overlap, size - 1 ) )
	
	step = size - overlap
	chunks: list[ str ] = [ ]
	
	i = 0
	n = len( text )
	while i < n:
		part = text[ i: i + size ].strip( )
		if part:
			chunks.append( part )
		i += step
	
	return chunks

def chunk_tokens( text: str, size: int, overlap: int,
		encoding_name: str = 'cl100k_base' ) -> List[ str ]:
	"""
	Purpose:
		Splits source text into overlapping model-token windows using the selected
		tiktoken encoding. Each returned string represents one canonical chunk used
		for display, CSV export, embedding generation, and vector persistence.

	Args:
		text: Source text to divide into token windows.
		size: Maximum number of model tokens in each chunk.
		overlap: Number of model tokens repeated between adjacent chunks.
		encoding_name: Tiktoken encoding used to encode and decode the source text.

	Returns:
		List[str]: Decoded model-token chunks in source order.

	Raises:
		ValueError: Raised when size is less than one or overlap is not smaller than size.
	"""
	import tiktoken
	
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	size = int( size )
	overlap = int( overlap )
	
	if size < 1:
		raise ValueError( 'Chunk size must be greater than zero.' )
	
	if overlap < 0:
		raise ValueError( 'Chunk overlap cannot be negative.' )
	
	if overlap >= size:
		raise ValueError( 'Chunk overlap must be smaller than chunk size.' )
	
	encoding = tiktoken.get_encoding( encoding_name )
	token_ids = encoding.encode( text, disallowed_special=( ) )
	
	if not token_ids:
		return [ ]
	
	step = size - overlap
	chunks: List[ str ] = [ ]
	
	for start in range( 0, len( token_ids ), step ):
		window = token_ids[ start: start + size ]
		
		if not window:
			continue
		
		chunk_text = encoding.decode( window )
		
		if chunk_text:
			chunks.append( chunk_text )
		
		if start + size >= len( token_ids ):
			break
	
	return chunks

def build_chunk_records( chunks: List[ str ], mode: str, configured_size: int,
	configured_overlap: int, preview_length: int = 250,
	source_text: str | None = None ) -> pd.DataFrame:
	"""
	Build canonical chunk records.

	Purpose:
		Creates the canonical chunk dataframe used by the Chunked Data table,
		Chunk Summary table, CSV export, validation metrics, and downstream
		embedding diagnostics. Each record preserves the complete chunk text
		while adding token, character, word, sentence, and preview measurements.

	Args:
		chunks: Canonical chunk strings in source order.
		mode: Chunking mode used to generate the chunks.
		configured_size: Chunk size selected by the user.
		configured_overlap: Chunk overlap selected by the user.
		preview_length: Maximum number of characters retained in each preview.

	Returns:
		pd.DataFrame: Canonical chunk records with one row per chunk.
	"""
	if not isinstance( chunks, list ) or not chunks:
		return pd.DataFrame(
			columns=[ 'Chunk ID', 'Chunk Text', 'Token Count', 'Character Count', 'Word Count',
				'Sentence Count', 'Chunk Preview', 'Chunk Mode', 'Configured Size',
				'Configured Overlap', ] )
	
	import tiktoken
	
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	records: List[ dict ] = [ ]
	
	for index, chunk in enumerate( chunks, start=1 ):
		if not isinstance( chunk, str ) or not chunk.strip( ):
			continue
		
		chunk_text = chunk.strip( )
		token_ids = encoding.encode( chunk_text, disallowed_special=( ), )
		
		try:
			words = word_tokenize( chunk_text )
		except LookupError:
			words = re.findall( r"\b[\w'-]+\b", chunk_text, flags=re.UNICODE, )
		
		try:
			sentences = sent_tokenize( chunk_text )
		except LookupError:
			sentences = [ value.strip( ) for value in re.split( r'(?<=[.!?])\s+', chunk_text, ) if
				value.strip( ) ]
		
		preview = re.sub( r'\s+', ' ', chunk_text, ).strip( )
		
		if len( preview ) > int( preview_length ):
			preview = f'{preview[ : int( preview_length ) ].rstrip( )}…'
		
		records.append(
			{ 'Chunk ID': index, 'Chunk Text': chunk_text, 'Token Count': len( token_ids ),
				'Character Count': len( chunk_text ), 'Word Count': len(
				[ word for word in words if isinstance( word, str ) and word.strip( ) ] ),
				'Sentence Count': len( sentences ), 'Chunk Preview': preview,
				'Chunk Mode': str( mode ), 'Configured Size': int( configured_size ),
				'Configured Overlap': int( configured_overlap ), } )
	
	return pd.DataFrame( records )

def rebuild_token_text( tokens: List[ str ] ) -> str:
	"""Rebuild display text from token values.

	Purpose:
		Reconstructs readable text from tokenized output while avoiding spaces before common
		sentence delimiters and normalizing punctuation spacing. The helper supports Chonky's
		tokenization and processed-text display panels.

	Args:
		tokens: Token values produced by tokenization, stemming, or lemmatization.

	Returns:
		str: Reconstructed display text with normalized punctuation spacing.
	"""
	if not isinstance( tokens, list ) or not tokens:
		return ''
	
	delimiters = { '.', '?', '!', ';', ':', ',', ')' }
	openers = { '(', '[', '{' }
	parts: List[ str ] = [ ]
	for token in tokens:
		if not isinstance( token, str ):
			continue
		
		value = token.strip( )
		if not value:
			continue
		
		if not parts:
			parts.append( value )
		elif value in delimiters:
			parts[ -1 ] = f'{parts[ -1 ]}{value}'
		elif parts[ -1 ] in openers:
			parts.append( value )
		else:
			parts.append( f' {value}' )
	
	text = ''.join( parts )
	text = re.sub( r'\s+', ' ', text ).strip( )
	text = re.sub( r'\s+([.!?;:,)])', r'\1', text )
	text = re.sub( r'([.!?;:])(?=\w)', r'\1 ', text )
	return text

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config( page_title='Chonky', layout='wide', page_icon=cfg.ICON,
	initial_sidebar_state='collapsed' )

# ======================================================================================
# Headers/Title
# ======================================================================================
st.logo( cfg.LOGO, size='large' )

# ======================================================================================
# Sidebar — API Key Configuration
# ======================================================================================
style_subheaders( )

if not isinstance( st.session_state.get( 'openai_api_key' ), str ):
	st.session_state.openai_api_key = ''

if not st.session_state.openai_api_key.strip( ):
	configured_openai_key = getattr( cfg, 'OPENAI_API_KEY', '' )
	
	if isinstance( configured_openai_key, str ) and configured_openai_key.strip( ):
		st.session_state.openai_api_key = configured_openai_key.strip( )

if not isinstance( st.session_state.get( 'gemini_api_key' ), str ):
	st.session_state.gemini_api_key = ''

if not st.session_state.gemini_api_key.strip( ):
	configured_gemini_key = getattr( cfg, 'GEMINI_API_KEY', '' )
	
	if isinstance( configured_gemini_key, str ) and configured_gemini_key.strip( ):
		st.session_state.gemini_api_key = configured_gemini_key.strip( )

if not isinstance( st.session_state.get( 'groq_api_key' ), str ):
	st.session_state.groq_api_key = ''

if not st.session_state.groq_api_key.strip( ):
	configured_groq_key = getattr( cfg, 'GROQ_API_KEY', '' )
	
	if isinstance( configured_groq_key, str ) and configured_groq_key.strip( ):
		st.session_state.groq_api_key = configured_groq_key.strip( )

if not isinstance( st.session_state.get( 'google_api_key' ), str ):
	st.session_state.google_api_key = ''

if not st.session_state.google_api_key.strip( ):
	configured_google_key = getattr( cfg, 'GOOGLE_API_KEY', '' )
	
	if isinstance( configured_google_key, str ) and configured_google_key.strip( ):
		st.session_state.google_api_key = configured_google_key.strip( )

if not isinstance( st.session_state.get( 'pinecone_api_key' ), str ):
	st.session_state.pinecone_api_key = ''

if not st.session_state.pinecone_api_key.strip( ):
	configured_pinecone_key = getattr( cfg, 'PINECONE_API_KEY', '' )
	
	if isinstance( configured_pinecone_key, str ) and configured_pinecone_key.strip( ):
		st.session_state.pinecone_api_key = configured_pinecone_key.strip( )

if not isinstance( st.session_state.get( 'chroma_api_key' ), str ):
	st.session_state.chroma_api_key = ''

if not st.session_state.chroma_api_key.strip( ):
	configured_chroma_key = getattr( cfg, 'CHROMA_API_KEY', '' )

	if isinstance( configured_chroma_key, str ) and configured_chroma_key.strip( ):
		st.session_state.chroma_api_key = configured_chroma_key.strip( )

if not isinstance( st.session_state.get( 'chroma_tenant' ), str ):
	st.session_state.chroma_tenant = ''

if not st.session_state.chroma_tenant.strip( ):
	configured_chroma_tenant = getattr( cfg, 'CHROMA_TENANT', '' ) or getattr(
		cfg, 'CHROMA_TENET_ID', '' )

	if isinstance( configured_chroma_tenant, str ) and configured_chroma_tenant.strip( ):
		st.session_state.chroma_tenant = configured_chroma_tenant.strip( )

if not isinstance( st.session_state.get( 'chroma_database' ), str ):
	st.session_state.chroma_database = ''

if not st.session_state.chroma_database.strip( ):
	configured_chroma_database = getattr( cfg, 'CHROMA_DATABASE', '' )

	if isinstance( configured_chroma_database, str ) and configured_chroma_database.strip( ):
		st.session_state.chroma_database = configured_chroma_database.strip( )

if not isinstance( st.session_state.get( 'google_application_credentials' ), str ):
	st.session_state.google_application_credentials = ''

if not st.session_state.google_application_credentials.strip( ):
	configured_google_credentials = getattr( cfg, 'GOOGLE_APPLICATION_CREDENTIALS', '' )
	
	if (isinstance( configured_google_credentials,
			str ) and configured_google_credentials.strip( )):
		st.session_state.google_application_credentials = (configured_google_credentials.strip( ))

with st.sidebar:
	st.text( 'Settings' )
	st.divider( )
	
	with st.expander( '🔐 API Keys', expanded=False ):
		
		st.text_input( 'OpenAI API Key', type='password', key='openai_api_key' )
		
		st.text_input( 'Groq API Key', type='password', key='groq_api_key' )
		
		st.text_input( 'Gemini API Key', type='password', key='gemini_api_key' )
		
		st.text_input( 'Google API Key', type='password', key='google_api_key' )
		
		st.text_input( 'Google Application Credentials (JSON Path)', type='password',
			key='google_application_credentials' )
		
		st.text_input( 'Pinecone API Key', type='password', key='pinecone_api_key' )

		st.text_input( 'Chroma API Key', type='password', key='chroma_api_key' )

		st.text_input( 'Chroma Tenant ID', type='password', key='chroma_tenant',
			help='Loaded from CHROMA_TENANT or the legacy CHROMA_TENET_ID environment variable.' )

		st.text_input( 'Chroma Database', key='chroma_database',
			help='Loaded from the CHROMA_DATABASE environment variable when available.' )

# ======================================================================================
# Tabs
# ======================================================================================
tabs = st.tabs( cfg.TABS )

# ======================================================================================
# Tab - Document Loading
# ======================================================================================
with tabs[ 0 ]:
	tokens = st.session_state[ 'tokens' ]
	documents = st.session_state[ 'documents' ]
	raw_text = st.session_state[ 'raw_text' ]
	
	# -------------- LEFT COLUMN - LOADERS
	left, right = st.columns( [ 0.35, 0.65 ], gap='medium' )
	with left:
		_loader_msg = st.session_state.pop( '_loader_status', None )
		if isinstance( _loader_msg, str ) and _loader_msg.strip( ):
			st.success( _loader_msg )
		
		with st.expander( label='Local Documents', expanded=True ):
			
			# -------------- NLTK Loader Expander
			with st.expander( label='Corpora Loader', icon='📚', expanded=False ):
				import nltk
				from nltk.corpus import (brown, gutenberg, reuters, webtext, inaugural,
				                         state_union)
				
				st.markdown( '###### NLTK Corpora' )
				corpus_name = st.selectbox( 'Select corpus',
					[ 'Brown', 'Gutenberg', 'Reuters', 'WebText', 'Inaugural',
						'State of the Union', ], key='nltk_corpus_name', )
				
				file_ids = [ ]
				try:
					if corpus_name == 'Brown':
						file_ids = brown.fileids( )
					elif corpus_name == 'Gutenberg':
						file_ids = gutenberg.fileids( )
					elif corpus_name == 'Reuters':
						file_ids = reuters.fileids( )
					elif corpus_name == 'WebText':
						file_ids = webtext.fileids( )
					elif corpus_name == 'Inaugural':
						file_ids = inaugural.fileids( )
					elif corpus_name == 'State of the Union':
						file_ids = state_union.fileids( )
				except LookupError:
					st.error( "NLTK corpus not found. Run:\n\npython -m nltk.downloader all\n\n"
					          "or download individual corpora." )
				
				selected_files = st.multiselect( 'Select files (leave empty to load all)',
					options=file_ids, key='nltk_file_ids', )
				
				st.divider( )
				st.markdown( '###### Local Corpus' )
				local_corpus_dir = st.text_input( 'Local directory',
					placeholder='path/to/text/files', key='nltk_local_dir', )
				 
				# -------------- Load / Clear / Save controls
				col_load, col_clear, col_save = st.columns( 3 )
				load_nltk = col_load.button( label='Load', key='nltk_load',
					icon='📤', width='stretch' )
				
				clear_nltk = col_clear.button( label='Clear', key='nltk_clear',
					icon='🧹', width='stretch' )
				
				_docs = st.session_state.get( 'documents' ) or [ ]
				_nltk_docs = [ d for d in _docs if d.metadata.get( 'loader' ) == 'NLTKLoader' ]
				_nltk_text = "\n\n".join( d.page_content for d in _nltk_docs )
				_export_name = f"nltk_{corpus_name.lower( ).replace( ' ', '_' )}.txt"
				col_save.download_button( 'Save', data=_nltk_text, file_name=_export_name,
					mime='text/plain', disabled=not bool( _nltk_text.strip( ) ),
					icon='💾', width='stretch' )
				 
				# -------------- Clear
				if clear_nltk and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( 'loader' ) != 'NLTKLoader' ]
					
					st.session_state.raw_text = ("\n\n".join( d.page_content for d in
						st.session_state.documents ) if st.session_state.documents else None)
					
					st.session_state.active_loader = None
					st.info( 'NLTKLoader documents removed.' )
				 
				# -------------- Load
				if load_nltk:
					documents = [ ]
					if file_ids:
						files_to_load = selected_files or file_ids
						for fid in files_to_load:
							try:
								if corpus_name == 'Brown':
									text = ' '.join( brown.words( fid ) )
								elif corpus_name == 'Gutenberg':
									text = gutenberg.raw( fid )
								elif corpus_name == 'Reuters':
									text = reuters.raw( fid )
								elif corpus_name == 'WebText':
									text = webtext.raw( fid )
								elif corpus_name == 'Inaugural':
									text = inaugural.raw( fid )
								elif corpus_name == 'State of the Union':
									text = state_union.raw( fid )
								
								if text.strip( ):
									documents.append( Document( page_content=text,
										metadata={ 'loader': 'NLTKLoader', 'corpus': corpus_name,
											'file_id': fid, }, ) )
							except Exception:
								continue
					
					# Local corpus
					if local_corpus_dir and os.path.isdir( local_corpus_dir ):
						for fname in os.listdir( local_corpus_dir ):
							path = os.path.join( local_corpus_dir, fname )
							if os.path.isfile( path ) and fname.lower( ).endswith( '.txt' ):
								with open( path, 'r', encoding='utf-8', errors='ignore' ) as f:
									text = f.read( )
								
								if text.strip( ):
									documents.append( Document( page_content=text,
										metadata={ 'loader': 'NLTKLoader', 'source': path, }, ) )
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in st.session_state.documents )
						
						st.session_state.processed_text = None
						st.session_state.active_loader = 'NLTKLoader'
						
						st.success( f'Loaded {len( documents )} document(s) from NLTK.' )
					else:
						st.warning( 'No documents were loaded.' )
			
			# -------------- Text Loader
			with st.expander( label='Text Loader', icon='📝', expanded=False ):
				files = st.file_uploader( 'Upload Text File(s)', type=[ 'txt', 'text', 'log' ],
					accept_multiple_files=True, key='txt_upload' )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_txt = col_load.button( label='Load', key='txt_load', icon='📤', width='stretch' )
				clear_txt = col_clear.button( label='Clear', key='txt_clear', icon='🧹', width='stretch' )
				can_save = (st.session_state.get( 'active_loader' ) == 'TextLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ) )
				
				if can_save:
					col_save.download_button( label='Save', data=st.session_state.get( 'raw_text' ),
						file_name='text_loader_output.txt', mime='text/plain', key='txt_save',
						icon='💾', width='stretch' )
				else:
					col_save.button( label='Save', key='txt_save_disabled', disabled=True,
						icon='💾', width='stretch' )
				 
				# -------------- Clear
				if clear_txt:
					clear_if_active( 'TextLoader' )
					st.info( 'Text Loader state cleared.' )
					st.rerun( )
				 
				# -------------- Load
				if load_txt and files:
					documents: list[ Document ] = [ ]
					
					with tempfile.TemporaryDirectory( ) as tmp:
						for uploaded_file in files:
							path = os.path.join( tmp, uploaded_file.name )
							
							with open( path, 'wb' ) as handle:
								handle.write( uploaded_file.read( ) )
							
							loader = TextLoader( )
							loaded = loader.load( path ) or [ ]
							for document in loaded:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'TextLoader'
								document.metadata.setdefault( 'source', uploaded_file.name )
							
							documents.extend( loaded )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.active_loader = 'TextLoader'
					st.success( f'Loaded {len( documents )} text document(s).' )
			
			# ------------ CSV Loader Expander
			with st.expander( label='CSV Loader', icon='📑', expanded=False ):
				csv_file = st.file_uploader( label="Upload CSV", type=[ "csv" ], key="csv_upload" )
				delimiter = st.text_input( "Delimiter", value=",", key="csv_delim", )
				quotechar = st.text_input( "Quote Character", value='"', key="csv_quote", )
				
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_csv = col_load.button( 'Load', key='csv_load', icon='📤' )
				clear_csv = col_clear.button( 'Clear', key='csv_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'CsvLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='csv_loader_output.txt', mime='text/plain', key='csv_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='csv_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_csv:
					clear_if_active( "CsvLoader" )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "CSV Loader state cleared."
				 
				# -------------- Load
				if load_csv and csv_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, csv_file.name )
						with open( path, "wb" ) as f:
							f.write( csv_file.read( ) )
						
						loader = CsvLoader( )
						documents = loader.load( path, columns=None, delimiter=delimiter,
							quotechar=quotechar, ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
						hasattr( d, "page_content" ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.active_loader = "CsvLoader"
					
					st.session_state[ "_loader_status" ] = \
						f"Loaded {len( documents )} CSV document(s)."
			
			# -------------- XML Loader Expander
			with st.expander( label='XML Loader', icon='🧬', expanded=False ):
				# -------------- Session-backed loader instance
				if 'xml_loader' not in st.session_state or st.session_state.xml_loader is None:
					st.session_state.xml_loader = XmlLoader( )
				
				loader = st.session_state.xml_loader
				
				xml_file = st.file_uploader( label='Select XML file', type=[ 'xml' ],
					accept_multiple_files=False, key='xml_file_uploader' )
				
				st.text( 'Semantic XML Loading (Unstructured)' )
				
				col1, col2 = st.columns( 2 )
				
				with col1:
					chunk_size = st.number_input( 'Chunk Size', min_value=100, max_value=5000,
						value=1000, step=100 )
				
				with col2:
					overlap_amount = st.number_input( 'Chunk Overlap', min_value=0, max_value=1000,
						value=200, step=50 )
				 
				# -------------- Semantic Load
				if st.button( 'Load XML (Semantic)', use_container_width=True ):
					if xml_file is None:
						st.warning( 'Please select an XML file.' )
					else:
						with tempfile.TemporaryDirectory( ) as tmp:
							path = os.path.join( tmp, xml_file.name )
							with open( path, 'wb' ) as f:
								f.write( xml_file.read( ) )
							
							with st.spinner( 'Loading XML via UnstructuredXMLLoader...' ):
								documents = loader.load( path )
						
						if documents:
							raw_text = '\n\n'.join( d.page_content for d in documents if
								hasattr( d, 'page_content' ) and isinstance( d.page_content,
									str ) and d.page_content.strip( ) )
							
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
							st.session_state.raw_text = raw_text
							st.session_state.processed_text = None
							st.session_state.active_loader = 'XmlLoader'
							st.session_state[ 'xml_documents' ] = documents
							st.session_state[ 'xml_tree_loaded' ] = False
							st.session_state[ 'xml_xpath_results' ] = None
							st.session_state[ 'xml_namespaces' ] = None
						else:
							st.warning( 'No extractable text found in XML.' )
				 
				# -------------- Split Semantic Documents
				if st.button( 'Split Semantic Documents', use_container_width=True ):
					with st.spinner( 'Splitting documents...' ):
						split_docs = loader.split( size=int( chunk_size ),
							amount=int( overlap_amount ) )
					
					if split_docs:
						st.session_state[ 'xml_split_documents' ] = split_docs
						st.success( f'Produced {len( split_docs )} document chunks.' )
				 
				# -------------- Structured XML Tree Loading
				st.divider( )
				st.text( 'Structured XML Tree Loading (XPath)' )
				
				if st.button( 'Load XML Tree', use_container_width=True ):
					if xml_file is None:
						st.warning( 'Please select an XML file.' )
					else:
						with tempfile.TemporaryDirectory( ) as tmp:
							path = os.path.join( tmp, xml_file.name )
							with open( path, 'wb' ) as f:
								f.write( xml_file.read( ) )
							
							with st.spinner( 'Parsing XML into ElementTree...' ):
								tree = loader.load_tree( path )
						
						if tree is not None:
							xml_text = etree.tostring( tree, pretty_print=True,
								encoding='unicode' )
							
							st.session_state.raw_text = xml_text
							st.session_state.processed_text = None
							st.session_state.active_loader = 'XmlLoader'
							st.session_state[ 'xml_tree_loaded' ] = True
							st.session_state[ 'xml_namespaces' ] = loader.xml_namespaces
							st.session_state[ 'xml_xpath_results' ] = None
							
							st.success( 'XML tree loaded successfully.' )
						else:
							st.warning( 'Failed to parse XML tree.' )
				 
				# -------------- XPath Query Interface
				xml_loader = st.session_state.get( 'xml_loader' )
				
				if xml_loader is None:
					st.info( 'No loader initialized.' )
				elif not hasattr( xml_loader, 'xml_root' ):
					st.info( 'XML loader does not support XML tree operations.' )
				elif xml_loader.xml_root is None:
					st.info( 'XML loader initialized but no XML tree loaded.' )
				else:
					st.markdown( '**XPath Query**' )
					
					xpath_expr = st.text_input( 'XPath Expression', value='//*',
						help='Use namespace prefixes if applicable.' )
					
					if st.button( 'Run XPath Query', use_container_width=True ):
						with st.spinner( 'Executing XPath...' ):
							elements = xml_loader.get_elements( xpath_expr )
						
						if elements is not None:
							st.session_state[ 'xml_xpath_results' ] = elements
							st.success( f'Returned {len( elements )} elements.' )
					
					if 'xml_xpath_results' in st.session_state and st.session_state[
						'xml_xpath_results' ] is not None:
						preview_count = min( 10, len( st.session_state[ 'xml_xpath_results' ] ) )
						
						st.caption( f'Previewing first {preview_count} elements' )
						
						for el in st.session_state[ 'xml_xpath_results' ][ :preview_count ]:
							st.code( etree.tostring( el, pretty_print=True, encoding='unicode' ),
								language='xml' )
				 
				# -------------- Debug / Introspection
				with st.expander( "ℹ Loader State" ):
					xml_loader = st.session_state.get( 'xml_loader' )
					
					if xml_loader is None:
						st.info( "No loader initialized." )
					else:
						st.json( { "file_path": getattr( xml_loader, 'file_path', None ),
							"documents_loaded": getattr( xml_loader, 'documents',
								None ) is not None,
							"xml_tree_loaded": getattr( xml_loader, 'xml_tree', None ) is not None,
							"namespaces": getattr( xml_loader, 'xml_namespaces', None ),
							"chunk_size": getattr( xml_loader, 'chunk_size', None ),
							"overlap_amount": getattr( xml_loader, 'overlap_amount', None ), } )
			
			# --------------- Word Loader
			with st.expander( label='Word Document Loader', icon='📘', expanded=False ):
				word_file = st.file_uploader( 'Upload Word Document', type=[ 'docx' ],
					key='word_upload', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_word = col_load.button( 'Load', key='word_load', icon='📤' )
				clear_word = col_clear.button( 'Clear', key='word_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'WordLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='word_loader_output.txt', mime='text/plain', key='word_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='word_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_word:
					clear_if_active( 'WordLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Word Loader state cleared.'
				 
				# -------------- Load
				if load_word and word_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, word_file.name )
						with open( path, 'wb' ) as f:
							f.write( word_file.read( ) )
						
						loader = WordLoader( )
						documents = loader.load( path ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'WordLoader'
						document.metadata.setdefault( 'source', word_file.name )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'WordLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} Word document(s).'
			
			# -------------- PDF Loader Expander
			with st.expander( label='PDF Loader', icon='📕', expanded=False ):
				pdf = st.file_uploader( 'Upload PDF', type=[ 'pdf' ], key='pdf_upload' )
				
				mode = st.selectbox( 'Mode', [ 'single', 'page' ], key='pdf_mode',
					help='Used only when legacy extraction is enabled.' )
				
				extract = st.selectbox( 'Extract', [ 'plain', 'layout' ], key='pdf_extract',
					help='Used only when legacy extraction is enabled.' )
				
				include = st.checkbox( 'Include Images', value=False, key='pdf_include',
					help='Used only when legacy extraction is enabled.' )
				
				fmt = st.selectbox( 'Format', [ 'markdown-img', 'html-img', 'text-img' ],
					key='pdf_fmt', help='Used only when legacy extraction is enabled.' )
				
				use_geometry = st.checkbox( 'Use Geometry Extraction', value=True,
					key='pdf_use_geometry',
					help='Uses PyMuPDF block coordinates for layout-aware extraction only.' )
				
				use_legacy_pdf_loader = st.checkbox( 'Use Legacy PdfLoader', value=False,
					key='pdf_use_legacy_loader',
					help='Falls back to the existing PdfLoader path instead of geometry '
					     'extraction.' )
				
				band_left, band_right = st.columns( 2, border=True )
				
				with band_left:
					header_band = st.slider( 'Header Band', min_value=0, max_value=30, value=8,
						step=1, key='pdf_header_band',
						help='Percentage of page height classified as the top candidate header '
						     'band.' )
				
				with band_right:
					footer_band = st.slider( 'Footer Band', min_value=0, max_value=30, value=8,
						step=1, key='pdf_footer_band',
						help='Percentage of page height classified as the bottom candidate footer '
						     'band.' )
				
				preserve_page_breaks = st.checkbox( 'Preserve Page Breaks', value=False,
					key='pdf_preserve_page_breaks',
					help='Adds explicit page-break markers between extracted pages.' )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_pdf = col_load.button( 'Load', key='pdf_load', icon='📤' )
				clear_pdf = col_clear.button( 'Clear', key='pdf_clear', icon='🧹' )
				save_pdf = col_save.empty( )
				
				# -------------- Clear
				if clear_pdf:
					clear_if_active( 'PdfLoader' )
					st.session_state.pdf_pages = None
					st.session_state[ '_loader_status' ] = 'PDF Loader state cleared.'
				 
				# -------------- Load
				if load_pdf and pdf:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, pdf.name )
						
						with open( path, 'wb' ) as f:
							f.write( pdf.read( ) )
						
						if use_geometry and not use_legacy_pdf_loader:
							parser = PdfParser( )
							pdf_pages = parser.extract_pages( path=path,
								header_ratio=float( header_band ) / 100.0,
								footer_ratio=float( footer_band ) / 100.0 ) or [ ]
							
							raw_text = parser.rebuild_pages( pages=pdf_pages,
								preserve_page_breaks=preserve_page_breaks ) or ''
							
							documents = [ Document( page_content=raw_text,
								metadata={ 'loader': 'PdfLoader', 'source': pdf.name,
									'extract': 'geometry', 'header_band': int( header_band ),
									'footer_band': int( footer_band ),
									'preserve_page_breaks': preserve_page_breaks, } ) ]
							
							st.session_state.pdf_pages = pdf_pages
						else:
							loader = PdfLoader( )
							documents = loader.load( path, mode=mode, extract=extract,
								include=include, format=fmt ) or [ ]
							
							for document in documents:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'PdfLoader'
								document.metadata.setdefault( 'source', pdf.name )
								document.metadata.setdefault( 'extract', 'legacy' )
							
							raw_text = '\n\n'.join( d.page_content for d in documents if
								hasattr( d, 'page_content' ) and isinstance( d.page_content,
									str ) and d.page_content.strip( ) )
							
							st.session_state.pdf_pages = None
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = raw_text
					st.session_state.processed_text = None
					st.session_state.displayed_text = ''
					st.session_state.processed_text_display = ''
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'PdfLoader'
					
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} PDF document(s).'
				 
				# -------------- Save
				can_save = (st.session_state.get( 'active_loader' ) == 'PdfLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					save_pdf.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='pdf_loader_output.txt', mime='text/plain', key='pdf_save',
						icon='💾' )
				else:
					save_pdf.button( 'Save', key='pdf_save_disabled', disabled=True, icon='💾' )
			
			# --------------- Power Point Loader
			with st.expander( label='Power Point Loader', icon='📽', expanded=False ):
				pptx = st.file_uploader( 'Upload PPTX', type=[ 'pptx' ], key='pptx_upload', )
				mode = st.selectbox( 'Mode', [ 'single', 'elements' ], key='pptx_mode', )
				
				# -------------- Buttons: Load / Clear / Save (same row, same style)
				col_load, col_clear, col_save = st.columns( 3 )
				load_pptx = col_load.button( 'Load', key='pptx_load', icon='📤' )
				
				clear_pptx = col_clear.button( 'Clear', key='pptx_clear', icon='🧹' )
				
				# ---------- Save
				can_save = (st.session_state.get(
					'active_loader' ) == 'PowerPointLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='powerpoint_loader_output.txt', mime='text/plain',
						key='pptx_save', icon='💾' )
				else:
					col_save.button( 'Save', key='pptx_save_disabled', disabled=True, icon='💾' )
				
				# ---------- Clear
				if clear_pptx:
					clear_if_active( 'PowerPointLoader' )
					st.info( 'PowerPoint Loader state cleared.' )
				
				# ---------- Load
				if load_pptx and pptx:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, pptx.name )
						with open( path, "wb" ) as f:
							f.write( pptx.read( ) )
						
						loader = PowerPointLoader( )
						documents = loader.load( path, mode=mode ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.active_loader = "PowerPointLoader"
					st.success( f"Loaded {len( documents )} PowerPoint document(s)." )
			
			# -------------- Jupyter Notebook Loader
			with st.expander( label='Jupyter Notebook Loader', icon='🪐', expanded=False ):
				notebook_file = st.file_uploader( 'Upload Notebook', type=[ 'ipynb' ],
					key='ipynb_upload', )
				
				include_outputs = st.checkbox( 'Include Outputs', value=False,
					key='ipynb_include_outputs', )
				
				max_output_length = st.number_input( 'Max Output Length', min_value=1, value=10,
					step=1, key='ipynb_max_output_length', )
				
				remove_newline = st.checkbox( 'Remove Newline', value=False,
					key='ipynb_remove_newline', )
				
				include_traceback = st.checkbox( 'Include Traceback', value=False,
					key='ipynb_traceback', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_ipynb = col_load.button( 'Load', key='ipynb_load', icon='📤' )
				clear_ipynb = col_clear.button( 'Clear', key='ipynb_clear', icon='🧹' )
				
				can_save = (st.session_state.get(
					'active_loader' ) == 'JupyterNotebookLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='jupyter_notebook_loader_output.txt', mime='text/plain',
						key='ipynb_save', icon='💾' )
				else:
					col_save.button( 'Save', key='ipynb_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_ipynb:
					clear_if_active( 'JupyterNotebookLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Jupyter Notebook Loader state cleared.'
				 
				# -------------- Load
				if load_ipynb and notebook_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, notebook_file.name )
						with open( path, 'wb' ) as f:
							f.write( notebook_file.read( ) )
						
						loader = JupyterNotebookLoader( )
						documents = loader.load( path=path, include_outputs=include_outputs,
							max_output_length=int( max_output_length ),
							remove_newline=remove_newline, traceback=include_traceback, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'JupyterNotebookLoader'
						document.metadata.setdefault( 'source', notebook_file.name )
						document.metadata.setdefault( 'include_outputs', include_outputs )
						document.metadata.setdefault( 'max_output_length',
							int( max_output_length ) )
						document.metadata.setdefault( 'remove_newline', remove_newline )
						document.metadata.setdefault( 'traceback', include_traceback )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'JupyterNotebookLoader'
					st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} notebook document(s).'
			
			# -------------- Excel Loader
			with st.expander( label='Excel Loader', icon='📊', expanded=False ):
				excel_file = st.file_uploader( 'Upload Excel file', type=[ 'xlsx', 'xls' ],
					key='excel_upload', )
				
				load_mode = st.selectbox( 'Load Mode',
					[ 'Tabular + SQLite', 'Unstructured Document' ], index=0,
					key='excel_load_mode',
					help=(
						'Use "Tabular + SQLite" to preserve the current sheet-to-SQLite workflow. '
						'Use "Unstructured Document" to route through ExcelLoader.'), )
				
				sheet_name = st.text_input( 'Sheet name (leave blank for all sheets)',
					key='excel_sheet' )
				
				table_prefix = st.text_input( 'table prefix', value='excel',
					help='Each sheet will be written as <prefix>_<sheetname>',
					key='excel_table_prefix' )
				
				unstructured_mode = st.selectbox( 'Document Mode', [ 'single', 'elements' ],
					index=0, key='excel_unstructured_mode',
					help='Used only with "Unstructured Documents".' )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_excel = col_load.button( 'Load', key='excel_load', icon='📤' )
				clear_excel = col_clear.button( 'Clear', key='excel_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'ExcelLoader' and
				            isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='excel_loader_output.txt', mime='text/plain', key='excel_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='excel_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear (remove only ExcelLoader documents)
				if clear_excel and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( 'loader' ) != 'ExcelLoader' ]
					
					st.session_state.raw_documents = [ d for d in st.session_state.documents if
						isinstance( getattr( d, 'metadata', None ),
							dict ) ] if st.session_state.documents else [ ]
					
					st.session_state.raw_text = ('\n\n'.join(
						d.page_content for d in st.session_state.documents if
							isinstance( d.page_content,
								str ) and d.page_content.strip( ) ) if st.session_state.documents
					                             else None)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = None
					
					st.info( "ExcelLoader documents removed." )
				 
				# -------------- Load
				if load_excel and excel_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						excel_path = os.path.join( tmp, excel_file.name )
						with open( excel_path, "wb" ) as f:
							f.write( excel_file.read( ) )
						
						documents = [ ]
						if load_mode == 'Tabular + SQLite':
							sqlite_path = os.path.join( "stores", "sqlite", "data.db" )
							os.makedirs( os.path.dirname( sqlite_path ), exist_ok=True )
							
							if sheet_name.strip( ):
								dfs = { sheet_name: pd.read_excel( excel_path,
									sheet_name=sheet_name, ) }
							else:
								dfs = pd.read_excel( excel_path, sheet_name=None, )
							
							conn = sqlite3.connect( sqlite_path )
							try:
								for sheet, df in dfs.items( ):
									if df.empty:
										continue
									table_name = f"{table_prefix}_{sheet}".replace( " ",
										"_" ).lower( )
									df.to_sql( table_name, conn, if_exists="replace",
										index=False, )
									text = df.to_csv( index=False )
									documents.append( Document( page_content=text,
										metadata={ 'loader': 'ExcelLoader',
											'source': excel_file.name, 'sheet': sheet,
											'table': table_name, 'sqlite_db': sqlite_path,
											'load_mode': 'Tabular + SQLite', }, ) )
							finally:
								conn.close( )
						
						else:
							loader = ExcelLoader( )
							documents = loader.load( excel_path, mode=unstructured_mode,
								has_headers=True ) or [ ]
							
							for document in documents:
								if not isinstance( getattr( document, 'metadata', None ), dict ):
									document.metadata = { }
								
								document.metadata[ 'loader' ] = 'ExcelLoader'
								document.metadata.setdefault( 'source', excel_file.name )
								document.metadata[ 'load_mode' ] = 'Unstructured Document'
								document.metadata[ 'document_mode' ] = unstructured_mode
					
					if documents:
						existing_documents = st.session_state.get( 'documents' )
						if isinstance( existing_documents, list ) and existing_documents:
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = list( documents )
						
						st.session_state.raw_documents = list( st.session_state.documents )
						st.session_state.raw_text = "\n\n".join(
							d.page_content for d in st.session_state.documents if
								isinstance( d.page_content, str ) and d.page_content.strip( ) )
						
						st.session_state.processed_text = None
						st.session_state.active_loader = 'ExcelLoader'
						
						if load_mode == 'Tabular + SQLite':
							st.success(
								f"Loaded {len( documents )} sheet(s) and stored in SQLite." )
						else:
							st.success(
								f"Loaded {len( documents )} Excel {unstructured_mode!r} mode." )
					else:
						if load_mode == 'Tabular + SQLite':
							st.warning( "No data loaded (empty sheets or invalid selection)." )
						else:
							st.warning( "No Excel document content was loaded." )
			
			# --------------------------- Markdown Loader
			with st.expander( label='Markdown Loader', icon='🧾', expanded=False ):
				md = st.file_uploader( 'Upload Markdown', type=[ 'md', 'markdown' ],
					key='md_upload', )
				
				mode = st.selectbox( 'Mode', [ 'single', 'elements' ], index=0, key='md_mode',
					help='Use "single" for one combined document or "elements" for element '
					     'parsing.' )
				 
				# -------------- Buttons: Load / Clear / Save (same row, same style)
				col_load, col_clear, col_save = st.columns( 3 )
				load_md = col_load.button( 'Load', key='md_load', icon='📤' )
				clear_md = col_clear.button( 'Clear', key='md_clear', icon='🧹' )
				
				# Save enabled only when MarkdownLoader is active and raw_text exists
				can_save = (
						st.session_state.get( 'active_loader' ) == 'MarkdownLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='markdown_loader_output.txt', mime='text/plain', key='md_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='md_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_md:
					clear_if_active( 'MarkdownLoader' )
					st.info( "Markdown Loader state cleared." )
				 
				# -------------- Load (same behavior, now with explicit mode)
				if load_md and md:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, md.name )
						with open( path, "wb" ) as f:
							f.write( md.read( ) )
						
						loader = MarkdownLoader( )
						documents = loader.load( path, mode=mode ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.active_loader = "MarkdownLoader"
					
					st.success( f"Loaded {len( documents )} Markdown document(s)." )
			
			# ------------- HTML Loader
			with st.expander( label='HTML Loader', icon='🌐', expanded=False ):
				html = st.file_uploader( 'Upload HTML', type=[ 'html', 'htm' ], key='html_upload' )
				
				# -------------- Buttons: Load / Clear / Save (same row, same style)
				col_load, col_clear, col_save = st.columns( 3 )
				load_html = col_load.button( 'Load', key='html_load', icon='📤' )
				clear_html = col_clear.button( 'Clear', key='html_clear', icon='🧹' )
				
				# Save enabled only when HtmlLoader is active and raw_text exists
				can_save = (st.session_state.get( 'active_loader' ) == 'HtmlLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='html_loader_output.txt', mime='text/plain', key='html_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='html_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_html:
					clear_if_active( "HtmlLoader" )
					st.info( "HTML Loader state cleared." )
				 
				# -------------- Load
				if load_html and html:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, html.name )
						with open( path, "wb" ) as f:
							f.write( html.read( ) )
						
						loader = HtmlLoader( )
						documents = loader.load( path )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
					st.session_state.active_loader = "HtmlLoader"
					st.success( f"Loaded {len( documents )} HTML document(s)." )
			
			# -------------- JSON Loader
			with st.expander( label='JSON Loader', icon='🧩', expanded=False ):
				js = st.file_uploader( 'Upload JSON', type=[ 'json', 'jsonl' ],
					key='json_upload', )
				
				jq_schema = st.text_input( 'jq Schema', value='.', key='json_jq_schema',
					help='Examples: ., .[], .messages[], .content' )
				
				content_key = st.text_input( 'Content Key (optional)', value='',
					key='json_content_key',
					help='Use when jq_schema returns objects and you want one field as '
					     'page_content.' )
				
				is_lines = st.checkbox( 'JSON Lines', value=False, key='json_lines', )
				
				is_text = st.checkbox( 'Extracted content is already text', value=True,
					key='json_text_content',
					help='Turn this off when jq_schema/content_key selects structured values '
					     'instead of plain text.' )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_json = col_load.button( 'Load', key='json_load', icon='📤' )
				clear_json = col_clear.button( 'Clear', key='json_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'JsonLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='json_loader_output.txt', mime='text/plain', key='json_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='json_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_json:
					clear_if_active( 'JsonLoader' )
					st.info( 'JSON Loader state cleared.' )
				 
				# -------------- Load
				if load_json and js:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, js.name )
						with open( path, 'wb' ) as f:
							f.write( js.read( ) )
						
						loader = JsonLoader( )
						documents = loader.load( path, jq_schema=jq_schema,
							content_key=content_key,
							is_text=is_text, is_lines=is_lines, ) or [ ]
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = "\n\n".join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.active_loader = "JsonLoader"
					st.success( f"Loaded {len( documents )} JSON document(s)." )
		
		with st.expander( label='Web Documents', expanded=False ):
			
			# ----------- ArXiv Loader
			with st.expander( label='ArXiv Loader', icon='🧠', expanded=False ):
				arxiv_query = st.text_input( 'Query', placeholder='e.g., transformer OR llm',
					key='arxiv_query', )
				
				arxiv_max_chars = st.number_input( 'Max characters per document', min_value=250,
					max_value=100000, value=1000, step=250, key='arxiv_max_chars',
					help='Maximum characters read', )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				arxiv_fetch = col_fetch.button( 'Load', key='arxiv_fetch', icon='📤' )
				arxiv_clear = col_clear.button( 'Clear', key='arxiv_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'ArXivLoader' and
				            isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='arxiv_loader_output.txt', mime='text/plain', key='arxiv_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='arxiv_save_disabled', disabled=True, icon='💾'  )
				
				if arxiv_clear and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( 'loader' ) != 'ArXivLoader' ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'ArXivLoader documents removed.'
				
				if arxiv_fetch and arxiv_query:
					loader = ArXivLoader( )
					documents = loader.load( arxiv_query, max_chars=int( arxiv_max_chars ), ) or [ ]
					
					for d in documents:
						d.metadata[ 'loader' ] = 'ArXivLoader'
						d.metadata[ 'source' ] = arxiv_query
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = 'ArXivLoader'
						
						st.session_state[ '_loader_status' ] = f'Fetched {len( documents )} document(s).'
			
			# -------------- Wikipedia Loader
			with st.expander( label='Wikipedia Loader', icon='📚', expanded=False ):
				wiki_query = st.text_input( 'Query',
					placeholder='e.g., Natural language processing', key='wiki_query', )
				
				wiki_max_docs = st.number_input( 'Max documents', min_value=1, max_value=250,
					value=25, step=1, key='wiki_max_docs',
					help='Maximum number of documents loaded', )
				
				wiki_max_chars = st.number_input( 'Max characters per document', min_value=250,
					max_value=100000, value=4000, step=250, key='wiki_max_chars',
					help='Upper limit on the number of characters', )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				wiki_fetch = col_fetch.button( 'Load', key='wiki_fetch', icon='📤' )
				wiki_clear = col_clear.button( 'Clear', key='wiki_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'WikiLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='wiki_loader_output.txt', mime='text/plain', key='wiki_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='wiki_save_disabled', disabled=True, icon='💾' )
				
				if wiki_clear and st.session_state.get( 'documents' ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( 'loader' ) != 'WikiLoader' ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'WikiLoader documents removed.'
				
				if wiki_fetch and wiki_query:
					loader = WikiLoader( )
					documents = loader.load( wiki_query, max_docs=int( wiki_max_docs ),
						max_chars=int( wiki_max_chars ), ) or [ ]
					
					for d in documents:
						d.metadata[ 'loader' ] = 'WikiLoader'
						d.metadata[ 'source' ] = wiki_query
					
					if documents:
						if st.session_state.get( 'documents' ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = 'WikiLoader'
						
						st.session_state[ '_loader_status' ] = (
							f'Fetched {len( documents )} Wikipedia document('
							f's).')
			
			# ----------- GitHub Loader
			with st.expander( label='GitHub Loader', icon='🐙', expanded=False ):
				gh_url = st.text_input( "GitHub API URL", placeholder="https://api.github.com",
					value="https://api.github.com", key="gh_url",
					help="GitHub REST API base URL.", )
				
				gh_repo = st.text_input( "Repo (owner/name)", placeholder="openai/openai-python",
					key="gh_repo", help="Name of the repository.", )
				
				gh_branch = st.text_input( "Branch", placeholder="main", value="main",
					key="gh_branch", help="The branch of the repository.", )
				
				gh_filetype = st.text_input( "File type filter", value=".md", key="gh_filetype",
					help="Filtering by file type. Example: .py, .md, .txt", )
				
				gh_access_token = st.text_input( "GitHub Access Token (optional)", value="",
					type="password", key="gh_access_token",
					help="Optional personal access token. Useful for private repos or higher rate "
					     "limits.", )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				gh_fetch = col_fetch.button( "Load", key="gh_fetch", icon='📤' )
				gh_clear = col_clear.button( "Clear", key="gh_clear", icon='🧹' )
				
				can_save = (
						st.session_state.get( "active_loader" ) == "GithubLoader" and isinstance(
					st.session_state.get( "raw_text" ), str ) and st.session_state.get(
					"raw_text" ).strip( ))
				
				if can_save:
					col_save.download_button( "Save", data=st.session_state.get( "raw_text" ),
						file_name="github_loader_output.txt", mime="text/plain", key="gh_save",
						icon='💾' )
				else:
					col_save.button( "Save", key="gh_save_disabled", disabled=True, icon='💾' )
				
				if gh_clear and st.session_state.get( "documents" ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( "loader" ) != "GithubLoader" ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "GithubLoader documents removed."
				
				if gh_fetch and gh_repo and gh_branch:
					loader = GithubLoader( )
					documents = loader.load( gh_url, gh_repo, gh_branch, gh_filetype,
						gh_access_token, ) or [ ]
					
					for d in documents:
						if not isinstance( getattr( d, "metadata", None ), dict ):
							d.metadata = { }
						d.metadata[ "loader" ] = "GithubLoader"
						d.metadata[ "source" ] = f"{gh_repo}@{gh_branch}"
					
					if documents:
						if st.session_state.get( "documents" ):
							st.session_state.documents.extend( documents )
						else:
							st.session_state.documents = documents
							st.session_state.raw_documents = list( documents )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = "GithubLoader"
						
						st.session_state[ "_loader_status" ] = f"Fetched {len( documents )} GitHub document(s)."
			
			# ----------- Outlook Loader
			with st.expander( label='Outlook Loader', icon='📨', expanded=False ):
				outlook_file = st.file_uploader( 'Upload Outlook Message', type=[ 'msg' ],
					key='outlook_upload', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_outlook = col_load.button( 'Load', key='outlook_load', icon='📤' )
				clear_outlook = col_clear.button( 'Clear', key='outlook_clear', icon='🧹' )
				
				can_save = (
						st.session_state.get( 'active_loader' ) == 'OutlookLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='outlook_loader_output.txt', mime='text/plain',
						key='outlook_save', icon='💾' )
				else:
					col_save.button( 'Save', key='outlook_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_outlook:
					clear_if_active( 'OutlookLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Outlook Loader state cleared.'
				 
				# -------------- Load
				if load_outlook and outlook_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, outlook_file.name )
						with open( path, 'wb' ) as f:
							f.write( outlook_file.read( ) )
						
						loader = OutlookLoader( )
						documents = loader.load( path ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OutlookLoader'
						document.metadata.setdefault( 'source', outlook_file.name )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OutlookLoader'
					st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} Outlook message document('
						f's).')
			
			# ------------- Web Loader
			with st.expander( label="Web Loader", icon='🌐', expanded=False ):
				urls = st.text_area( "Enter one URL per line",
					placeholder="https://example.com\nhttps://another.com", key="web_urls", )
				
				web_timeout = st.number_input( "Timeout (seconds)", min_value=1, max_value=120,
					value=10, step=1, key="web_timeout", )
				
				web_ignore = st.checkbox( "Continue On Failure", value=True, key="web_ignore",
					help="Keep loading remaining URLs if one page fails." )
				
				col_fetch, col_clear, col_save = st.columns( 3 )
				load_web = col_fetch.button( "Load", key="web_fetch", icon='📤' )
				clear_web = col_clear.button( "Clear", key="web_clear", icon='🧹' )
				
				can_save = (st.session_state.get( "active_loader" ) == "WebLoader" and isinstance(
					st.session_state.get( "raw_text" ), str ) and st.session_state.get(
					"raw_text" ).strip( ))
				
				if can_save:
					col_save.download_button( "Save", data=st.session_state.get( "raw_text" ),
						file_name="web_loader_output.txt", mime="text/plain", key="web_save",
						icon='💾' )
				else:
					col_save.button( "Save", key="web_save_disabled", disabled=True, icon='💾' )
				
				if clear_web and st.session_state.get( "documents" ):
					st.session_state.documents = [ d for d in st.session_state.documents if
						d.metadata.get( "loader" ) != "WebLoader" ]
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ "_loader_status" ] = "WebLoader documents removed."
				
				if load_web and urls.strip( ):
					loader = WebLoader( recursive=False )
					new_docs = [ ]
					
					for url in [ u.strip( ) for u in urls.splitlines( ) if u.strip( ) ]:
						documents = loader.load( urls=url, timeout=int( web_timeout ),
							ignore=bool( web_ignore ), progress=True ) or [ ]
						
						for d in documents:
							if not isinstance( getattr( d, "metadata", None ), dict ):
								d.metadata = { }
							d.metadata[ "loader" ] = "WebLoader"
							d.metadata[ "source" ] = url
						
						new_docs.extend( documents )
					
					if new_docs:
						if st.session_state.get( "documents" ):
							st.session_state.documents.extend( new_docs )
						else:
							st.session_state.documents = new_docs
							st.session_state.raw_documents = list( new_docs )
						
						st.session_state.raw_text = rebuild_raw_text_from_documents( )
						st.session_state.active_loader = "WebLoader"
						
						st.session_state[
							"_loader_status" ] = f"Fetched {len( new_docs )} web document(s)."
			
			# -------------- Web Crawler
			with st.expander( label='Web Crawler', icon='🕷️', expanded=False ):
				start_url = st.text_input( 'Start URL', placeholder='https://example.com',
					key='crawl_start_url', )
				
				max_depth = st.number_input( 'Max crawl depth', min_value=1, max_value=5, value=2,
					step=1, key='crawl_depth', )
				
				crawl_timeout = st.number_input( 'Timeout (seconds)', min_value=1, max_value=120,
					value=10, step=1, key='crawl_timeout', )
				
				stay_on_domain = st.checkbox( 'Stay on starting domain', value=True,
					key='crawl_domain_lock', )
				
				col_run, col_clear, col_save = st.columns( 3 )
				run_crawl = col_run.button( 'Load', key='crawl_run', icon='📤' )
				clear_crawl = col_clear.button( 'Clear', key='crawl_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'WebCrawler' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='web_crawler_output.txt', mime='text/plain', key='crawl_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='crawl_save_disabled', disabled=True, icon='💾' )
				
				if clear_crawl:
					clear_if_active( 'WebCrawler' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Web Crawler state cleared.'
				
				if run_crawl and isinstance( start_url, str ) and start_url.strip( ):
					loader = WebCrawler( url=start_url.strip( ), recursive=True,
						max_depth=int( max_depth ), prevent_outside=bool( stay_on_domain ),
						timeout=int( crawl_timeout ), ignore=True, progress=True, )
					
					documents = loader.load( urls=start_url.strip( ), depth=int( max_depth ),
						timeout=int( crawl_timeout ), ignore=True, progress=True,
						prevent_outside=bool( stay_on_domain ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'WebCrawler'
						document.metadata.setdefault( 'source', start_url.strip( ) )
						document.metadata.setdefault( 'max_depth', int( max_depth ) )
						document.metadata.setdefault( 'timeout', int( crawl_timeout ) )
						document.metadata.setdefault( 'prevent_outside', bool( stay_on_domain ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'WebCrawler'
					st.session_state[ '_loader_status' ] = f'Crawled {len( documents )} document(s).'
			
			# ------------- Email Loader
			with st.expander( label='E-mail Loader', icon='📧', expanded=False ):
				email_file = st.file_uploader( 'Upload Email File', type=[ 'eml' ],
					key='email_upload', )
				
				email_mode = st.selectbox( 'Mode', options=[ 'elements', 'single' ], index=0,
					key='email_mode', )
				
				email_attachments = st.checkbox( 'Process Attachments', value=False,
					key='email_attachments', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_email = col_load.button( 'Load', key='email_load', icon='📤' )
				clear_email = col_clear.button( 'Clear', key='email_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'EmailLoader' and
				            isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='email_loader_output.txt', mime='text/plain', key='email_save',
						icon='💾' )
				else:
					col_save.button( 'Save', key='email_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_email:
					clear_if_active( 'EmailLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Email Loader state cleared.'
				 
				# -------------- Load
				if load_email and email_file:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, email_file.name )
						with open( path, 'wb' ) as f:
							f.write( email_file.read( ) )
						
						loader = EmailLoader( )
						documents = loader.load( path=path, mode=email_mode,
							attachments=email_attachments, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'EmailLoader'
						document.metadata.setdefault( 'source', email_file.name )
						document.metadata.setdefault( 'mode', email_mode )
						document.metadata.setdefault( 'attachments', email_attachments )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'EmailLoader'
					st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} email document(s).'
			
			# ----------------- PubMed Loader
			with st.expander( label='Pub Med Loader', icon='🧬', expanded=False ):
				pubmed_query = st.text_input( 'PubMed Query', value='', key='pubmed_query',
					placeholder='e.g. transformer models biomedical NLP', )
				
				pubmed_max_docs = st.number_input( 'Maximum Documents', min_value=1, value=5,
					step=1, key='pubmed_max_docs', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_pubmed = col_load.button( 'Load', key='pubmed_load', icon='📥' )
				clear_pubmed = col_clear.button( 'Clear', key='pubmed_clear', icon='🧹' )
				
				can_save = (st.session_state.get(
					'active_loader' ) == 'PubMedSearchLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='pubmed_loader_output.txt', mime='text/plain',
						key='pubmed_save', icon='💾' )
				else:
					col_save.button( 'Save', key='pubmed_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_pubmed:
					clear_if_active( 'PubMedSearchLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'PubMed Loader state cleared.'
				 
				# -------------- Load
				if load_pubmed and isinstance( pubmed_query, str ) and pubmed_query.strip( ):
					loader = PubMedSearchLoader( )
					documents = loader.load( query=pubmed_query.strip( ),
						max_docs=int( pubmed_max_docs ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'PubMedSearchLoader'
						document.metadata.setdefault( 'query', pubmed_query.strip( ) )
						document.metadata.setdefault( 'max_docs', int( pubmed_max_docs ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'PubMedSearchLoader'
					st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} PubMed document(s).'
			
			# -------------- Open City Loader
			with st.expander( label='Open City Loader', icon='🏙️', expanded=False ):
				open_city_id = st.text_input( 'City Domain', value='', key='open_city_id',
					placeholder='e.g. data.sfgov.org',
					help='City domain identifier for the Socrata-backed portal.', icon='💾' )
				
				open_city_dataset_id = st.text_input( 'Dataset ID', value='',
					key='open_city_dataset_id', placeholder='e.g. vw6y-z8j6',
					help='Dataset identifier from the city portal.', icon='💾' )
				
				open_city_limit = st.number_input( 'Maximum Records', min_value=1, value=100,
					step=1, key='open_city_limit', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_open_city = col_load.button( 'Load', key='open_city_load', icon='📥' )
				clear_open_city = col_clear.button( 'Clear', key='open_city_clear', icon='🧹' )
				
				can_save = (
						st.session_state.get( 'active_loader' ) == 'OpenCityLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='open_city_loader_output.txt', mime='text/plain',
						key='open_city_save', icon='💾' )
				else:
					col_save.button( 'Save', key='open_city_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_open_city:
					clear_if_active( 'OpenCityLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Open City Loader state cleared.'
				 
				# -------------- Load
				if (load_open_city and isinstance( open_city_id,
						str ) and open_city_id.strip( ) and isinstance( open_city_dataset_id,
					str ) and open_city_dataset_id.strip( )):
					loader = OpenCityLoader( )
					documents = loader.load( city_id=open_city_id.strip( ),
						dataset_id=open_city_dataset_id.strip( ),
						limit=int( open_city_limit ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OpenCityLoader'
						document.metadata.setdefault( 'city_id', open_city_id.strip( ) )
						document.metadata.setdefault( 'dataset_id', open_city_dataset_id.strip( ) )
						document.metadata.setdefault( 'limit', int( open_city_limit ) )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OpenCityLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} Open City document(s).'
		
		with st.expander( label='Cloud Documents', expanded=False ):
			
			# ------------ OneDrive Loader
			with st.expander( label='OneDrive Loader', icon='🟦', expanded=False ):
				onedrive_drive_id = st.text_input( 'Drive ID', value='', key='onedrive_drive_id',
					placeholder='OneDrive drive identifier', )
				
				onedrive_folder_path = st.text_input( 'Folder Path', value='',
					key='onedrive_folder_path', placeholder='Optional folder path within the '
					                                        'drive',
					help='Leave blank to load the drive target directly.', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_onedrive = col_load.button( 'Load', key='onedrive_load', icon='📥' )
				clear_onedrive = col_clear.button( 'Clear', key='onedrive_clear', icon='🧹' )
				
				can_save = (st.session_state.get(
					'active_loader' ) == 'OneDriveDocLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='onedrive_loader_output.txt', mime='text/plain',
						key='onedrive_save', icon='💾' )
				else:
					col_save.button( 'Save', key='onedrive_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_onedrive:
					clear_if_active( 'OneDriveDocLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'OneDrive Loader state cleared.'
				 
				# -------------- Load
				if (load_onedrive and isinstance( onedrive_drive_id,
						str ) and onedrive_drive_id.strip( )):
					loader = OneDriveDocLoader( )
					
					if isinstance( onedrive_folder_path, str ) and onedrive_folder_path.strip( ):
						documents = loader.load_folder( id=onedrive_drive_id.strip( ),
							path=onedrive_folder_path.strip( ), ) or [ ]
					else:
						documents = loader.load( id=onedrive_drive_id.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'OneDriveDocLoader'
						document.metadata.setdefault( 'drive_id', onedrive_drive_id.strip( ) )
						document.metadata.setdefault( 'folder_path',
							onedrive_folder_path.strip( ) or None, )
						
						if onedrive_folder_path.strip( ):
							document.metadata.setdefault( 'source',
								f"{onedrive_drive_id.strip( )}:{onedrive_folder_path.strip( )}" )
						else:
							document.metadata.setdefault( 'source', onedrive_drive_id.strip( ), )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'OneDriveDocLoader'
					st.session_state[ '_loader_status' ] = \
						f'Loaded {len( documents )} OneDrive document(s).'
			
			# --------------------------- Google Cloud File Loader
			with st.expander( label='Google Cloud File Loader', icon='☁️', expanded=False ):
				gcs_project_name = st.text_input( 'Project Name', value='',
					key='gcs_file_project_name', placeholder='e.g. my-gcp-project', )
				
				gcs_bucket = st.text_input( 'Bucket', value='', key='gcs_file_bucket',
					placeholder='e.g. my-bucket', )
				
				gcs_blob = st.text_input( 'Blob', value='', key='gcs_file_blob',
					placeholder='e.g. documents/report.pdf', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_gcs_file = col_load.button( 'Load', key='gcs_file_load', icon='📥' )
				clear_gcs_file = col_clear.button( 'Clear', key='gcs_file_clear', icon='🧹' )
				
				can_save = (st.session_state.get(
					'active_loader' ) == 'GoogleCloudFileLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='google_cloud_file_loader_output.txt', mime='text/plain',
						key='gcs_file_save', icon='💾' )
				else:
					col_save.button( 'Save', key='gcs_file_save_disabled', disabled=True, icon='💾' )
				 
				# -------------- Clear
				if clear_gcs_file:
					clear_if_active( 'GoogleCloudFileLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = ('Google Cloud File Loader state '
					                                        'cleared.')
				
				# -------------- Load
				if (load_gcs_file and isinstance( gcs_project_name,
						str ) and gcs_project_name.strip( ) and isinstance( gcs_bucket,
					str ) and gcs_bucket.strip( ) and isinstance( gcs_blob,
					str ) and gcs_blob.strip( )):
					loader = GoogleCloudFileLoader( )
					documents = loader.load( project_name=gcs_project_name.strip( ),
						bucket=gcs_bucket.strip( ), blob=gcs_blob.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'GoogleCloudFileLoader'
						document.metadata.setdefault( 'project_name', gcs_project_name.strip( ) )
						document.metadata.setdefault( 'bucket', gcs_bucket.strip( ) )
						document.metadata.setdefault( 'blob', gcs_blob.strip( ) )
						document.metadata.setdefault( 'source',
							f"gs://{gcs_bucket.strip( )}/{gcs_blob.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'GoogleCloudFileLoader'
					st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} Google Cloud file '
						f'document(s).')
			
			# --------------------------- AWS File Loader
			with st.expander( label='AWS File Loader', icon='🪣', expanded=False ):
				aws_file_bucket = st.text_input( 'Bucket', value='', key='aws_file_bucket',
					placeholder='e.g. my-s3-bucket', )
				
				aws_file_key = st.text_input( 'Object Key', value='', key='aws_file_key',
					placeholder='e.g. documents/report.pdf', )
				
				aws_file_region = st.text_input( 'Region Name', value='', key='aws_file_region',
					placeholder='e.g. us-east-1', )
				
				aws_file_api_version = st.text_input( 'API Version', value='',
					key='aws_file_api_version', placeholder='Optional', )
				
				aws_file_use_ssl = st.checkbox( 'Use SSL', value=True, key='aws_file_use_ssl', )
				
				aws_file_verify = st.text_input( 'Verify', value='', key='aws_file_verify',
					placeholder='Optional path or True / False',
					help='Leave blank for default behavior, or provide a CA bundle path.', )
				
				aws_file_endpoint_url = st.text_input( 'Endpoint URL', value='',
					key='aws_file_endpoint_url', placeholder='Optional custom endpoint', )
				 
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_aws_file = col_load.button( label='Load', key='aws_file_load', icon='📤' )
				clear_aws_file = col_clear.button( label='Clear', key='aws_file_clear', icon='🧹' )
				can_save = (st.session_state.get(
					'active_loader' ) == 'AwsFileLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='aws_file_loader_output.txt', mime='text/plain',
						key='aws_file_save', icon='📥' )
				else:
					col_save.button( 'Save', key='aws_file_save_disabled', disabled=True, icon='💾' )
				
				# -------------- Clear
				if clear_aws_file:
					clear_if_active( 'AwsFileLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'AWS File Loader state cleared.'
				
				# -------------- Load
				if (load_aws_file and isinstance( aws_file_bucket,
						str ) and aws_file_bucket.strip( ) and isinstance( aws_file_key,
					str ) and aws_file_key.strip( )):
					verify_value: str | bool | None = None
					
					if isinstance( aws_file_verify, str ) and aws_file_verify.strip( ):
						verify_text = aws_file_verify.strip( )
						
						if verify_text.lower( ) == 'true':
							verify_value = True
						elif verify_text.lower( ) == 'false':
							verify_value = False
						else:
							verify_value = verify_text
					
					loader = AwsFileLoader( )
					documents = loader.load( bucket=aws_file_bucket.strip( ),
						key=aws_file_key.strip( ), region_name=aws_file_region.strip( ) or None,
						api_version=aws_file_api_version.strip( ) or None,
						use_ssl=bool( aws_file_use_ssl ), verify=verify_value,
						endpoint_url=aws_file_endpoint_url.strip( ) or None, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'AwsFileLoader'
						document.metadata.setdefault( 'bucket', aws_file_bucket.strip( ) )
						document.metadata.setdefault( 'key', aws_file_key.strip( ) )
						document.metadata.setdefault( 'region_name',
							aws_file_region.strip( ) or None )
						document.metadata.setdefault( 'api_version',
							aws_file_api_version.strip( ) or None )
						document.metadata.setdefault( 'use_ssl', bool( aws_file_use_ssl ) )
						document.metadata.setdefault( 'verify', verify_value )
						document.metadata.setdefault( 'endpoint_url',
							aws_file_endpoint_url.strip( ) or None )
						document.metadata.setdefault( 'source',
							f"s3://{aws_file_bucket.strip( )}/{aws_file_key.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'AwsFileLoader'
					st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} AWS file document(s).'
			
			# -------- Google Bucket Loader
			with st.expander( label='Google Bucket Loader', icon='🗂️', expanded=False ):
				gcs_bucket_project_name = st.text_input( 'Project Name', value='',
					key='gcs_bucket_project_name', placeholder='e.g. my-gcp-project', )
				
				gcs_bucket_name = st.text_input( 'Bucket', value='', key='gcs_bucket_name',
					placeholder='e.g. my-bucket', )
				
				gcs_bucket_prefix = st.text_input( 'Prefix', value='', key='gcs_bucket_prefix',
					placeholder='Optional folder / object prefix', )
				
				gcs_bucket_continue_on_failure = st.checkbox( 'Continue On Failure', value=False,
					key='gcs_bucket_continue_on_failure', )
				
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_gcs_bucket = col_load.button( 'Load', key='gcs_bucket_load', icon='📥' )
				clear_gcs_bucket = col_clear.button( 'Clear', key='gcs_bucket_clear', icon='🧹' )
				
				can_save = (st.session_state.get(
					'active_loader' ) == 'GoogleBucketLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='google_bucket_loader_output.txt', mime='text/plain',
						key='gcs_bucket_save', icon='💾' )
				else:
					col_save.button( 'Save', key='gcs_bucket_save_disabled', disabled=True, icon='💾' )
				
				# -------------- Clear
				if clear_gcs_bucket:
					clear_if_active( 'GoogleBucketLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'Google Bucket Loader state cleared.'
				
				# -------------- Load
				if (load_gcs_bucket and isinstance( gcs_bucket_project_name,
						str ) and gcs_bucket_project_name.strip( ) and isinstance( gcs_bucket_name,
					str ) and gcs_bucket_name.strip( )):
					loader = GoogleBucketLoader( )
					documents = loader.load( project_name=gcs_bucket_project_name.strip( ),
						bucket=gcs_bucket_name.strip( ), prefix=gcs_bucket_prefix.strip( ) or None,
						continue_on_failure=bool( gcs_bucket_continue_on_failure ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'GoogleBucketLoader'
						document.metadata.setdefault( 'project_name',
							gcs_bucket_project_name.strip( ), )
						document.metadata.setdefault( 'bucket', gcs_bucket_name.strip( ), )
						document.metadata.setdefault( 'prefix',
							gcs_bucket_prefix.strip( ) or None, )
						document.metadata.setdefault( 'continue_on_failure',
							bool( gcs_bucket_continue_on_failure ), )
						
						if gcs_bucket_prefix.strip( ):
							document.metadata.setdefault( 'source',
								f"gs://{gcs_bucket_name.strip( )}/{gcs_bucket_prefix.strip( )}" )
						else:
							document.metadata.setdefault( 'source',
								f"gs://{gcs_bucket_name.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'GoogleBucketLoader'
					st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} Google bucket document('
						f's).')
			
			# --------------------------- AWS Bucket Loader
			with st.expander( label='AWS Bucket Loader', icon='🗃️', expanded=False ):
				aws_bucket_name = st.text_input( 'Bucket', value='', key='aws_bucket_name',
					placeholder='e.g. my-s3-bucket', )
				
				aws_bucket_prefix = st.text_input( 'Prefix', value='', key='aws_bucket_prefix',
					placeholder='Optional folder / object prefix', )
				
				aws_bucket_region = st.text_input( 'Region Name', value='',
					key='aws_bucket_region',
					placeholder='e.g. us-east-1', )
				
				aws_bucket_api_version = st.text_input( 'API Version', value='',
					key='aws_bucket_api_version', placeholder='Optional', )
				
				aws_bucket_use_ssl = st.checkbox( 'Use SSL', value=True,
					key='aws_bucket_use_ssl', )
				
				aws_bucket_verify = st.text_input( 'Verify', value='', key='aws_bucket_verify',
					placeholder='Optional path or True / False',
					help='Leave blank for default behavior, or provide a CA bundle path.', )
				
				aws_bucket_endpoint_url = st.text_input( 'Endpoint URL', value='',
					key='aws_bucket_endpoint_url', placeholder='Optional custom endpoint', )
				
				# -------------- Buttons: Load / Clear / Save
				col_load, col_clear, col_save = st.columns( 3 )
				load_aws_bucket = col_load.button( 'Load', key='aws_bucket_load', icon='📥' )
				clear_aws_bucket = col_clear.button( 'Clear', key='aws_bucket_clear', icon='🧹' )
				
				can_save = (
						st.session_state.get( 'active_loader' ) == 'AwsBucketLoader' and
						isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='aws_bucket_loader_output.txt', mime='text/plain',
						key='aws_bucket_save', icon='💾' )
				else:
					col_save.button( 'Save', key='aws_bucket_save_disabled', disabled=True, icon='💾' )
				
				# -------------- Clear
				if clear_aws_bucket:
					clear_if_active( 'AwsBucketLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'AWS Bucket Loader state cleared.'
				
				# -------------- Load
				if (load_aws_bucket and isinstance( aws_bucket_name, str ) and aws_bucket_name.strip( )):
					verify_value: str = None
					
					if isinstance( aws_bucket_verify, str ) and aws_bucket_verify.strip( ):
						verify_text = aws_bucket_verify.strip( )
						
						if verify_text.lower( ) == 'true':
							verify_value = True
						elif verify_text.lower( ) == 'false':
							verify_value = False
						else:
							verify_value = verify_text
					
					loader = AwsBucketLoader( )
					documents = loader.load( bucket=aws_bucket_name.strip( ),
						prefix=aws_bucket_prefix.strip( ),
						region_name=aws_bucket_region.strip( ) or None,
						api_version=aws_bucket_api_version.strip( ) or None,
						use_ssl=bool( aws_bucket_use_ssl ), verify=verify_value,
						endpoint_url=aws_bucket_endpoint_url.strip( ) or None, ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'AwsBucketLoader'
						document.metadata.setdefault( 'bucket', aws_bucket_name.strip( ) )
						document.metadata.setdefault( 'prefix', aws_bucket_prefix.strip( ) )
						document.metadata.setdefault( 'region_name',
							aws_bucket_region.strip( ) or None )
						document.metadata.setdefault( 'api_version',
							aws_bucket_api_version.strip( ) or None )
						document.metadata.setdefault( 'use_ssl', bool( aws_bucket_use_ssl ) )
						document.metadata.setdefault( 'verify', verify_value )
						document.metadata.setdefault( 'endpoint_url',
							aws_bucket_endpoint_url.strip( ) or None )
						
						if aws_bucket_prefix.strip( ):
							document.metadata.setdefault( 'source',
								f"s3://{aws_bucket_name.strip( )}/{aws_bucket_prefix.strip( )}" )
						else:
							document.metadata.setdefault( 'source',
								f"s3://{aws_bucket_name.strip( )}" )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'AwsBucketLoader'
					st.session_state[
						'_loader_status' ] = f'Loaded {len( documents )} AWS bucket document(s).'
			
			# --------------------------- SharePoint Loader
			with st.expander( label='SharePoint Loader', icon='🟩', expanded=False ):
				spfx_library_id = st.text_input( 'Library ID', value='', key='spfx_library_id',
					placeholder='SharePoint document library identifier', )
				
				spfx_folder_id = st.text_input( 'Folder ID', value='', key='spfx_folder_id',
					placeholder='Optional folder identifier within the library',
					help='Leave blank to load the library directly.', )
				
				# --------------------------------------------------
				# Buttons: Load / Clear / Save
				# --------------------------------------------------
				col_load, col_clear, col_save = st.columns( 3 )
				load_spfx = col_load.button( 'Load', key='spfx_load', icon='📥' )
				clear_spfx = col_clear.button( 'Clear', key='spfx_clear', icon='🧹' )
				
				can_save = (st.session_state.get( 'active_loader' ) == 'SpfxLoader' and isinstance(
					st.session_state.get( 'raw_text' ), str ) and st.session_state.get(
					'raw_text' ).strip( ))
				
				if can_save:
					col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
						file_name='sharepoint_loader_output.txt', mime='text/plain',
						key='spfx_save', icon='💾' )
				else:
					col_save.button( 'Save', key='spfx_save_disabled', disabled=True, icon='💾' )
				
				# -------------- Clear
				if clear_spfx:
					clear_if_active( 'SpfxLoader' )
					st.session_state.raw_text = rebuild_raw_text_from_documents( )
					st.session_state[ '_loader_status' ] = 'SharePoint Loader state cleared.'
				
				# -------------- Load
				if (load_spfx and isinstance( spfx_library_id, str ) and spfx_library_id.strip( )):
					loader = SpfxLoader( )
					if isinstance( spfx_folder_id, str ) and spfx_folder_id.strip( ):
						documents = loader.load_folder( library_id=spfx_library_id.strip( ),
							folder_id=spfx_folder_id.strip( ), ) or [ ]
					else:
						documents = loader.load( library_id=spfx_library_id.strip( ), ) or [ ]
					
					for document in documents:
						if not isinstance( getattr( document, 'metadata', None ), dict ):
							document.metadata = { }
						
						document.metadata[ 'loader' ] = 'SpfxLoader'
						document.metadata.setdefault( 'library_id', spfx_library_id.strip( ) )
						document.metadata.setdefault( 'folder_id',
							spfx_folder_id.strip( ) or None, )
						
						if spfx_folder_id.strip( ):
							document.metadata.setdefault( 'source',
								f"{spfx_library_id.strip( )}:{spfx_folder_id.strip( )}" )
						else:
							document.metadata.setdefault( 'source', spfx_library_id.strip( ), )
					
					st.session_state.documents = documents
					st.session_state.raw_documents = list( documents )
					st.session_state.raw_text = '\n\n'.join( d.page_content for d in documents if
						hasattr( d, 'page_content' ) and isinstance( d.page_content,
							str ) and d.page_content.strip( ) )
					st.session_state.processed_text = None
					st.session_state.lines = None
					st.session_state.chunked_documents = None
					st.session_state.df_chunks = None
					st.session_state.active_loader = 'SpfxLoader'
					st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} SharePoint document(s).'
	
	# -------------- RIGHT COLUMN — DOCUMENT RENDERING
	with right:
		documents = st.session_state.documents
		if not documents:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f'Active Loader: {st.session_state.active_loader}' )
			st.write( f'Documents: {len( documents )}' )
			for i, d in enumerate( documents[ :5 ] ):
				with st.expander( f'Document {i + 1}', expanded=True ):
					st.json( d.metadata )
					st.text_area( 'Content', d.page_content[ : ], height=450,
						key=f'preview_doc_{i}' )
	
	# -------------- NLP METRIC CALCULATIONS
	metrics_container = st.container( )
	with metrics_container:
		if documents is not None:
			# -------------- Tokenization (session-cached)
			if st.session_state.tokens is None:
				try:
					raw_text = rebuild_raw_text_from_documents( )
					tokens = [ t.lower( ) for t in word_tokenize( raw_text ) if t.isalpha( ) ]
				except LookupError:
					st.error( 'NLTK resources missing' )
				
				if not tokens:
					st.warning( 'No valid alphabetic tokens found.' )
				
				st.session_state.tokens = tokens
				st.session_state.vocabulary = set( tokens )
				st.session_state.token_counts = Counter( tokens )
			
			tokens = st.session_state.tokens
			vocabulary = st.session_state.vocabulary
			counts = st.session_state.token_counts
			char_count = len( raw_text )
			token_count = len( tokens )
			vocab_size = len( vocabulary )
			hapax_count = sum( 1 for c in counts.values( ) if c == 1 )
			hapax_ratio = hapax_count / vocab_size if vocab_size else 0.0
			avg_word_len = sum( len( t ) for t in tokens ) / token_count
			ttr = vocab_size / token_count
			stopword_ratio = 0.0
			lexical_density = 0.0
			try:
				stop_words = set( stopwords.words( 'english' ) )
				stopword_ratio = sum( 1 for t in tokens if t in stop_words ) / token_count
				lexical_density = 1.0 - stopword_ratio
			except LookupError:
				pass
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			
			st.markdown( '#### Top Tokens' )
			
			# ------------ Top Tokens
			with st.expander( label='Tokens', icon='🉑', expanded=True ):
				top_tokens = counts.most_common( 10 )
				df_top = pd.DataFrame( top_tokens, columns=[ 'token', 'count' ] ).set_index(
					'token' )
				st.bar_chart( df_top, color='#01438A' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
			st.markdown( '#### Corpus Metrics' )
			
			# ------------ Corpus Metrics
			with st.expander( label='Text', icon='📖', expanded=True ):
				col1, col2, col3, col4 = st.columns( 4, border=True )
				with col1:
					metric_with_tooltip( 'Characters', f'{char_count:,}',
						'Total number of characters in the selected text.' )
				
				with col2:
					metric_with_tooltip( 'Tokens', f'{token_count:,}',
						'Token Count: total number of tokenized words after cleanup.' )
				
				with col3:
					metric_with_tooltip( 'Unique Tokens', f'{vocab_size:,}',
						'Vocabulary Size: number of distinct word types in the text.' )
				
				with col4:
					metric_with_tooltip( 'TTR', f'{ttr:.3f}',
						'Type–Token Ratio: unique words ÷ total words.' )
				
				col5, col6, col7, col8 = st.columns( 4, border=True )
				with col5:
					metric_with_tooltip( 'Hapax Ratio', f'{hapax_ratio:.3f}',
						'Hapax Ratio: proportion of words that occur only once.' )
				
				with col6:
					metric_with_tooltip( 'Avg Length', f'{avg_word_len:.2f}',
						'Average number of characters per token.' )
				
				with col7:
					metric_with_tooltip( 'Stopword Ratio', f'{stopword_ratio:.2%}',
						'Percentage of stopwords in the text.' )
				
				with col8:
					metric_with_tooltip( 'Lexical Density', f'{lexical_density:.2%}',
						'Proportion of content-bearing words.' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			
			st.markdown( '#### Comprehension Metrics' )
			
			# ------------ Readability
			with st.expander( label='Words', icon='👀', expanded=True ):
				if TEXTSTAT_AVAILABLE:
					r1, r2, r3, r4 = st.columns( 4, border=True )
					with r1:
						metric_with_tooltip( 'Flesch Reading Ease',
							f'{textstat.flesch_reading_ease( raw_text ):.1f}',
							'Higher scores indicate easier readability.' )
					
					with r2:
						metric_with_tooltip( 'Flesch–Kincaid Grade',
							f'{textstat.flesch_kincaid_grade( raw_text ):.1f}',
							'Estimated U.S. grade level required.' )
					
					with r3:
						metric_with_tooltip( 'Gunning Fog',
							f'{textstat.gunning_fog( raw_text ):.1f}',
							'Readability based on sentence length and complex words.' )
					
					with r4:
						metric_with_tooltip( 'Coleman–Liau Index',
							f'{textstat.coleman_liau_index( raw_text ):.1f}',
							'Readability based on characters and sentences.' )
				else:
					st.caption( 'Install `textstat` to enable readability metrics.' )

# -------------- Uploaded Document Identity
sync_document_identity( )

# ======================================================================================
# Tab — Text Processing
# ======================================================================================
with tabs[ 1 ]:
	raw_text = st.session_state.get( 'raw_text' )
	active_loader = st.session_state.get( 'active_loader' )
	
	# -------------- Session State Defaults
	st.session_state.setdefault( 'raw_text_view', '' )
	st.session_state.setdefault( 'processed_text', '' )
	st.session_state.setdefault( 'displayed_text', '' )
	st.session_state.setdefault( 'processed_text_display', '' )
	st.session_state.setdefault( 'processing_display_version', 0 )
	st.session_state.setdefault( 'start_time', 0.0 )
	st.session_state.setdefault( 'end_time', 0.0 )
	st.session_state.setdefault( 'total_time', 0.0 )
	st.session_state.setdefault( 'nltk_word_tokens', [ ] )
	st.session_state.setdefault( 'nltk_sentence_tokens', [ ] )
	st.session_state.setdefault( 'nltk_stemmed_tokens', [ ] )
	st.session_state.setdefault( 'nltk_lemmatized_tokens', [ ] )
	st.session_state.setdefault( 'nltk_pos_tags', [ ] )
	st.session_state.setdefault( 'nltk_named_entities', [ ] )
	
	def refresh_processing_display( ) -> None:
		"""
		
			Purpose:
			--------
			Increment the processing display version so Streamlit recreates the read-only
			text-area widgets after processing, reset, or clear actions.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
		
		"""
		st.session_state.processing_display_version = (
				int( st.session_state.get( 'processing_display_version', 0 ) ) + 1)
	
	def coerce_text( value: object ) -> str:
		"""
		
			Purpose:
			--------
			Convert parser output into a display-safe string without allowing None values
			to break later processing steps.
			
			Parameters:
			-----------
			value : object
				The value returned by a processing method.
			
			Returns:
			--------
			str
				The string representation of the value, or an empty string.
		
		"""
		if value is None:
			return ''
		
		if isinstance( value, str ):
			return value
		
		if isinstance( value, list ):
			return '\n'.join(
				str( item ) for item in value if item is not None and str( item ).strip( ) )
		
		return str( value )
	
	if not isinstance( raw_text, str ) or not raw_text.strip( ):
		st.info( 'Load a document before running text processing.' )
	elif not active_loader:
		st.warning( 'No active loader detected. Load documents first.' )
	else:
		st.session_state.raw_text_view = raw_text
		has_text = isinstance( raw_text, str ) and bool( raw_text.strip( ) )
		
		# ------------------------- Layout
		left, right = st.columns( [ 1, 1.5 ], border=True )
		with left:
			active = st.session_state.get( 'active_loader' )
			
			# ==============================================================
			# Common Text Processing (TextParser)
			# ==============================================================
			with st.expander( label='Text Processing', icon='🧠', expanded=True ):
				remove_html = st.checkbox( 'Remove HTML',
					help='Removes Hypertext Markup Tags, eg. <, \\>, etc', value=False )
				
				remove_markdown = st.checkbox( 'Remove Markdown',
					help=r'Removes symbols used in .md files #, ##, ###, -, etc', value=False )
				
				remove_symbols = st.checkbox( 'Remove Symbols',
					help=r'Removes @, #, $, ^, *, =, |, \\, <, >, ~', value=False )
				
				remove_numbers = st.checkbox( 'Remove Numbers',
					help='Removes numeric digits 0 through 9', value=False )
				
				remove_xml = st.checkbox( 'Remove XML',
					help=r'Removes xml tags ( ex. <xml> & <\xml> )', value=False )
				
				remove_punctuation = st.checkbox( 'Remove Punctuation',
					help=r'Removes punctuation while preserving sentence delimiters', value=False )
				
				reduce_repeats = st.checkbox( 'Remove Repeats',
					help='Reduces consecutive punctuation sequences to the left-most mark '
					     'and inserts a following space when needed. Examples: ".." -> ". ", '
					     '"??" -> "? ", "?;" -> "? ".', value=False )
				
				remove_images = st.checkbox( 'Remove Images',
					help=r'Remove images from text, including Markdown, HTML <img> tags, '
					     r'and image URLs', value=True )
				
				remove_stopwords = st.checkbox( 'Remove Stopwords',
					help=r'Removes common words (e.g., "the", "is", "and", etc.)', value=False )
				
				remove_numerals = st.checkbox( 'Remove Numerals',
					help='Removes roman numerals I, II, IV, XI, etc', value=False )
				
				remove_encodings = st.checkbox( 'Remove Encoding',
					help=r'Removes encoding artifacts and over-encoded byte strings', value=True )
				
				normalize_text = st.checkbox( 'Normalize (lowercase)', value=False )
				
				remove_fragments = st.checkbox( 'Remove Fragments',
					help='Removes isolated malformed extraction debris while preserving valid '
					     'short words, acronyms, identifiers, punctuation, and numbers.',
					value=True )
				
				remove_errors = st.checkbox( 'Remove Errors',
					help='Removes replacement characters, null bytes, unsafe control characters, '
					     'and recognized text-encoding artifacts without dictionary filtering.',
					value=True )
				
				collapse_whitespace = st.checkbox( 'Collapse Whitespace',
					help='Converts every consecutive sequence of whitespace characters into one '
					     'ordinary space.', value=True )
			
			# ==============================================================
			# NLTK Processing
			# ==============================================================
			with st.expander( 'NLTK Processing', icon='🪶', expanded=False ):
				nltk_word_tokenize = st.checkbox( 'Word Tokenize',
					help='Splits text into word tokens.' )
				nltk_sentence_tokenize = st.checkbox( 'Sentence Tokenize',
					help='Splits text into sentence tokens.' )
				nltk_stem = st.checkbox( 'Stem Words', help='Applies stemming to word tokens.' )
				nltk_lemmatize = st.checkbox( 'Lemmatize Words',
					help='Applies lemmatization to word tokens.' )
				nltk_pos_tag = st.checkbox( 'Part-of-Speech Tagging',
					help='Applies POS tagging to word tokens.' )
				nltk_named_entities = st.checkbox( 'Named Entity Recognition',
					help='Extracts named entities where available.' )
			
			# ==============================================================
			# -------------- Word-Specific Processing (WordParser)
			# ==============================================================
			extract_tables = False
			extract_paragraphs = False
			
			with st.expander( 'Word Processing', icon='📄', expanded=False ):
				if active == 'WordLoader':
					extract_tables = st.checkbox( 'Extract Tables' )
					extract_paragraphs = st.checkbox( 'Extract Paragraphs' )
				else:
					st.caption( 'Available when Word documents are loaded.' )
			
			# -------------- PDF-Specific Processing
			remove_pdf_repeats = False
			clean_pdf_artifacts = False
			repair_pdf_spacing = False
			rejoin_pdf_hyphenation = False
			repair_embedded_hyphenation = False
			
			with st.expander( 'PDF Processing', icon='📕', expanded=False ):
				if active == 'PdfLoader':
					remove_pdf_repeats = st.checkbox( 'Remove Repeated Marginalia', value=True,
						key='pdf_remove_repeats',
						help='Removes repeated header/footer-band blocks using geometry '
						     'metadata.' )
					
					clean_pdf_artifacts = st.checkbox( 'Clean Artifacts', value=True,
						key='pdf_clean_artifacts',
						help='Removes generic parser artifacts, file paths, image tags, control '
						     'characters, and leader-dot runs.' )
					
					repair_pdf_spacing = st.checkbox( 'Repair Spacing', value=True,
						key='pdf_repair_spacing',
						help='Repairs generic spacing defects without document-specific rules.' )
					
					rejoin_pdf_hyphenation = st.checkbox( 'Rejoin Hyphenation', value=True,
						key='pdf_rejoin_hyphenation',
						help='Repairs line-break hyphenation and soft-hyphen artifacts.' )
					
					repair_embedded_hyphenation = st.checkbox( 'Repair Embedded Hyphen Splits',
						value=True, key='pdf_repair_embedded_hyphenation',
						help='Conservatively repairs embedded alphabetic extraction splits.' )
				else:
					st.caption( 'Available when PDF documents are loaded.' )
			
			# -------------- HTML Processing
			strip_scripts = False
			keep_headings = False
			keep_paragraphs = False
			keep_tables = False
			with st.expander( 'HTML Processing', icon='🌐', expanded=False ):
				if active == 'HtmlLoader':
					strip_scripts = st.checkbox( 'Strip <script> / <style>' )
					keep_headings = st.checkbox( 'Keep Headings' )
					keep_paragraphs = st.checkbox( 'Keep Paragraphs' )
					keep_tables = st.checkbox( 'Keep Tables' )
				else:
					st.caption( 'Available when HTML documents are loaded.' )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
			
			# ------------ Actions (Apply)
			col_apply, col_reset, col_clear, col_save = st.columns( 4 )
			apply_processing = col_apply.button( label='Apply', disabled=not has_text,
				key='processing_apply_button', icon='✔️', width='stretch' )
			reset_processing = col_reset.button( label='Reset', disabled=not has_text,
				key='processing_reset_button', icon='🔁', width='stretch' )
			clear_processing = col_clear.button( label='Clear', disabled=not has_text,
				key='processing_clear_button', icon='🧹', width='stretch' )
			save_processed_slot = col_save.empty( )
			
			# ------------ Button Events
			if reset_processing:
				st.session_state.processed_text = ''
				st.session_state.displayed_text = ''
				st.session_state.processed_text_display = ''
				st.session_state.start_time = 0.0
				st.session_state.end_time = 0.0
				st.session_state.total_time = 0.0
				invalidate_embedding_state( clear_chunks=True )
				refresh_processing_display( )
				st.success( 'Processed text reset.' )
			
			if clear_processing:
				st.session_state.processed_text = ''
				st.session_state.displayed_text = ''
				st.session_state.processed_text_display = ''
				st.session_state.start_time = 0.0
				st.session_state.end_time = 0.0
				st.session_state.total_time = 0.0
				st.session_state.nltk_word_tokens = [ ]
				st.session_state.nltk_sentence_tokens = [ ]
				st.session_state.nltk_stemmed_tokens = [ ]
				st.session_state.nltk_lemmatized_tokens = [ ]
				st.session_state.nltk_pos_tags = [ ]
				st.session_state.nltk_named_entities = [ ]
				invalidate_embedding_state( clear_chunks=True )
				refresh_processing_display( )
				st.success( 'Processed text cleared.' )
			
			if apply_processing:
				start_time = time.perf_counter( )
				
				# ------------  Initialize from raw text
				processed_text = raw_text if isinstance( raw_text, str ) else ''
				tp = TextParser( )
				nlp = NltkParser( )
				
				# ------------ PDF reconstruction and PDF-specific cleanup
				if active == 'PdfLoader':
					pdf_parser = PdfParser( )
					pdf_pages = st.session_state.get( 'pdf_pages' )
					if remove_pdf_repeats and isinstance( pdf_pages, list ) and pdf_pages:
						clean_pages = pdf_parser.remove_repeats( pdf_pages ) or [ ]
						processed_text = coerce_text( pdf_parser.rebuild_pages( pages=clean_pages,
							preserve_page_breaks=st.session_state.get( 'pdf_preserve_page_breaks',
								False ) ) )
					
					if clean_pdf_artifacts:
						processed_text = coerce_text( pdf_parser.clean_artifacts( processed_text ) )
					
					if repair_pdf_spacing:
						processed_text = coerce_text( pdf_parser.repair_spacing( processed_text ) )
					
					if rejoin_pdf_hyphenation:
						processed_text = coerce_text( pdf_parser.rejoin_hyphenation(
							processed_text,
							repair_embedded=repair_embedded_hyphenation ) )
				
				# ------------------------- Structural cleanup
				if remove_html:
					processed_text = coerce_text( tp.remove_html( processed_text ) )
				
				if remove_markdown:
					processed_text = coerce_text( tp.remove_markdown( processed_text ) )
				
				if remove_images:
					processed_text = coerce_text( tp.remove_images( processed_text ) )
				
				if remove_xml:
					processed_text = coerce_text( tp.remove_xml( processed_text ) )
				
				# ------------ Encoding and extraction cleanup
				if remove_encodings:
					processed_text = coerce_text( tp.remove_encodings( processed_text ) )
				
				if remove_errors:
					processed_text = coerce_text( tp.remove_errors( processed_text ) )
				
				if remove_fragments:
					processed_text = coerce_text( tp.remove_fragments( processed_text ) )
				
				# --------------- Noise and non-lexical cleanup
				if remove_symbols:
					processed_text = coerce_text( tp.remove_symbols( processed_text ) )
				
				if remove_numbers:
					processed_text = coerce_text( tp.remove_numbers( processed_text ) )
				
				if remove_numerals:
					processed_text = coerce_text( tp.remove_numerals( processed_text ) )
				
				if remove_punctuation:
					processed_text = coerce_text( tp.remove_punctuation( processed_text ) )
				
				if reduce_repeats:
					processed_text = coerce_text( tp.reduce_repeats( processed_text ) )
				
				# ---------------- Word normalization
				if normalize_text:
					processed_text = coerce_text( tp.normalize_text( processed_text ) )
				
				# ------------------ Lexical refinement
				if remove_stopwords:
					processed_text = coerce_text( tp.remove_stopwords( processed_text ) )
				
				# --------------- Final defensive artifact cleanup
				if remove_encodings:
					processed_text = coerce_text( tp.remove_encodings( processed_text ) )
				
				if remove_errors:
					processed_text = coerce_text( tp.remove_errors( processed_text ) )
				
				if collapse_whitespace:
					processed_text = coerce_text( tp.collapse_whitespace( processed_text ) )
				
				# -------------  Word-specific processing
				if active == 'WordLoader':
					word_parser = WordParser( )
					if extract_tables and hasattr( word_parser, 'extract_tables' ):
						processed_text = coerce_text( word_parser.extract_tables( processed_text ) )
					
					if extract_paragraphs and hasattr( word_parser, 'extract_paragraphs' ):
						processed_text = coerce_text(
							word_parser.extract_paragraphs( processed_text ) )
				
				# ---------------  HTML-specific processing
				if active == 'HtmlLoader' and strip_scripts:
					processed_text = coerce_text( tp.remove_html( processed_text ) )
				
				# -------------  Token processing
				display_text = processed_text
				st.session_state.nltk_word_tokens = [ ]
				st.session_state.nltk_sentence_tokens = [ ]
				st.session_state.nltk_stemmed_tokens = [ ]
				st.session_state.nltk_lemmatized_tokens = [ ]
				st.session_state.nltk_pos_tags = [ ]
				st.session_state.nltk_named_entities = [ ]
				if nltk_word_tokenize:
					st.session_state.nltk_word_tokens = (
							nlp.word_tokenizer( processed_text ) or [ ] )
					display_text = rebuild_token_text( st.session_state.nltk_word_tokens )
				
				if nltk_sentence_tokenize:
					st.session_state.nltk_sentence_tokens = (
							nlp.sentence_tokenizer( processed_text ) or [ ])
					
					display_text = '\n'.join(
						sentence for sentence in st.session_state.nltk_sentence_tokens if
						isinstance( sentence, str ) and sentence.strip( ) )
				
				if nltk_stem:
					st.session_state.nltk_stemmed_tokens = (
							nlp.word_stemmer( processed_text ) or [ ])
					processed_text = rebuild_token_text( st.session_state.nltk_stemmed_tokens )
					display_text = processed_text
				
				if nltk_lemmatize:
					st.session_state.nltk_lemmatized_tokens = (
							nlp.word_lemmatizer( processed_text ) or [ ])
					
					processed_text = rebuild_token_text( st.session_state.nltk_lemmatized_tokens )
					display_text = processed_text
				
				if nltk_pos_tag:
					st.session_state.nltk_pos_tags = (nlp.pos_tagger( processed_text ) or [ ])
					
					display_text = '\n'.join(
						f'{token}\t{tag}' for token, tag in st.session_state.nltk_pos_tags if
						isinstance( token, str ) and token.strip( ) )
				
				if nltk_named_entities:
					st.session_state.nltk_named_entities = (
							nlp.named_entity_recognition( processed_text ) or [ ])
					
					display_text = '\n'.join( f'{entity}\t{label}' for entity, label in
					st.session_state.nltk_named_entities if
					isinstance( entity, str ) and entity.strip( ) )
				
				#------------------  Final punctuation and delimiter cleanup
				if reduce_repeats:
					processed_text = coerce_text( tp.reduce_repeats( processed_text ) )
					display_text = coerce_text( tp.reduce_repeats( display_text ) )
				
				# ------------------ Finalize timing
				end_time = time.perf_counter( )
				st.session_state.start_time = start_time
				st.session_state.end_time = end_time
				st.session_state.total_time = end_time - start_time
				
				# --------------- Commit processed text
				st.session_state.processed_text = coerce_text( processed_text )
				st.session_state.displayed_text = coerce_text( display_text )
				if not st.session_state.displayed_text.strip( ):
					st.session_state.displayed_text = (st.session_state.processed_text)
				
				st.session_state.processed_text_display = (st.session_state.displayed_text)
				invalidate_embedding_state( clear_chunks=True )
				refresh_processing_display( )
				if st.session_state.processed_text.strip( ):
					st.success( f'Text processing applied '
					            f'({st.session_state.total_time:.1f} s)' )
				else:
					st.warning(
						'Processing completed, but the selected options produced empty text.' )
			
			# ---------------- Save Processed Text
			can_save_processed = (
					isinstance( st.session_state.get( 'processed_text' ), str ) and bool(
				st.session_state.get( 'processed_text' ).strip( ) ))
			
			if can_save_processed:
				processed_document_name = st.session_state.get( 'document_name' ) or 'processed_text'
				processed_file_name = f'{processed_document_name}.txt'

				save_processed_slot.download_button( 'Save',
					data=st.session_state.get( 'processed_text' ), file_name=processed_file_name,
					mime='text/plain', key='processed_text_save', icon='💾', width='stretch' )
			else:
				save_processed_slot.button( 'Save', key='processed_text_save_disabled',
					disabled=True, icon='💾', width='stretch' )
		
		# --------------- RIGHT COLUMN — Text Views
		with right:
			raw_text_view = st.session_state.get( 'raw_text_view' )
			raw_text_current = st.session_state.get( 'raw_text' )
			processed_current = st.session_state.get( 'processed_text' )
			displayed_current = st.session_state.get( 'displayed_text' )
			display_version = int( st.session_state.get( 'processing_display_version', 0 ) )
			st.text_area( label='Raw Text',
				value=raw_text_view if isinstance( raw_text_view, str ) else '', height=200,
				disabled=True, key=f'raw_text_view_display_{display_version}' )
			
			with st.expander( '📊 Processing Statistics:', expanded=False ):
				if (isinstance( raw_text_current,
						str ) and raw_text_current.strip( ) and isinstance( processed_current,
					str ) and processed_current.strip( )):
					raw_tokens = raw_text_current.split( )
					proc_tokens = processed_current.split( )
					raw_chars = len( raw_text_current )
					proc_chars = len( processed_current )
					raw_vocab = len( set( raw_tokens ) )
					proc_vocab = len( set( proc_tokens ) )
					st.text( 'Measures:' )
					ttr = (proc_vocab / len( proc_tokens ) if proc_tokens else 0.0)
					a1, a2, a3, a4 = st.columns( 4, border=True )
					a1.metric( 'Characters', f'{proc_chars:,}' )
					a2.metric( 'Tokens', f'{len( proc_tokens ):,}' )
					a3.metric( 'Unique Tokens', f'{proc_vocab:,}' )
					a4.metric( 'TTR', f'{ttr:.3f}' )
					
					st.divider( )
					
					st.text( 'Deltas:' )
					d1, d2, d3, d4 = st.columns( 4, border=True )
					char_delta = proc_chars - raw_chars
					token_delta = len( proc_tokens ) - len( raw_tokens )
					vocab_delta = proc_vocab - raw_vocab
					compression = (proc_chars / raw_chars if raw_chars > 0 else 0.0)
					d1.metric( 'Δ Characters', f'{char_delta:+,}' )
					d2.metric( 'Δ Tokens', f'{token_delta:+,}' )
					d3.metric( 'Δ Vocabulary', f'{vocab_delta:+,}' )
					d4.metric( 'Compression Ratio', f'{compression:.2%}' )
				else:
					st.caption( 'Load and process text to view absolute and delta statistics.' )
			
			st.text_area( 'Processed Text',
				value=displayed_current if isinstance( displayed_current, str ) else '',
				height=800, key=f'processed_text_display_area_{display_version}' )

# ======================================================================================
# Tab - Semantic Analysis
# ======================================================================================
with tabs[ 2 ]:
	import pandas as pd
	import tiktoken
	from nltk.tokenize import word_tokenize
	
	processed_text = st.session_state.get( 'processed_text' )
	
	# -------------- Guard
	if not isinstance( processed_text, str ) or not processed_text.strip( ):
		st.info( 'Run text processing before semantic analysis.' )
		st.stop( )
	
	# ---------- Chunking Modes
	chunk_modes = st.session_state.get( 'chunk_modes' )
	if not isinstance( chunk_modes, (list, tuple) ) or not chunk_modes:
		chunk_modes = [ 'tokens', 'chars' ]
		st.session_state.chunk_modes = list( chunk_modes )
	
	# ------------ Controls
	st.markdown( '#### Semantic Analysis' )
	col_1, col_2, col_3 = st.columns( 3, border=True )
	with col_1:
		mode = st.selectbox( 'Chunking Mode', options=list( chunk_modes ), key='chunk_mode',
			help='Select whether chunk size and overlap are measured in characters or model tokens.', )
	
	with col_2:
		chunk_size = st.number_input( 'Chunk Size', min_value=10, max_value=5000, value=800,
			step=10, key='chunk_count', )
	
	with col_3:
		overlap = st.number_input( 'Overlap', min_value=0, max_value=2000, value=100, step=50,
			key='overlap_input', )
	
	col_run, col_reset, col_none = st.columns( 3 )
	run_chunking = col_run.button( label='Chunk', key='run_button',
		icon='🏃', width='stretch' )
	
	reset_chunking = col_reset.button( label='Reset', key='reset_control',
		icon='🔁', width='stretch'  )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
	st.markdown( '#### Chunking Metrics' )
	
	# ------------ Reset
	if reset_chunking:
		st.session_state.chunked_documents = None
		st.session_state.df_chunk_records = None
		st.session_state.chunk_size = None
		st.session_state.chunk_overlap = None
		st.session_state.chunk_mode_value = None
		invalidate_embedding_state( clear_chunks=False )
		st.info( 'Chunking controls reset.' )
	
	# ----------- Chunk Generation
	if run_chunking:
		if int( overlap ) >= int( chunk_size ):
			st.error( 'Overlap must be smaller than chunk size.' )
		else:
			if mode == 'chars':
				chunked_documents = chunk_characters( text=processed_text, size=int( chunk_size ),
					overlap=int( overlap ) )
			elif mode == 'tokens':
				chunked_documents = chunk_tokens( text=processed_text, size=int( chunk_size ),
					overlap=int( overlap ), encoding_name='cl100k_base' )
			else:
				chunked_documents = [ ]
				st.error( f'Unsupported chunking mode: {mode}' )
			
			# --------- Canonical Chunk Records
			if chunked_documents:
				df_chunk_records = build_chunk_records( chunks=chunked_documents, mode=mode,
					configured_size=int( chunk_size ), configured_overlap=int( overlap ),
					source_text=processed_text )
				
				st.session_state.chunked_documents = chunked_documents
				st.session_state.df_chunk_records = df_chunk_records
				st.session_state.chunk_size = int( chunk_size )
				st.session_state.chunk_overlap = int( overlap )
				st.session_state.chunk_mode_value = mode
				invalidate_embedding_state( clear_chunks=False )
				st.success( f'Chunking complete: {len( chunked_documents ):,} chunks generated '
				            f'(mode={mode}, size={int( chunk_size )}, '
				            f'overlap={int( overlap )}).' )
			else:
				st.session_state.chunked_documents = None
				st.session_state.df_chunk_records = None
				st.session_state.chunk_size = None
				st.session_state.chunk_overlap = None
				st.session_state.chunk_mode_value = None
				invalidate_embedding_state( clear_chunks=False )
				st.warning( 'No chunks were generated.' )
	
	# ------------------------- Chunk Validation
	df_chunk_records = st.session_state.get( 'df_chunk_records' )
	if isinstance( df_chunk_records, pd.DataFrame ) and not df_chunk_records.empty:
		token_counts = df_chunk_records[ 'Token Count' ]
		metric_1, metric_2, metric_3, metric_4 = st.columns( 4, border=True, )
		metric_1.metric( 'Chunks', f'{len( df_chunk_records ):,}', )
		metric_2.metric( 'Minimum Tokens', f'{int( token_counts.min( ) ):,}', )
		metric_3.metric( 'Median Tokens', f'{int( token_counts.median( ) ):,}', )
		metric_4.metric( 'Maximum Tokens', f'{int( token_counts.max( ) ):,}', )
	
	# ------------- Tokenization and Vocabulary
	processor = TextParser( )
	tokens = word_tokenize( processed_text )
	vocabulary = processor.create_vocabulary( tokens )
	st.session_state.tokens = tokens
	st.session_state.vocabulary = vocabulary
	
	# --------------  Frequency Distribution
	df_frequency = processor.create_frequency_distribution( tokens )
	st.session_state.df_frequency = df_frequency
	if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
		st.session_state.df_token_frequency = df_frequency.rename(
			columns={ 'Word': 'Token' } ).copy( )
	else:
		st.session_state.df_token_frequency = None
	
	# ------------ Three-Column Layout
	col_tokens, col_vocab, col_freq = st.columns( [ 1, 1, 2 ], border=True,
		vertical_alignment='top', )
	
	with col_tokens:
		st.write( f'Tokens: {len( tokens ):,}' )
		
		st.data_editor( pd.DataFrame( tokens, columns=[ 'Token' ], ), num_rows='dynamic',
			use_container_width=True, height='stretch', disabled=True, )
	
	with col_vocab:
		st.write( f'Vocabulary: {len( vocabulary ):,}' )
		st.data_editor( pd.DataFrame( vocabulary, columns=[ 'Word' ], ), num_rows='dynamic',
			use_container_width=True, height='stretch', disabled=True, )
	
	with col_freq:
		st.markdown( '#### Frequency Distribution' )
		st.caption( 'Top 100 most frequent tokens' )
		if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
			numeric_cols = df_frequency.select_dtypes( include='number', )
			if not numeric_cols.empty:
				freq_col = numeric_cols.columns[ 0 ]
				label_cols = [ column for column in df_frequency.columns if column != freq_col ]
				if label_cols:
					label_col = label_cols[ 0 ]
					df_top = (df_frequency.sort_values( freq_col, ascending=False, ).head( 100 ))
					st.bar_chart( df_top.set_index( label_col )[ freq_col ],
						use_container_width=True, )
				else:
					st.info( 'No token label column available for charting.' )
			else:
				st.info( 'No numeric frequency column available for charting.' )
		else:
			st.info( 'Frequency distribution unavailable.' )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )

# ======================================================================================
# Tab - Data Tokenization
# ======================================================================================
with tabs[ 3 ]:
	st.subheader( 'Chunk Diagnostics' )
	line_col, chunk_col = st.columns( [ 0.5, 0.5 ], border=True, vertical_alignment='top' )
	df_frequency = st.session_state.get( 'df_frequency' )
	df_tables = st.session_state.get( 'df_tables' )
	df_count = st.session_state.get( 'df_count' )
	df_schema = st.session_state.get( 'df_schema' )
	df_preview = st.session_state.get( 'df_preview' )
	df_chunks = st.session_state.get( 'df_chunks' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embeddings = st.session_state.get( 'embeddings' )
	lines = st.session_state.get( 'lines' )
	chunks = st.session_state.get( 'chunks' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	active_table = st.session_state.get( 'active_table' )
	chunk_modes = st.session_state.get( 'chunk_modes' )
	raw_text = st.session_state.get( 'raw_text' )
	processed_text = st.session_state.get( 'processed_text' )
	tokens = st.session_state.get( 'tokens' )
	
	def pad_or_trim_row( row: list, size: int ) -> list:
		"""
			Purpose:
			--------
			Normalizes a token row to a fixed width by trimming or right-padding with blanks.
			
			Parameters:
			-----------
			row : list
				The token row to normalize.
			
			size : int
				The required fixed width.
			
			Returns:
			--------
			list
				A list of length == size.
		"""
		if not isinstance( row, list ):
			row = [ ]
		
		if len( row ) >= size:
			return row[ :size ]
		
		return row + ([ '' ] * (size - len( row )))
	
	def _safe_sent_tokenize( text: str ) -> list[ str ]:
		"""
			Purpose:
			--------
			Segments text into natural-language sentences for diagnostics using a
			boundary-preserving source.
			
			Parameters:
			-----------
			text : str
				The input text to segment.
			
			Returns:
			--------
			list[str]
				A list of non-empty sentence strings.
		"""
		if not isinstance( text, str ) or not text.strip( ):
			return [ ]
		
		try:
			segments = sent_tokenize( text )
		except LookupError:
			segments = re.split( r'(?<=[.!?;])\s+', text )
		
		return [ s.strip( ) for s in segments if isinstance( s, str ) and s.strip( ) ]
	
	def _safe_word_tokenize( text: str ) -> list[ str ]:
		"""
			Purpose:
			--------
			Tokenizes text into words for diagnostics.
			
			Parameters:
			-----------
			text : str
				The input text to tokenize.
			
			Returns:
			--------
			list[str]
				A list of non-empty tokens.
		"""
		if not isinstance( text, str ) or not text.strip( ):
			return [ ]
		
		try:
			parts = word_tokenize( text )
		except LookupError:
			parts = re.findall( r"\b\w+\b", text, flags=re.UNICODE )
		
		return [ p for p in parts if isinstance( p, str ) and p.strip( ) ]
	
	# ------------------------- Fixed vector-space schema
	dimensions = [ 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
		'D13', 'D14' ]
	
	# ------------------------- Canonical diagnostics state
	if isinstance( df_frequency, pd.DataFrame ) and not df_frequency.empty:
		if 'Word' in df_frequency.columns and 'Frequency' in df_frequency.columns:
			st.session_state.df_token_frequency = df_frequency.rename(
				columns={ 'Word': 'Token' } ).copy( )
		else:
			st.session_state.df_token_frequency = df_frequency.copy( )
	else:
		st.session_state.df_token_frequency = None
	
	# Sentence diagnostics must come from a boundary-preserving source.
	diagnostic_text = None
	if isinstance( raw_text, str ) and raw_text.strip( ):
		diagnostic_text = raw_text
	elif isinstance( processed_text, str ) and processed_text.strip( ):
		diagnostic_text = processed_text
	sentences = _safe_sent_tokenize( diagnostic_text ) if diagnostic_text else [ ]
	st.session_state.sentences = sentences if sentences else None
	if sentences:
		sentence_rows = [ _safe_word_tokenize( s ) for s in sentences ]
		sentence_rows = [ pad_or_trim_row( row, size=len( dimensions ) ) for row in sentence_rows ]
		if sentence_rows:
			st.session_state.df_sentence_tokens = pd.DataFrame( sentence_rows, columns=dimensions )
		else:
			st.session_state.df_sentence_tokens = None
	else:
		st.session_state.df_sentence_tokens = None
	
	# ------------------------- Canonical Sentence Diagnostic State
	lines = sentences if isinstance( sentences, list ) else [ ]
	st.session_state.lines = lines if lines else None
	
	# ------------------------- Canonical Chunk Data
	chunked_documents = st.session_state.get( 'chunked_documents' )
	df_chunk_records = st.session_state.get( 'df_chunk_records' )
	
	# Rebuild canonical records only when chunk text exists but its dataframe
	# is unavailable, such as after session-state migration or code reload.
	if (isinstance( chunked_documents, list ) and chunked_documents and (
			not isinstance( df_chunk_records, pd.DataFrame ) or df_chunk_records.empty)):
		df_chunk_records = build_chunk_records( chunks=chunked_documents,
			mode=st.session_state.get( 'chunk_mode_value', 'tokens', ),
			configured_size=int( st.session_state.get( 'chunk_size', 0, ) or 0 ),
			configured_overlap=int( st.session_state.get( 'chunk_overlap', 0, ) or 0 ),
			source_text=processed_text, )
		
		st.session_state.df_chunk_records = df_chunk_records
	
	# ------------------------- LEFT COLUMN — Chunked Data
	with line_col:
		st.text( 'Chunked Data' )
		if isinstance( df_chunk_records, pd.DataFrame ) and not df_chunk_records.empty:
			chunk_document_name = st.session_state.get( 'document_name' ) or 'chunked_data'
			chunk_file_name = f'{chunk_document_name}.csv'
			st.download_button( 'Save CSV', data=df_chunk_records.to_csv( index=False ),
				file_name=chunk_file_name, mime='text/csv', key='chunk_records_save', icon='💾',
				width='stretch' )

			chunk_columns = [ 'Chunk ID', 'Chunk Text', 'Token Count', 'Character Count', ]
			st.data_editor( df_chunk_records[ chunk_columns ], num_rows='fixed', width='stretch',
				height='stretch', disabled=True, key='chunk_records_editor', column_config={
					'Chunk ID': st.column_config.NumberColumn( 'Chunk ID', format='%d',
						width='small', ),
					'Chunk Text': st.column_config.TextColumn( 'Chunk Text', width='large', ),
					'Token Count': st.column_config.NumberColumn( 'Tokens', format='%d',
						width='small', ),
					'Character Count': st.column_config.NumberColumn( 'Characters', format='%d',
						width='small', ), }, )
		else:
			st.info( 'Run chunking in Semantic Analysis first.' )
	
	# ------------------------- RIGHT COLUMN — Chunk Summary
	with chunk_col:
		if isinstance( df_chunk_records, pd.DataFrame ) and not df_chunk_records.empty:
			st.text( f'Chunk Summary: {len( df_chunk_records ):,} chunks' )
			
			summary_columns = [ 'Chunk ID', 'Token Count', 'Character Count', 'Word Count',
				'Sentence Count', 'Chunk Preview', 'Configured Size', 'Configured Overlap', ]
			
			df_chunk_summary = df_chunk_records[ summary_columns ].copy( )
			
			st.data_editor( df_chunk_summary, num_rows='fixed', width='stretch', height='stretch',
				disabled=True, key='chunk_summary_editor', column_config={
					'Chunk ID': st.column_config.NumberColumn( 'Chunk ID', format='%d',
						width='small', ),
					'Token Count': st.column_config.NumberColumn( 'Tokens', format='%d',
						width='small', ),
					'Character Count': st.column_config.NumberColumn( 'Characters', format='%d',
						width='small', ),
					'Word Count': st.column_config.NumberColumn( 'Words', format='%d',
						width='small', ),
					'Sentence Count': st.column_config.NumberColumn( 'Sentences', format='%d',
						width='small', ),
					'Chunk Preview': st.column_config.TextColumn( 'Preview', width='large', ),
					'Configured Size': st.column_config.NumberColumn( 'Size', format='%d',
						width='small', ),
					'Configured Overlap': st.column_config.NumberColumn( 'Overlap', format='%d',
						width='small', ), }, )
		else:
			st.caption( 'Chunk summary not available yet.' )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( 'Token Diagnostics' )
	row1_col1, row1_col2 = st.columns( [ 0.5, 0.5 ], border=True )
	with row1_col1:
		st.caption( 'Top-N Token Frequency Distribution' )
		df_token_frequency = st.session_state.get( 'df_token_frequency' )
		if isinstance( df_token_frequency, pd.DataFrame ) and not df_token_frequency.empty:
			top_n = st.slider( 'Top-N Tokens', min_value=10, max_value=100, value=30, step=10,
				key='token_freq_top_n' )
			
			df_top = df_token_frequency.sort_values( by='Frequency', ascending=False ).head(
				top_n )
			st.bar_chart( df_top.set_index( 'Token' )[ 'Frequency' ], use_container_width=True )
		else:
			st.info( 'Token frequency data not available.' )
	
	with row1_col2:
		st.caption( 'Sentence Length Distribution (Tokens per Sentence)' )
		sentences = st.session_state.get( 'sentences' )
		if isinstance( sentences, list ) and sentences:
			sentence_lengths = [ len( _safe_word_tokenize( s ) ) for s in sentences ]
			sentence_lengths = [ n for n in sentence_lengths if isinstance( n, int ) and n > 0 ]
			if sentence_lengths:
				df_sentence_lengths = pd.Series( sentence_lengths ).value_counts( ).sort_index( )
				df_sentence_lengths = df_sentence_lengths.rename_axis(
					'Tokens per Sentence' ).to_frame( 'Sentence Count' )
				
				st.bar_chart( df_sentence_lengths, use_container_width=True )
			else:
				st.info( 'No valid sentence lengths computed.' )
		else:
			st.info( 'Sentence data not available.' )
	
	row2_col1, row2_col2 = st.columns( [ 0.5, 0.5 ], border=True )
	with row2_col1:
		st.caption( 'Token Grid Sparsity (Padding Analysis)' )
		df_sentence_tokens = st.session_state.get( 'df_sentence_tokens' )
		if isinstance( df_sentence_tokens, pd.DataFrame ) and not df_sentence_tokens.empty:
			total_cells = df_sentence_tokens.shape[ 0 ] * df_sentence_tokens.shape[ 1 ]
			empty_cells = (df_sentence_tokens == '').sum( ).sum( )
			filled_cells = total_cells - empty_cells
			padding_ratio = empty_cells / total_cells if total_cells > 0 else 0.0
			fill_ratio = filled_cells / total_cells if total_cells > 0 else 0.0
			m1, m2, m3 = st.columns( 3 )
			m1.metric( 'Total Cells', f'{total_cells:,}' )
			m2.metric( 'Filled Cells', f'{filled_cells:,}' )
			m3.metric( 'Padding %', f'{padding_ratio:.1%}' )
			st.progress( fill_ratio )
		else:
			st.info( 'Sentence token grid not available.' )
	
	with row2_col2:
		st.caption( 'Embedding Readiness Scorecard' )
		tokens = st.session_state.get( 'tokens' )
		sentences = st.session_state.get( 'sentences' )
		token_counts = (Counter( tokens ) if isinstance( tokens, list ) and tokens else None)
		if token_counts:
			total_tokens = len( tokens )
			unique_tokens = len( token_counts )
			hapax_count = sum( 1 for c in token_counts.values( ) if c == 1 )
			hapax_ratio = hapax_count / unique_tokens if unique_tokens > 0 else 0.0
			sentence_lengths = (
				[ len( _safe_word_tokenize( s ) ) for s in sentences ] if isinstance( sentences,
					list ) and sentences else [ ] )
			sentence_lengths = [ n for n in sentence_lengths if isinstance( n, int ) and n > 0 ]
			avg_sentence_len = (
				sum( sentence_lengths ) / len( sentence_lengths ) if sentence_lengths else 0.0)
			
			r1, r2 = st.columns( 2 )
			r1.metric( 'Total Tokens', f'{total_tokens:,}' )
			r2.metric( 'Unique Tokens', f'{unique_tokens:,}' )
			r3, r4 = st.columns( 2 )
			r3.metric( 'Avg Tokens / Sentence', f'{avg_sentence_len:.1f}' )
			r4.metric( 'Hapax Ratio', f'{hapax_ratio:.1%}' )
			st.caption( 'Lower padding and moderate hapax ratios yield more stable embeddings.' )
		else:
			st.info( 'Token readiness metrics unavailable.' )

# Tab - Embeddings
with tabs[ 4 ]:
	import hashlib
	import tiktoken
	
	# ------------------------- Embedding Utilities
	def embedding_key( name: str ) -> str:
		"""Create an embedding widget key.
		
		Purpose:
			Returns a namespaced Streamlit widget key that prevents collisions between
			embedding controls and controls rendered elsewhere in the application.
		
		Args:
			name: Descriptive widget-key suffix.
		
		Returns:
			str: Namespaced Streamlit widget key.
		"""
		return f'emb__{name}'
	
	def project_chunks_for_embedding( values: list ) -> List[ str ]:
		"""Project chunk values into embedding-ready text.
		
		Purpose:
			Converts supported chunk strings, token lists, LangChain documents, and record
			dictionaries into ordered, nonempty strings without introducing blank embedding
			inputs or altering valid source content.
		
		Args:
			values: Chunk values produced by the document-chunking workflow.
		
		Returns:
			List[ str ]: Ordered, nonempty strings suitable for embedding generation.
		"""
		texts: List[ str ] = [ ]
		
		if not isinstance( values, list ):
			return texts
		
		for value in values:
			text = ''
			
			if isinstance( value, str ):
				text = value.strip( )
			elif isinstance( value, Document ):
				page_content = getattr( value, 'page_content', '' )
				
				if isinstance( page_content, str ):
					text = page_content.strip( )
			elif isinstance( value, list ):
				tokens = [ token.strip( ) for token in value if
					isinstance( token, str ) and token.strip( ) ]
				
				if tokens:
					text = ' '.join( tokens )
			elif isinstance( value, dict ):
				for field in ('Chunk Text', 'chunk_text', 'text', 'page_content', 'content'):
					field_value = value.get( field )
					
					if isinstance( field_value, str ) and field_value.strip( ):
						text = field_value.strip( )
						break
			
			if text:
				texts.append( text )
		
		return texts
	
	def resolve_embedding_texts( source: str, processed_value: object,
		chunk_values: object ) -> List[ str ]:
		"""Resolve the selected embedding source.
		
		Purpose:
			Returns validated embedding input strings from the selected processed-text or
			chunked-document source while preserving source order and excluding unusable values.
		
		Args:
			source: Embedding source selected by the user.
			processed_value: Current processed-text session value.
			chunk_values: Current chunked-document session value.
		
		Returns:
			List[ str ]: Validated embedding input strings.
		"""
		if source == 'Processed Text':
			if isinstance( processed_value, str ) and processed_value.strip( ):
				return [ processed_value.strip( ) ]
			
			return [ ]
		
		if source == 'Chunked Documents':
			if isinstance( chunk_values, list ) and chunk_values:
				return project_chunks_for_embedding( chunk_values )
			
			return [ ]
		
		return [ ]
	
	def create_embedding_signature( source: str, texts: List[ str ] ) -> str:
		"""Create an embedding-source signature.
		
		Purpose:
			Generates a deterministic digest from the selected source and ordered embedding
			texts so previously generated vectors cannot be silently associated with changed
			source content.
		
		Args:
			source: Embedding source selected by the user.
			texts: Ordered embedding input strings.
		
		Returns:
			str: SHA-256 digest representing the embedding source content.
		"""
		digest = hashlib.sha256( )
		digest.update( source.encode( 'utf-8' ) )
		
		for text in texts:
			encoded_text = text.encode( 'utf-8' )
			digest.update( len( encoded_text ).to_bytes( 8, byteorder='big' ) )
			digest.update( encoded_text )
		
		return digest.hexdigest( )
	
	def count_embedding_tokens( texts: List[ str ],
		encoding_name: str = 'cl100k_base' ) -> List[ int ]:
		"""Count tokens in embedding inputs.
		
		Purpose:
			Calculates model-token counts for each embedding input so oversized requests are
			blocked before a provider call without truncating or modifying source content.
		
		Args:
			texts: Ordered embedding input strings.
			encoding_name: TikToken encoding used for local token measurement.
		
		Returns:
			List[ int ]: Token counts aligned with the supplied texts.
		"""
		if not texts:
			return [ ]
		
		encoding = tiktoken.get_encoding( encoding_name )
		
		return [ len( encoding.encode( text, disallowed_special=( ) ) ) for text in texts ]
	
	def normalize_embedding_vectors( values: object,
		expected_count: int ) -> tuple[ List[ List[ float ] ], int ]:
		"""Normalize and validate embedding vectors.
		
		Purpose:
			Converts supported provider output into a finite, rectangular numeric matrix and
			verifies that each input text has exactly one embedding with a consistent positive
			dimension.
		
		Args:
			values: Raw embedding output returned by a provider wrapper.
			expected_count: Number of source texts submitted to the provider.
		
		Returns:
			tuple[List[List[float]], int]: Validated vectors and their common dimension.
		
		Raises:
			ValueError: Raised when provider output is empty, misaligned, nonnumeric, ragged,
				or contains non-finite values.
		"""
		if isinstance( values, pd.DataFrame ):
			raw_vectors = values.values.tolist( )
		elif isinstance( values, pd.Series ):
			raw_vectors = values.tolist( )
		elif isinstance( values, np.ndarray ):
			raw_vectors = values.tolist( )
		elif isinstance( values, (list, tuple) ):
			raw_vectors = list( values )
		else:
			raise ValueError( 'The embedding provider returned an unsupported result type.' )
		
		if not raw_vectors:
			raise ValueError( 'The embedding provider returned no vectors.' )
		
		if len( raw_vectors ) != expected_count:
			raise ValueError( f'The embedding provider returned {len( raw_vectors )} vector(s) '
			                  f'for '
			                  f'{expected_count} input text(s).' )
		
		normalized_vectors: List[ List[ float ] ] = [ ]
		expected_dimension: int | None = None
		
		for index, raw_vector in enumerate( raw_vectors, start=1 ):
			if isinstance( raw_vector, np.ndarray ):
				vector_values = raw_vector.tolist( )
			elif isinstance( raw_vector, pd.Series ):
				vector_values = raw_vector.tolist( )
			elif isinstance( raw_vector, (list, tuple) ):
				vector_values = list( raw_vector )
			else:
				raise ValueError( f'Embedding vector {index} is not a supported numeric '
				                  f'sequence.' )
			
			if not vector_values:
				raise ValueError( f'Embedding vector {index} is empty.' )
			
			try:
				vector = [ float( value ) for value in vector_values ]
			except (TypeError, ValueError) as exception:
				raise ValueError(
					f'Embedding vector {index} contains a nonnumeric value.' ) from exception
			
			if not np.isfinite( np.asarray( vector, dtype=float ) ).all( ):
				raise ValueError( f'Embedding vector {index} contains NaN or infinite values.' )
			
			if expected_dimension is None:
				expected_dimension = len( vector )
				
				if expected_dimension < 1:
					raise ValueError( 'Embedding vectors must contain at least one dimension.' )
			elif len( vector ) != expected_dimension:
				raise ValueError( f'Embedding vector {index} has {len( vector )} dimensions; '
				                  f'{expected_dimension} were expected.' )
			
			normalized_vectors.append( vector )
		
		return normalized_vectors, int( expected_dimension or 0 )
	
	def clear_embedding_results( ) -> None:
		"""Clear generated embedding results.
		
		Purpose:
			Resets all generated-vector, provider, model, source-provenance, and output-table
			state while preserving the user's current provider-control selections.
		"""
		st.session_state.embeddings = None
		st.session_state.embedding_documents = None
		st.session_state.embedding_texts = [ ]
		st.session_state.df_embedding_output = pd.DataFrame( )
		st.session_state.embedding_provider = None
		st.session_state.embedding_model = None
		st.session_state.embedding_task = None
		st.session_state.embedding_dimensions = None
		st.session_state.embedding_vector_dimension = None
		st.session_state.embedding_source_signature = None
		st.session_state.embedding_is_stale = False
	
	def save_embedding_results( provider: str, model: str, source: str, texts: List[ str ],
		vectors: List[ List[ float ] ], source_signature: str, task: str | None = None,
		requested_dimensions: int | None = None, vector_dimension: int | None = None ) -> None:
		"""Persist validated embedding results.
		
		Purpose:
			Commits validated vectors and immutable source-text provenance to Streamlit session
			state only after provider execution and output validation complete successfully.
		
		Args:
			provider: Provider that generated the embeddings.
			model: Provider model used for generation.
			source: Selected embedding source.
			texts: Ordered source strings submitted to the provider.
			vectors: Validated embedding vectors aligned with the source strings.
			source_signature: Digest representing the source content.
			task: Optional provider task type.
			requested_dimensions: Optional provider dimension request.
			vector_dimension: Verified dimension of each returned vector.
		"""
		records: List[ dict ] = [ ]
		
		for index, (text, vector) in enumerate( zip( texts, vectors ) ):
			record = { 'provider': provider, 'model': model, 'row_index': index, 'text': text,
				'embedding': vector }
			
			if task:
				record[ 'task' ] = task
			
			if requested_dimensions is not None:
				record[ 'requested_dimensions' ] = int( requested_dimensions )
			
			if vector_dimension is not None:
				record[ 'vector_dimension' ] = int( vector_dimension )
			
			records.append( record )
		
		output = pd.DataFrame( records )
		
		st.session_state.df_embedding_output = output
		st.session_state.embedding_documents = output.to_dict( 'records' )
		st.session_state.embeddings = vectors
		st.session_state.embedding_texts = list( texts )
		st.session_state.embedding_provider = provider
		st.session_state.embedding_model = model
		st.session_state.embedding_source = source
		st.session_state.embedding_task = task
		st.session_state.embedding_dimensions = requested_dimensions
		st.session_state.embedding_vector_dimension = vector_dimension
		st.session_state.embedding_source_signature = source_signature
		st.session_state.embedding_is_stale = False
	
	# ------------------------- Embedding State
	if not isinstance( st.session_state.get( 'df_embedding_output' ), pd.DataFrame ):
		st.session_state.df_embedding_output = pd.DataFrame( )
	
	if 'embedding_documents' not in st.session_state:
		st.session_state.embedding_documents = None
	
	if 'embedding_texts' not in st.session_state:
		st.session_state.embedding_texts = [ ]
	
	if 'embedding_source_signature' not in st.session_state:
		st.session_state.embedding_source_signature = None
	
	if 'embedding_is_stale' not in st.session_state:
		st.session_state.embedding_is_stale = False
	
	processed_text = st.session_state.get( 'processed_text' )
	chunked_documents = st.session_state.get( 'chunked_documents' )
	maximum_input_tokens = 8000
	
	# ------------------------- Layout
	left, right = st.columns( [ 1, 1.5 ], border=True )
	with left:
		st.markdown( '##### Embedding Providers' )
		embedding_source = st.radio( 'Text Source',
			options=[ 'Processed Text', 'Chunked Documents' ], horizontal=True,
			key=embedding_key( 'text_source' ) )
		
		texts = resolve_embedding_texts( embedding_source, processed_text, chunked_documents )
		has_texts = bool( texts )
		source_signature = create_embedding_signature( embedding_source,
			texts ) if has_texts else ''
		
		token_counts = count_embedding_tokens( texts ) if has_texts else [ ]
		oversized_inputs = [ (index + 1, token_count) for index, token_count in
			enumerate( token_counts ) if token_count > maximum_input_tokens ]
		
		inputs_within_limit = not oversized_inputs
		current_provider = st.session_state.get( 'embedding_provider' )
		stored_signature = st.session_state.get( 'embedding_source_signature' )
		has_existing_embeddings = isinstance( st.session_state.get( 'embeddings' ),
			(list, np.ndarray) )
		
		embedding_is_stale = bool( has_existing_embeddings and isinstance( stored_signature,
			str ) and stored_signature and source_signature and stored_signature !=
		                           source_signature )
		
		st.session_state.embedding_is_stale = embedding_is_stale
		
		st.caption( f'Texts to embed: {len( texts ):,}' )
		
		if token_counts:
			st.caption( f'Total input tokens: {sum( token_counts ):,} | '
			            f'Maximum input: {max( token_counts ):,}' )
		
		if not has_texts:
			st.info( 'No text available. Run processing or chunking first.' )
		
		if oversized_inputs:
			first_index, first_count = oversized_inputs[ 0 ]
			st.error( f'Input {first_index:,} contains {first_count:,} tokens and exceeds the '
			          f'{maximum_input_tokens:,}-token safety limit. Use Chunked Documents or '
			          'reduce the chunk size before generating embeddings.' )
		
		if embedding_is_stale:
			st.warning( 'Existing embeddings were generated from different source content. '
			            'Regenerate them before using diagnostics, saving output, or persisting '
			            'vectors.' )
		
		def can_save_provider_output( provider: str ) -> bool:
			return bool( isinstance( st.session_state.get( 'df_embedding_output' ),
				pd.DataFrame ) and not st.session_state.df_embedding_output.empty and
			             st.session_state.get(
				'embedding_provider' ) == provider and not st.session_state.get(
				'embedding_is_stale', False ) )
		
		# ==============================================================================
		# OpenAI Embeddings
		# ==============================================================================
		with st.expander( label='OpenAI Embeddings', icon='🧠', expanded=False ):
			openai_model = st.selectbox( 'Model', options=cfg.GPT_MODELS,
				key=embedding_key( 'openai_model' ) )
			
			openai_api_key = st.session_state.get( 'openai_api_key', '' )
			openai_key_available = bool(
				isinstance( openai_api_key, str ) and openai_api_key.strip( ) )
			
			if not openai_key_available:
				st.warning( 'An OpenAI API key is required.' )
			
			openai_can_embed = bool(
				has_texts and inputs_within_limit and openai_model and openai_key_available )
			
			col_run, col_clear, col_save = st.columns( 3 )
			
			openai_run = col_run.button( 'Embed', key=embedding_key( 'openai_embed' ),
				use_container_width=True, disabled=not openai_can_embed )
			
			openai_clear = col_clear.button( 'Clear', key=embedding_key( 'openai_clear' ),
				use_container_width=True, disabled=st.session_state.df_embedding_output.empty )
			
			if can_save_provider_output( 'OpenAI' ):
				openai_filename = (
					f'openai_{str( st.session_state.embedding_model ).replace( "/", "_" )}'
					f'_embeddings.csv')
				
				col_save.download_button( 'Save CSV',
					data=st.session_state.df_embedding_output.to_csv( index=False ),
					file_name=openai_filename, mime='text/csv', use_container_width=True,
					key=embedding_key( 'openai_save' ), icon='💾' )
			else:
				col_save.button( 'Save CSV', disabled=True, use_container_width=True,
					key=embedding_key( 'openai_save_disabled' ), icon='💾' )
			
			if openai_clear:
				clear_embedding_results( )
				st.success( 'Embeddings cleared.' )
			
			if openai_run and openai_can_embed:
				try:
					with st.spinner( 'Embedding with OpenAI...' ):
						embedder = GPT( )
						raw_vectors = embedder.embed( texts, model=openai_model )
						vectors, vector_dimension = normalize_embedding_vectors( raw_vectors,
							len( texts ) )
						
						save_embedding_results( provider='OpenAI', model=openai_model,
							source=embedding_source, texts=texts, vectors=vectors,
							source_signature=source_signature, vector_dimension=vector_dimension )
					
					st.success( f'Generated {len( vectors ):,} embedding(s) with '
					            f'{vector_dimension:,} dimensions.' )
				except Exception as exception:
					st.error( f'OpenAI embedding generation failed: {exception}' )
		
		# ==============================================================================
		# Gemini Embeddings
		# ==============================================================================
		with st.expander( label='Gemini Embeddings', icon='✨', expanded=False ):
			gemini_model = st.selectbox( 'Model', options=cfg.GEMINI_MODELS,
				key=embedding_key( 'gemini_model' ), disabled=not has_texts )
			
			gemini_initialization_error: str | None = None
			gemini_task_options = getattr( Gemini, 'task_options', None )
			if not isinstance( gemini_task_options, (list, tuple) ):
				try:
					gemini_task_options = Gemini( ).task_options
				except Exception as exception:
					gemini_task_options = [ ]
					gemini_initialization_error = str( exception )
			
			gemini_task_options = [ option for option in gemini_task_options if
				isinstance( option, str ) and option.strip( ) ]
			
			gemini_task = st.selectbox( 'Task Type', options=gemini_task_options,
				key=embedding_key( 'gemini_task' ),
				disabled=not bool( has_texts and gemini_task_options ),
				help='Required to determine embedding intent.' )
			
			gemini_dimensions = st.number_input( 'Dimensions', min_value=128, max_value=2048,
				step=128, value=768, key=embedding_key( 'gemini_dimensions' ),
				disabled=not has_texts, help='Optional. Must be supported by the selected model.' )
			
			gemini_api_key = st.session_state.get( 'gemini_api_key', '' )
			
			if not isinstance( gemini_api_key, str ) or not gemini_api_key.strip( ):
				gemini_api_key = st.session_state.get( 'google_api_key', '' )
			
			gemini_key_available = bool(
				isinstance( gemini_api_key, str ) and gemini_api_key.strip( ) )
			
			if gemini_initialization_error:
				st.error( f'Gemini embedding controls could not be initialized: '
				          f'{gemini_initialization_error}' )
			elif not gemini_task_options:
				st.warning( 'No Gemini embedding task types are available.' )
			
			if not gemini_key_available:
				st.warning( 'A Gemini or Google API key is required.' )
			
			gemini_can_embed = bool(
				has_texts and inputs_within_limit and gemini_model and gemini_task and
				gemini_key_available and not gemini_initialization_error )
			
			col_run, col_clear, col_save = st.columns( 3 )
			gemini_run = col_run.button( 'Embed', key=embedding_key( 'gemini_embed' ),
				use_container_width=True, disabled=not gemini_can_embed )
			
			gemini_clear = col_clear.button( 'Clear', key=embedding_key( 'gemini_clear' ),
				use_container_width=True, disabled=st.session_state.df_embedding_output.empty )
			
			if can_save_provider_output( 'Gemini' ):
				gemini_filename = (
					f'gemini_{str( st.session_state.embedding_model ).replace( "/", "_" )}'
					f'_embeddings.csv')
				
				col_save.download_button( 'Save CSV',
					data=st.session_state.df_embedding_output.to_csv( index=False ),
					file_name=gemini_filename, mime='text/csv', use_container_width=True,
					key=embedding_key( 'gemini_save' ), icon='💾' )
			else:
				col_save.button( 'Save CSV', disabled=True, use_container_width=True,
					key=embedding_key( 'gemini_save_disabled' ), icon='💾' )
			
			if gemini_clear:
				clear_embedding_results( )
				st.success( 'Embeddings cleared.' )
			
			if gemini_run and gemini_can_embed:
				try:
					with st.spinner( 'Embedding with Gemini...' ):
						embedder = Gemini( )
						raw_vectors = embedder.embed( texts, task=gemini_task, model=gemini_model,
							dimensions=int( gemini_dimensions ) )
						
						vectors, vector_dimension = normalize_embedding_vectors( raw_vectors,
							len( texts ) )
						
						save_embedding_results( provider='Gemini', model=gemini_model,
							source=embedding_source, texts=texts, vectors=vectors,
							source_signature=source_signature, task=gemini_task,
							requested_dimensions=int( gemini_dimensions ),
							vector_dimension=vector_dimension )
					
					st.success( f'Generated {len( vectors ):,} embedding(s) with '
					            f'{vector_dimension:,} dimensions.' )
				except Exception as exception:
					st.error( f'Gemini embedding generation failed: {exception}' )
	
	# ------------------------- Embedding Input
	with right:
		st.markdown( '##### Embedding Data' )
		
		if texts:
			df_embedding_input = pd.DataFrame( { 'Row': range( 1, len( texts ) + 1 ), 'Token Count': token_counts,
					'Text': texts } )
		else:
			df_embedding_input = pd.DataFrame( columns=[ 'Row', 'Token Count', 'Text' ] )
		
		st.data_editor( df_embedding_input, use_container_width=True, hide_index=True,
			disabled=True, key=embedding_key( 'embedding_input_view' ) )
	
	# ------------------------- Embedding Results
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( 'Tensor Data' )
	
	if st.session_state.df_embedding_output.empty:
		st.info( 'No embeddings generated yet.' )
	elif st.session_state.get( 'embedding_is_stale', False ):
		st.warning(
			'The displayed embeddings are stale because the selected source content changed. '
			'Regenerate the embeddings before using this output.' )
		
		st.data_editor( st.session_state.df_embedding_output, use_container_width=True,
			hide_index=True, disabled=True, key=embedding_key( 'stale_embedding_output_view' ) )
	else:
		st.data_editor( st.session_state.df_embedding_output, use_container_width=True,
			hide_index=True, disabled=True, key=embedding_key( 'embedding_output_view' ) )
	
	# -------- Tensor Embedding — Dimensionality Reduction Diagnostics (t-SNE / UMAP)
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	st.subheader( 'Embedding Diagnostics (t-SNE / UMAP)' )
	embeddings = st.session_state.get( 'embeddings' )
	embedding_texts = st.session_state.get( 'embedding_texts' )
	embedding_is_stale = bool( st.session_state.get( 'embedding_is_stale', False ) )
	emb_array: np.ndarray | None = None
	diagnostic_error: str | None = None
	
	# ------------------------- Embedding validation
	if embedding_is_stale:
		diagnostic_error = (
			'The current embeddings are stale because the selected source content changed. '
			'Regenerate the embeddings before running diagnostics.')
	elif not isinstance( embeddings, (list, tuple, np.ndarray, pd.DataFrame) ):
		diagnostic_error = ('Generate embeddings to enable dimensionality reduction diagnostics.')
	elif not isinstance( embedding_texts, list ) or not embedding_texts:
		diagnostic_error = (
			'Embedding source-text provenance is unavailable. Regenerate the embeddings '
			'before running diagnostics.')
	else:
		try:
			if isinstance( embeddings, pd.DataFrame ):
				emb_array = embeddings.to_numpy( dtype=float )
			else:
				emb_array = np.asarray( embeddings, dtype=float )
			
			if emb_array.ndim != 2:
				raise ValueError( 'Embedding data must be a two-dimensional numeric matrix.' )
			
			if emb_array.shape[ 0 ] < 3:
				raise ValueError(
					'At least three embedding vectors are required for dimensionality reduction.' )
			
			if emb_array.shape[ 1 ] < 1:
				raise ValueError( 'Embedding vectors must contain at least one dimension.' )
			
			if emb_array.shape[ 0 ] != len( embedding_texts ):
				raise ValueError(
					f'Embedding row count ({emb_array.shape[ 0 ]:,}) does not match the '
					f'source-text count ({len( embedding_texts ):,}). Regenerate the embeddings '
					'to restore source alignment.' )
			
			if not np.isfinite( emb_array ).all( ):
				raise ValueError( 'Embedding data contains NaN or infinite values.' )
			
			for index, text in enumerate( embedding_texts, start=1 ):
				if not isinstance( text, str ) or not text.strip( ):
					raise ValueError( f'Embedding source text {index:,} is empty or invalid.' )
		except (TypeError, ValueError) as exception:
			emb_array = None
			diagnostic_error = str( exception )
	
	# ------------- Guard: validated embeddings availability
	if diagnostic_error:
		if embeddings is None:
			st.info( diagnostic_error )
		else:
			st.warning( diagnostic_error )
	elif emb_array is not None:
		sample_count = int( emb_array.shape[ 0 ] )
		ctrl_col1, ctrl_col2, ctrl_col3 = st.columns( 3, border=True )
		with ctrl_col1:
			reduction_method = st.selectbox( 'Reduction Method', options=[ 't-SNE', 'UMAP' ],
				key='embedding_reduction_method' )
		
		with ctrl_col2:
			if reduction_method == 't-SNE':
				maximum_perplexity = min( 50, sample_count - 1 )
				minimum_perplexity = 2
				default_perplexity = min( 30, maximum_perplexity )
				perplexity = st.slider( 't-SNE Perplexity', min_value=minimum_perplexity,
					max_value=maximum_perplexity, value=default_perplexity, step=1,
					key='tsne_perplexity' )
			else:
				maximum_neighbors = min( 50, sample_count - 1 )
				minimum_neighbors = 2
				default_neighbors = min( 15, maximum_neighbors )
				
				n_neighbors = st.slider( 'UMAP Neighbors', min_value=minimum_neighbors,
					max_value=maximum_neighbors, value=default_neighbors, step=1,
					key='umap_neighbors' )
		
		with ctrl_col3:
			random_state = st.number_input( 'Random Seed', min_value=0, value=42, step=1,
				key='embedding_reduction_seed' )
		
		# ----------- Dimensionality reduction
		reduced: np.ndarray | None = None
		reduction_error: str | None = None
		try:
			if reduction_method == 't-SNE':
				from sklearn.manifold import TSNE
				
				reducer = TSNE( n_components=2, perplexity=float( perplexity ),
					random_state=int( random_state ), init='pca', learning_rate='auto' )
				
				reduced = reducer.fit_transform( emb_array )
			else:
				import umap
				
				reducer = umap.UMAP( n_components=2, n_neighbors=int( n_neighbors ),
					random_state=int( random_state ), min_dist=0.1 )
				
				reduced = reducer.fit_transform( emb_array )
			
			reduced = np.asarray( reduced, dtype=float )
			if reduced.ndim != 2 or reduced.shape != (sample_count, 2):
				raise ValueError(
					'The dimensionality reducer returned an invalid coordinate matrix.' )
			
			if not np.isfinite( reduced ).all( ):
				raise ValueError(
					'The dimensionality reducer returned NaN or infinite coordinates.' )
		except Exception as exception:
			reduced = None
			reduction_error = str( exception )
		
		# ------------------------- Visualization
		if reduction_error:
			st.error( f'Dimensionality reduction failed: {reduction_error}' )
		elif reduced is not None:
			previews: List[ str ] = [ ]
			
			for text in embedding_texts:
				preview = re.sub( r'\s+', ' ', text ).strip( )
				
				if len( preview ) > 120:
					preview = f'{preview[ :120 ].rstrip( )}…'
				
				previews.append( preview )
			
			df_reduced = pd.DataFrame(
				{ 'X': reduced[ :, 0 ], 'Y': reduced[ :, 1 ], 'Chunk Index': range( sample_count ),
					'Preview': previews } )
			
			chart_container = st.container( border=True )
			with chart_container:
				st.caption(
					'Each point represents one embedded chunk; proximity indicates similarity '
					'in the selected two-dimensional projection.' )
				
				st.scatter_chart( df_reduced, x='X', y='Y', size=60, use_container_width=True )
			
			st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
			st.subheader( 'Dimension-Reduced Data' )
			with st.expander( 'View Reduced Coordinates (Table)', expanded=True ):
				st.data_editor( df_reduced, use_container_width=True, hide_index=True,
					disabled=True, key='embedding_reduced_coordinates' )

# ======================================================================================
# Tab - Vector Store
# ======================================================================================
with tabs[ 5 ]:
	st.subheader( 'Vector Database' )

	# ----------- Required upstream state
	embeddings = st.session_state.get( 'embeddings' )
	embedding_texts = st.session_state.get( 'embedding_texts' )
	embedding_model = st.session_state.get( 'embedding_model' )
	embedding_provider = st.session_state.get( 'embedding_provider' )
	collection_name = st.session_state.get( 'collection_name' ) or 'default_document'
	document_name = st.session_state.get( 'document_name' ) or 'default_document'

	# ------------ Guard: embeddings must exist before continuing
	if not (isinstance( embeddings, list ) and isinstance( embedding_texts,
			list ) and embedding_texts and embedding_model and embedding_provider):
		st.info( 'Generate embeddings before persisting to the vector database.' )
		st.stop( )

	# ------------------------- Derive vector metadata
	emb_array = np.asarray( embeddings, dtype=float )

	if emb_array.ndim == 1:
		emb_array = emb_array.reshape( 1, -1 )

	if emb_array.ndim != 2 or emb_array.shape[ 0 ] < 1:
		st.error( 'Invalid embeddings array.' )
		st.stop( )

	dim = int( emb_array.shape[ 1 ] )

	# ------------------------- Vector Store Selection
	selector_col, cloud_col = st.columns( [ 0.65, 0.35 ], border=True )
	with selector_col:
		vector_store_provider = st.radio( 'Storage Mode', options=[ 'SQLiteVec', 'Cloud' ],
			horizontal=True, key='vector_store_provider',
			help='SQLiteVec preserves the existing local workflow. Cloud enables Chroma or Pinecone.' )

	with cloud_col:
		use_pinecone = st.toggle( 'Use Pinecone', key='vector_store_cloud_pinecone',
			disabled=vector_store_provider != 'Cloud',
			help='Off stores vectors in Chroma. On stores vectors in Pinecone.' )
		cloud_provider = 'Pinecone' if use_pinecone else 'Chroma'
		st.caption( f'Cloud Provider: {cloud_provider}' )

	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

	# ------------------------- SQLiteVec
	if vector_store_provider == 'SQLiteVec':
		st.markdown( '#### SQLiteVec' )
		local_document_name = st.text_input( 'Document / Collection Name', value=document_name,
			key='sqlite_document_name' )
		table_name = (f'{local_document_name}__'
		              f'{embedding_provider}__'
		              f'{embedding_model}__'
		              f'{dim}')

		st.caption( f'Vector Table: `{table_name}`' )

		# ------------------------- Database connection
		db_path = st.text_input( 'SQLite Database Path', value='vectors.db',
			key='sqlite_vector_db_path' )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

		col_create, col_insert, col_delete = st.columns( 3 )
		with col_create:
			if st.button( label='Create Vector Table', icon='➕', key='sqlite_create_vector_table' ):
				conn = sqlite3.connect( db_path )
				conn.enable_load_extension( True )
				sqlite_vec.load( conn )
				SQLiteVec.create_table( conn, table_name=table_name, dimension=dim )
				conn.close( )
				st.success( f'Created vector table `{table_name}`.' )

		# ------------------------- Insert Embeddings
		with col_insert:
			if st.button( 'Insert Embeddings', icon='🔣', key='sqlite_insert_embeddings' ):
				conn = sqlite3.connect( db_path )
				conn.enable_load_extension( True )
				sqlite_vec.load( conn )
				vector_store = SQLiteVec( connection=conn, table_name=table_name,
					embedding=SentenceTransformerEmbeddings( model_name=embedding_model ) )

				vector_store.add_texts( texts=embedding_texts, embeddings=embeddings )
				conn.close( )
				st.success( f'Inserted {len( embeddings )} embeddings into `{table_name}`.' )

		# ------------------------- Drop Embeddings
		with col_delete:
			if st.button( label='Drop Vector Table', type='secondary', icon='❌',
					key='sqlite_drop_vector_table' ):
				conn = sqlite3.connect( db_path )
				cur = conn.cursor( )
				cur.execute( f'DROP TABLE IF EXISTS {table_name}' )
				conn.commit( )
				conn.close( )
				st.warning( f'Dropped vector table `{table_name}`.' )

		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

		# ------------------------- Verify Embeddings
		if st.checkbox( 'Inspect Vector Table', key='sqlite_inspect_vector_table' ):
			try:
				conn = sqlite3.connect( db_path )
				df_preview = pd.read_sql_query( f'SELECT * FROM {table_name} LIMIT 5', conn )
				conn.close( )
				st.data_editor( df_preview, use_container_width=True, num_rows='dynamic' )
			except Exception as exception:
				st.error( f'Vector-table inspection failed: {exception}' )

		# ------------------------- Similarity Search (sqlite-vec)
		st.subheader( 'Similarity Search' )

		query_text = st.text_area( 'Query Text',
			placeholder='Enter text to search for semantically similar chunks…', height=100,
			key='sqlite_query_text' )

		top_k = st.slider( 'Top-K Results', min_value=1, max_value=20, value=5, step=1,
			key='sqlite_top_k' )
		similarity_threshold = st.slider( 'Minimum Similarity Threshold', min_value=0.0,
			max_value=1.0, value=0.0, step=0.01, key='sqlite_similarity_threshold',
			help='Only results with similarity ≥ threshold will be shown.' )

		if not query_text.strip( ):
			st.info( 'Enter a query to run similarity search.' )
			results = None
		else:
			try:
				conn = sqlite3.connect( db_path )
				conn.enable_load_extension( True )
				sqlite_vec.load( conn )
				embedding_fn = SentenceTransformerEmbeddings( model_name=embedding_model )
				vector_store = SQLiteVec( connection=conn, table_name=table_name,
					embedding=embedding_fn )

				results = vector_store.similarity_search_with_score( query=query_text, k=top_k )
				conn.close( )
			except Exception as ex:
				st.error( f'Similarity search failed: {ex}' )
				results = None

		# ------------------------------------------------------------------
		# Results Rendering (with similarity threshold)
		# ------------------------------------------------------------------
		if results:
			filtered_results = [ (doc, score) for (doc, score) in results if
				score >= similarity_threshold ]

			st.caption( f'Results shown with similarity ≥ {similarity_threshold:.2f}. '
			            f'{len( filtered_results )} of {len( results )} results retained.' )

			if not filtered_results:
				st.warning( 'No results met the selected similarity threshold. '
				            'Try lowering the threshold or increasing Top-K.' )

			for rank, (doc, score) in enumerate( filtered_results, start=1 ):
				with st.expander( f'#{rank} — Similarity Score: {score:.4f}',
						expanded=(rank == 1) ):
					st.text_area( 'Chunk Text', doc.page_content, height=200, disabled=True,
						key=f'sqlite_result_{rank}' )
		else:
			st.info( 'No results to display.' )

	# ======================================================================================
	# Cloud Vector Stores
	# ======================================================================================
	else:
		remote_valid, remote_message = validate_remote_vector_state( )
		chroma_api_key = st.session_state.get( 'chroma_api_key', '' )
		chroma_tenant = st.session_state.get( 'chroma_tenant', '' )
		chroma_database = st.session_state.get( 'chroma_database', '' )
		pinecone_api_key = st.session_state.get( 'pinecone_api_key', '' )
		pinecone_index_name = chroma_database.lower( ) if isinstance( chroma_database, str ) else ''
		pinecone_index_valid = bool( re.fullmatch( r'[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?',
			pinecone_index_name ) )
		st.session_state.pinecone_index_name = pinecone_index_name

		st.markdown( f'#### {cloud_provider}' )
		st.caption( f'Document: `{document_name}` | Collection / Namespace: `{collection_name}`' )

		if cloud_provider == 'Chroma':
			st.caption( f'Chroma Database: `{chroma_database or "not configured"}`' )
		else:
			st.caption( f'Pinecone Index: `{pinecone_index_name or "not configured"}`' )

		if not remote_valid:
			st.warning( remote_message )

		if cloud_provider == 'Chroma':
			credentials_valid = bool( isinstance( chroma_api_key, str ) and chroma_api_key.strip( ) and
				isinstance( chroma_tenant, str ) and chroma_tenant.strip( ) and
				isinstance( chroma_database, str ) and chroma_database.strip( ) )
			if not credentials_valid:
				st.warning( 'Chroma API Key, Tenant ID, and Database are required.' )
		else:
			credentials_valid = bool( isinstance( pinecone_api_key, str ) and pinecone_api_key.strip( ) and
				isinstance( chroma_database, str ) and chroma_database.strip( ) and
				pinecone_index_valid )
			if not pinecone_index_valid and pinecone_index_name:
				st.warning( 'The lowercased Chroma database name is not a valid Pinecone index name. '
				            'Pinecone index names must use lowercase letters, digits, or hyphens.' )
			elif not credentials_valid:
				st.warning( 'Pinecone API Key and Chroma Database are required. The Pinecone index name '
				            'is the Chroma database name lowercased.' )

		can_persist_remote = bool( remote_valid and credentials_valid and collection_name )
		col_persist, col_inspect, col_clear = st.columns( 3 )
		persist_remote = col_persist.button( 'Persist Vectors', icon='🔣', width='stretch',
			key='cloud_persist_vectors', disabled=not can_persist_remote )
		inspect_remote = col_inspect.button( 'Inspect Store', icon='🔎', width='stretch',
			key='cloud_inspect_store', disabled=not credentials_valid )
		clear_remote = col_clear.button( 'Clear Collection', icon='❌', width='stretch',
			key='cloud_clear_collection', disabled=not credentials_valid )

		# ------------------------------------------------------------------
		# Persist remote vectors
		# ------------------------------------------------------------------
		if persist_remote:
			try:
				ids = [ f'{collection_name}_{index + 1:06d}' for index in range( len( embedding_texts ) ) ]
				metadatas = [ build_vector_metadata( index, embedding_texts[ index ] ) for index in
					range( len( embedding_texts ) ) ]

				if cloud_provider == 'Chroma':
					import chromadb

					client = chromadb.CloudClient( api_key=chroma_api_key.strip( ),
						tenant=chroma_tenant.strip( ), database=chroma_database.strip( ) )
					collection = client.get_or_create_collection( name=collection_name,
						embedding_function=None )
					batch_size = 100
					for batch_start in range( 0, len( ids ), batch_size ):
						batch_end = batch_start + batch_size
						collection.upsert( ids=ids[ batch_start:batch_end ],
							embeddings=embeddings[ batch_start:batch_end ],
							documents=embedding_texts[ batch_start:batch_end ],
							metadatas=metadatas[ batch_start:batch_end ] )
					st.success( f'Persisted {len( embeddings )} vectors to Chroma collection '
					            f'`{collection_name}`.' )
				else:
					from pinecone import Pinecone

					pinecone_client = Pinecone( api_key=pinecone_api_key.strip( ) )
					available_indexes = pinecone_client.list_indexes( ).names( )

					if pinecone_index_name not in available_indexes:
						raise ValueError( f'Pinecone index `{pinecone_index_name}` does not exist.' )

					index_description = pinecone_client.describe_index( pinecone_index_name )
					index_dimension = int( getattr( index_description, 'dimension', 0 ) or 0 )

					if index_dimension != dim:
						raise ValueError( f'Pinecone index dimension {index_dimension} does not match '
						                  f'the embedding dimension {dim}.' )

					pinecone_index = pinecone_client.Index( pinecone_index_name )
					vectors = [ { 'id': vector_id, 'values': vector, 'metadata': metadata } for
						vector_id, vector, metadata in zip( ids, embeddings, metadatas ) ]
					batch_size = 100
					for batch_start in range( 0, len( vectors ), batch_size ):
						batch_end = batch_start + batch_size
						pinecone_index.upsert( vectors=vectors[ batch_start:batch_end ],
							namespace=collection_name )
					st.success( f'Persisted {len( embeddings )} vectors to Pinecone index '
					            f'`{pinecone_index_name}` namespace `{collection_name}`.' )
			except Exception as exception:
				st.error( f'{cloud_provider} persistence failed: {exception}' )

		# ------------------------------------------------------------------
		# Inspect remote vector store
		# ------------------------------------------------------------------
		if inspect_remote:
			try:
				if cloud_provider == 'Chroma':
					import chromadb

					client = chromadb.CloudClient( api_key=chroma_api_key.strip( ),
						tenant=chroma_tenant.strip( ), database=chroma_database.strip( ) )
					collection = client.get_collection( name=collection_name, embedding_function=None )
					st.metric( 'Stored Vectors', f'{collection.count( ):,}' )
					preview = collection.peek( limit=5 )
					df_preview = pd.DataFrame( {
						'ID': preview.get( 'ids', [ ] ),
						'Document': preview.get( 'documents', [ ] ),
						'Metadata': preview.get( 'metadatas', [ ] ),
					} )
					st.data_editor( df_preview, use_container_width=True, hide_index=True,
						disabled=True, key='chroma_preview' )
				else:
					from pinecone import Pinecone

					pinecone_client = Pinecone( api_key=pinecone_api_key.strip( ) )
					pinecone_index = pinecone_client.Index( pinecone_index_name )
					stats = pinecone_index.describe_index_stats( )
					st.json( stats.to_dict( ) if hasattr( stats, 'to_dict' ) else stats )
			except Exception as exception:
				st.error( f'{cloud_provider} inspection failed: {exception}' )

		# ------------------------------------------------------------------
		# Clear remote collection / namespace
		# ------------------------------------------------------------------
		if clear_remote:
			try:
				if cloud_provider == 'Chroma':
					import chromadb

					client = chromadb.CloudClient( api_key=chroma_api_key.strip( ),
						tenant=chroma_tenant.strip( ), database=chroma_database.strip( ) )
					client.delete_collection( name=collection_name )
					st.warning( f'Deleted Chroma collection `{collection_name}`.' )
				else:
					from pinecone import Pinecone

					pinecone_client = Pinecone( api_key=pinecone_api_key.strip( ) )
					pinecone_index = pinecone_client.Index( pinecone_index_name )
					pinecone_index.delete( delete_all=True, namespace=collection_name )
					st.warning( f'Cleared Pinecone namespace `{collection_name}`.' )
			except Exception as exception:
				st.error( f'{cloud_provider} clear operation failed: {exception}' )

		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.subheader( 'Similarity Search' )
		query_text = st.text_area( 'Query Text',
			placeholder='Enter text to search for semantically similar chunks…', height=100,
			key='cloud_query_text' )
		top_k = st.slider( 'Top-K Results', min_value=1, max_value=20, value=5, step=1,
			key='cloud_top_k' )
		run_remote_search = st.button( 'Search', icon='🔎', key='cloud_search',
			disabled=not bool( credentials_valid and remote_valid and query_text.strip( ) ) )

		if run_remote_search:
			try:
				query_vector = generate_query_embedding( query_text )

				if len( query_vector ) != dim:
					raise ValueError( f'Query-vector dimension {len( query_vector )} does not match '
					                  f'the stored dimension {dim}.' )

				if cloud_provider == 'Chroma':
					import chromadb

					client = chromadb.CloudClient( api_key=chroma_api_key.strip( ),
						tenant=chroma_tenant.strip( ), database=chroma_database.strip( ) )
					collection = client.get_collection( name=collection_name, embedding_function=None )
					response = collection.query( query_embeddings=[ query_vector ], n_results=int( top_k ),
						include=[ 'documents', 'metadatas', 'distances' ] )
					documents = response.get( 'documents', [ [ ] ] )[ 0 ]
					metadatas = response.get( 'metadatas', [ [ ] ] )[ 0 ]
					distances = response.get( 'distances', [ [ ] ] )[ 0 ]

					for rank, (result_text, metadata, distance) in enumerate(
							zip( documents, metadatas, distances ), start=1 ):
						with st.expander( f'#{rank} — Distance: {float( distance ):.4f}',
								expanded=(rank == 1) ):
							st.json( metadata )
							st.text_area( 'Chunk Text', result_text or '', height=200, disabled=True,
								key=f'chroma_search_result_{rank}' )
				else:
					from pinecone import Pinecone

					pinecone_client = Pinecone( api_key=pinecone_api_key.strip( ) )
					pinecone_index = pinecone_client.Index( pinecone_index_name )
					response = pinecone_index.query( vector=query_vector, top_k=int( top_k ),
						include_metadata=True, namespace=collection_name )
					matches = getattr( response, 'matches', [ ] )

					for rank, match in enumerate( matches, start=1 ):
						metadata = getattr( match, 'metadata', { } ) or { }
						score = float( getattr( match, 'score', 0.0 ) or 0.0 )
						with st.expander( f'#{rank} — Similarity Score: {score:.4f}',
								expanded=(rank == 1) ):
							st.json( metadata )
							chunk_text = str( metadata.get( 'chunk_text', '' ) )
							st.text_area( 'Chunk Text', chunk_text, height=200, disabled=True,
								key=f'pinecone_search_result_{rank}' )
			except Exception as exception:
				st.error( f'{cloud_provider} similarity search failed: {exception}' )

	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

