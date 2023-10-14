# CosineSimilaritySearch

A class for generating and searching a cosine similarity matrix for a set of documents.

## Parameters

- `documents_path (str)`: The path to the file containing the list of documents.

## Attributes

- `tfidf_vectorizer (TfidfVectorizer)`: The TF-IDF vectorizer for text data.
- `cosine_matrix (numpy.ndarray)`: The cosine similarity matrix between documents.

## Methods

### `__init__(self, documents_path)`

Constructor for the `CosineSimilaritySearch` class.

### `_load_documents(self)`

Load documents from the specified file. Returns a list of document texts.

### `_save_cosine_matrix(self, filename)`

Save the cosine similarity matrix to a file.

Parameters:

- `filename (str)`: The name of the file to save the matrix to.

### `_load_cosine_matrix(self, filename)`

Load the cosine similarity matrix from a file.

Parameters:

- `filename (str)`: The name of the file to load the matrix from.

### `generate_cosine_matrix(self)`

Generate the cosine similarity matrix for the loaded documents.

### `search_similar_documents(self, query, top_n=5)`

Search for similar documents to a given query.

Parameters:

- `query (str)`: The query text.
- `top_n (int)`: The number of most similar documents to retrieve.

Returns a list of tuples containing (similarity_score, document_text).

### `regenerate_cosine_matrix(self)`

Regenerate the cosine similarity matrix for the loaded documents and save it.

## Usage

```python
cosine_search = CosineSimilaritySearch('blog_posts.txt')
cosine_search.generate_cosine_matrix()

query = input("Enter your Query: ")
similar_docs = cosine_search.search_similar_documents(query, top_n=5)

print(similar_docs)
```
