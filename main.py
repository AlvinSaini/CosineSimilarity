import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class CosineSimilaritySearch:
    """
    A class for generating and searching a cosine similarity matrix for a set of documents.

    Parameters:
    documents_path (str): The path to the file containing the list of documents.

    Attributes:
    tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer for text data.
    cosine_matrix (numpy.ndarray): The cosine similarity matrix between documents.
    """

    def __init__(self, documents_path):
        self.documents_path = documents_path
        self.tfidf_vectorizer = TfidfVectorizer()
        self.cosine_matrix = None

    def _load_documents(self):
        """
        Load documents from the specified file.

        Returns:
        list: A list of document texts.
        """
        with open(self.documents_path, 'r', encoding='utf-8') as file:
            return file.readlines()

    def _save_cosine_matrix(self, filename):
        """
        Save the cosine similarity matrix to a file.

        Parameters:
        filename (str): The name of the file to save the matrix to.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.cosine_matrix, file)

    def _load_cosine_matrix(self, filename):
        """
        Load the cosine similarity matrix from a file.

        Parameters:
        filename (str): The name of the file to load the matrix from.
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.cosine_matrix = pickle.load(file)
        else:
            self.cosine_matrix = None

    def generate_cosine_matrix(self):
        """
        Generate the cosine similarity matrix for the loaded documents.
        """
        documents = self._load_documents()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.cosine_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    def search_similar_documents(self, query, top_n=5):
        """
        Search for similar documents to a given query.

        Parameters:
        query (str): The query text.
        top_n (int): The number of most similar documents to retrieve.

        Returns:
        list: A list of tuples containing (similarity_score, document_text).
        """
        if self.cosine_matrix is None:
            raise ValueError("Cosine matrix not generated. Please call generate_cosine_matrix() first.")

        query_vector = self.tfidf_vectorizer.transform([query])
        cosine_similarities_to_query = linear_kernel(query_vector, self.tfidf_vectorizer.transform(self._load_documents()))
        most_similar_indices = cosine_similarities_to_query.argsort()[0][::-1][:top_n]

        similar_documents = []
        for index in most_similar_indices:
            similarity_score = cosine_similarities_to_query[0][index]
            document = self._load_documents()[index]
            similar_documents.append((similarity_score, document))

        return similar_documents

    def regenerate_cosine_matrix(self):
        """
        Regenerate the cosine similarity matrix for the loaded documents and save it.
        """
        self.generate_cosine_matrix()
        if self.cosine_matrix is not None:
            self._save_cosine_matrix('cosine_matrix.pkl')
            print("Cosine matrix regenerated and saved.")




if __name__ == '__main__':
    cosine_search = CosineSimilaritySearch('blog_posts.txt')
    cosine_search.generate_cosine_matrix()

    run = True
    while run:
        query = input("Enter your Query: ")

        if query.lower() == "quit()":
            run = False
            break
        else:
            similar_docs = cosine_search.search_similar_documents(query, top_n=5)
            print(similar_docs)




