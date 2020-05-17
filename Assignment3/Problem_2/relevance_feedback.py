import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def relevance_feedback(vec_docs, vec_queries, sim, n=10, return_vec_queries=False):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    vec_queries_copy = vec_queries.copy()

    rf_sim = sim
    alpha = 0.8
    beta = 0.2
    for epoch in range(3):
        for query_i in range(0, vec_queries_copy.shape[0]):
            print("\r {} - {}/{}".format(epoch, query_i+1, vec_queries_copy.shape[0]), end="")
            sorted_sim_j = np.argsort(rf_sim[:, query_i])
            top_sim_docs = sorted_sim_j[-n:]
            for doc in top_sim_docs:
                vec_queries_copy[query_i] += vec_docs[doc] * alpha
            bottom = sorted_sim_j[:n]
            for word in bottom:
                vec_queries_copy[query_i] -= vec_docs[word] * beta
        print()

    rf_sim = cosine_similarity(vec_docs, vec_queries_copy)

    if return_vec_queries:
        return rf_sim, vec_queries_copy

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    rf_sim = sim
    alpha = 0.8
    beta = 0.2
    vec_queries_copy = vec_queries.copy()
    inv_dic = {v: k for k, v in tfidf_model.vocabulary_.items()}
    dic = {v: k for k, v in inv_dic.items()}

    number_of_terms = 10

    A = [[0 for j in range(9493)] for i in range(1033)]
    inv_matrix = tfidf_model.inverse_transform(vec_docs)
    for i in range(1033):
        doc = inv_matrix[i]
        for i in range(len(doc)):
            # print(doc[j], end=" ")
            key = dic[doc[i]]
            A[i][key] += 1
    A = np.array(A).astype(np.float)
    print("C.shape", A.shape)
    C = np.matmul(A.transpose(), A)
    thesaurus_docs = C
    print("thesaurus_docs.shape", thesaurus_docs.shape, rf_sim.shape)

    for epoch in range(3):
        for query_i in range(0, vec_queries_copy.shape[0]):
            print("\r {} - {}/{}".format(epoch, query_i + 1, vec_queries_copy.shape[0]), end=" ")
            new_term_set = []

            sorted_sim_j = np.argsort(rf_sim[:, query_i])
            top_sim_docs = sorted_sim_j[-n:]
            for doc in top_sim_docs:
                vec_queries_copy[query_i] += vec_docs[doc] * alpha
            least_sim_docs = sorted_sim_j[:n]
            for doc in least_sim_docs:
                vec_queries_copy[query_i] -= vec_docs[doc] * beta

            top_query_words = np.argsort(vec_queries_copy[query_i, :].toarray().flatten())[-n:]
            for word in top_query_words:
                x = thesaurus_docs[word].flatten()
                top_term_indices = np.argsort(x)[-number_of_terms:]
                for term_index in top_term_indices:
                    new_term_set.append(inv_dic[term_index])
            if len(new_term_set) > 0:
                adding = tfidf_model.transform([' '.join(new_term_set)])
                vec_queries_copy[query_i] += adding
        print()

    rf_sim = cosine_similarity(vec_docs, vec_queries_copy)

    return rf_sim
