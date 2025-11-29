import axios from 'axios';
import type { RecommendationResult, Book, BookCluster } from './types';

const API_URL = 'http://localhost:8000';

export const api = {
  getClusters: async (): Promise<BookCluster[]> => {
    const response = await axios.get<BookCluster[]>(`${API_URL}/clusters`);
    return response.data;
  },

  recommendByQuery: async (query: string): Promise<RecommendationResult[]> => {
    const response = await axios.post<RecommendationResult[]>(`${API_URL}/recommend/query`, {
      query,
      top_k: 5,
    });
    return response.data;
  },

  searchBooks: async (query: string): Promise<{ books: Book[] }> => {
    const response = await axios.get(`${API_URL}/books/search`, {
      params: { query, page: 1, page_size: 20 },
    });
    return response.data;
  },

  submitFeedback: async (query: string, bookId: string, type: 'positive' | 'negative') => {
    await axios.post(`${API_URL}/feedback`, {
      query,
      book_id: bookId,
      feedback_type: type,
    });
  },

  explainRecommendation: async (query: string, book: Book, score: number) => {
    const response = await axios.post(`${API_URL}/explain`, {
      query_text: query,
      recommended_book: book,
      similarity_score: score
    });
    return response.data;
  },

  getRelatedBooks: async (title: string): Promise<RecommendationResult[]> => {
    const response = await axios.post<RecommendationResult[]>(`${API_URL}/recommend/title`, {
      title,
      top_k: 3,
    });
    return response.data;
  }
};
