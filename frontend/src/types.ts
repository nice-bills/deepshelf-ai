export interface Book {
  id: string;
  title: string;
  authors: string[];
  description?: string;
  genres: string[];
  cover_image_url?: string;
}

export interface RecommendationResult {
  book: Book;
  similarity_score: number;
}

export interface RecommendByQueryRequest {
  query: string;
  top_k: number;
}

export interface BookCluster {
  id: number;
  name: string;
  size: number;
  top_books: Book[];
}
