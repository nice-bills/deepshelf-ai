# 📚 Serendipity Frontend (Book Recommender)

A modern, responsive React application for discovering books using semantic search and AI-powered personalization.

![Serendipity UI](https://placehold.co/800x400?text=Serendipity+UI+Preview)

## ✨ Features

*   **Semantic Search:** Search by "vibe", plot description, or character traits (e.g., "A sci-fi thriller about AI consciousness").
*   **Persona Picker (New!):** Experience the personalization engine by switching between pre-defined user personas (e.g., "The Futurist", "The Horror Fan") to see how recommendations adapt.
*   **Curated Collections:** Explore automatically generated book clusters based on semantic similarity.
*   **Personalized Recommendations:** Get tailored suggestions based on your (simulated) reading history.
*   **Modern UI:** Dark mode support, glassmorphism design, and responsive grid layout.

## 🛠️ Tech Stack

*   **Framework:** React 18 (Vite)
*   **Language:** TypeScript
*   **Styling:** Tailwind CSS
*   **Icons:** Lucide React
*   **State:** React Hooks (Context-free for simplicity)

## 🚀 Getting Started

### Prerequisites

*   Node.js (v18 or higher)
*   npm or yarn

### Installation

1.  **Navigate to the frontend directory:**
    ```bash
    cd books/frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

### Running Locally

1.  **Start the Development Server:**
    ```bash
    npm run dev
    ```
    The app will run at `http://localhost:5173`.

2.  **Connect to Backend:**
    By default, the frontend expects the **Books API** to be running at `http://localhost:8000`.
    
    If your backend is running elsewhere (e.g., in Docker or Hugging Face Spaces), create a `.env` file in `books/frontend`:
    ```env
    VITE_API_URL=http://localhost:8000
    ```

## 🧪 Testing the "Persona Picker"

1.  Open the app.
2.  Click on one of the **Demo Personas** cards (e.g., "The Futurist").
3.  The app will:
    *   Load the persona's reading history.
    *   Trigger a personalization request to the backend.
    *   Display a curated list of books matching that persona's taste.
    *   Show a toast notification confirming the switch.

## 📦 Deployment

### Build for Production

```bash
npm run build
```
This generates static assets in the `dist/` folder, ready for deployment to Vercel, Netlify, or GitHub Pages.

### Docker

A `Dockerfile` is provided in the root `books` directory (or use the multi-stage build pattern) to serve the static assets via Nginx.