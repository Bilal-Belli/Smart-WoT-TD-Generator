# Smart Web of Things (WoT) Thing Description (TD) Generator & RAG Search System

An intelligent, full-stack application designed to **generate**, **validate**, and **search** W3C-compliant [Web of Things (WoT) Thing Descriptions (TDs)](https://www.w3.org/TR/wot-thing-description/).

This project integrates state-of-the-art Large Language Models (LLMs) with a robust self-correction validation loop and a highly optimized **two-stage Retrieval-Augmented Generation (RAG) search pipeline** to provide a seamless developer experience for WoT devices.

---
## 🚀 Demo

https://github.com/user-attachments/assets/a7bb78ba-3b34-4427-8b50-a5cd99752786

## 🚀 Key Features

### 1. Agentic TD Generation Loop

- **Multi-Model Support:** Generate TDs using Gemini, OpenAI, Mistral, Ollama, or DeepSeek.
- **Self-Correction Loop:** The server runs an automatic iterative generation loop (up to 10 attempts). When the LLM outputs a TD, it is validated using the official `@thing-description-playground/cli`.
- **Error Feedback:** If validation fails, the error logs are fed back into the LLM as prompt feedback, prompting it to fix structural/semantic issues until a valid TD is produced.
- **Local DB Storage:** Automatically stores verified TDs in a local `things-database.json` database.

### 2. Two-Stage RAG Search Pipeline (`wot-rag.py`)

- **Stage 1 (Local Pre-Filtering):** Uses a lightweight local embedding model (`sentence-transformers/all-MiniLM-L6-v2`) to quickly filter and rank candidates down to the top 5 relevant devices, ensuring low latency.
- **Stage 2 (Precise LLM Routing):** Leverages Gemini's native tool-calling (`with_structured_output` with a Pydantic schema) to determine the best-matching device ID from the candidate pool in a single, highly accurate API call.
- **Database Fallbacks:** Integrates with **MongoDB** for semantic queries, with a built-in local JSON fallback if MongoDB is not available.

### 3. Modern Interactive Web UI

- **TD Generator Dashboard (`/`):** A custom UI to enter device descriptions, select LLM providers, and watch the live agentic iteration logs as they validate.
- **Saved TDs Repository (`/saved-things`):** View, explore, and delete successfully validated Thing Descriptions.
- **Interactive Search Terminal (`/search-terminal`):** A web-based terminal interface communicating over WebSockets (`Socket.io`) and `node-pty` to directly interact with the python RAG system in real-time.

---

## 🛠️ Tech Stack

### Backend

- **Node.js & Express:** Hosts APIs for generation, retrieval, and management of TDs.
- **Socket.io:** Handles real-time duplex communication for the web terminal.
- **node-pty:** Spawns and manages pseudo-terminal (pty) processes.
- **W3C WoT Playground Validator:** Performs official JSON/W3C validation.

### Python (RAG Pipeline)

- **LangChain:** Orchestrates documents, prompts, embeddings, and vector stores.
- **LangChain Google GenAI:** Integrates the Gemini API (`gemini-2.5-flash` and `gemini-embedding-001`).
- **ChromaDB:** Local vector database used to index candidates per query.
- **Sentence-Transformers:** Embedded local text vectorization for fast pre-filtering.
- **PyMongo:** Connects to MongoDB for document storage.

### Frontend

- **HTML5, CSS3, & Vanilla JavaScript:** Provides a fully interactive user interface.
- **Xterm.js (via socket.io):** Powering the real-time terminal UI.

---

## 📥 Installation

### 1. Prerequisites

- **Node.js** (v16.0.0 or higher)
- **Python** (v3.10 or higher)
- **MongoDB** (Optional, default settings will fall back to local JSON files if MongoDB is not running)

### 2. Clone the Repository

```bash
git clone https://github.com/Bilal-Belli/Smart-WoT-TD-Generator.git
cd Smart-WoT-TD-Generator
```

### 3. Install Node.js Dependencies

```bash
npm install
```

### 4. Install Python Dependencies

It is recommended to install python dependencies in a virtual environment or globally:

```bash
pip install numpy sentence-transformers langchain langchain-community langchain-google-genai langchain-chroma pymongo python-dotenv pydantic
```

### 5. Environment Configuration

Copy `.env.example` to a new file named `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in the required API keys for the model providers you intend to use:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
OLLAMA_API_KEY=your_ollama_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## ⚙️ Custom Configuration

If your local Python environment or directory paths differ from the default configuration, you may need to update the WebSocket handler paths in [server.js](file:///c:/Users/Hp/OneDrive/Bureau/TD-IA-Gen/server.js#L464-L477).

Locate the `io.on('connection')` block in `server.js` and update the shell command arguments if needed:

```javascript
const shell = 'powershell.exe';
const args = [
    '-NoProfile', 
    '-ExecutionPolicy', 'Bypass', 
    '-Command', 
    `& C:\\Python312\\python.exe ./wot-rag.py` // Update path to your python executable
];
```

---

## 🚀 Running the Project

### 1. Start the Express Server

```bash
npm start
```

The server will start on [http://localhost:3000](http://localhost:3000).

### 2. Access the Application

Open your web browser and navigate to:

- **Dashboard / Generator:** `http://localhost:3000/`
- **Saved TDs:** `http://localhost:3000/saved-things`
- **Interactive Search Terminal:** `http://localhost:3000/search-terminal`

### 3. Running RAG Search in CLI

You can also run the RAG search system interactively directly in your terminal:

```bash
python wot-rag.py
```

Type your natural language search query (e.g. *"Show me the weather station with a humidity sensor"*), or type `stats` to view search telemetry.
