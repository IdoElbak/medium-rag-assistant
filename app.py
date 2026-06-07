import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize API Clients
llmod_client = OpenAI(
    api_key=os.environ.get("LLMOD_API_KEY"),
    base_url="https://api.llmod.ai"
)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("medium-rag")

# RAG Hyperparameters 
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 20
OVERFETCH_K = 150

class PromptRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def serve_testing_gui():
    """A simple, un-graded UI to make testing easier for you."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>RAG Tester</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h2>Test Your Medium RAG Assistant</h2>
        <textarea id="question" rows="4" style="width: 100%;" placeholder="Ask a question..."></textarea><br><br>
        <button onclick="ask()" style="padding: 10px 20px;">Send to /api/prompt</button>
        <hr>
        <h3>Response:</h3>
        <p id="response-text" style="white-space: pre-wrap; background: #f4f4f4; padding: 15px;"></p>
        <h3>Retrieved Context (Raw JSON):</h3>
        <pre id="context-text" style="background: #eee; padding: 15px; overflow-x: auto;"></pre>

        <script>
            async function ask() {
                const q = document.getElementById('question').value;
                document.getElementById('response-text').innerText = "Thinking...";
                document.getElementById('context-text').innerText = "";
                
                try {
                    const res = await fetch('/api/prompt', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: q})
                    });
                    
                    const data = await res.json();
                    
                    // This block catches the server crash and stops 'undefined' from rendering
                    if (!res.ok) {
                        document.getElementById('response-text').innerText = "Server Error: " + (data.detail || "An unknown error occurred.");
                        return;
                    }
                    
                    document.getElementById('response-text').innerText = data.response;
                    document.getElementById('context-text').innerText = JSON.stringify(data.context, null, 2);
                } catch (err) {
                    document.getElementById('response-text').innerText = "Network Error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/api/stats")
def get_stats():
    """Returns the hyperparameter configuration exactly as required."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.post("/api/prompt")
def handle_prompt(request: PromptRequest):
    question = request.question
    
    # try:
    #     # Embed user's query
    #     embed_response = llmod_client.embeddings.create(
    #         input=question,
    #         model="4UHRUIN-text-embedding-3-small"
    #     )
    #     query_vector = embed_response.data[0].embedding
    
    try:
        # De-noise the prompt into a clean search query
        query_optimizer = llmod_client.chat.completions.create(
            model="4UHRUIN-gpt-5-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a search query optimizer. Extract only the core subject matter, "
                        "historical entities, and unique semantic keywords from the user's prompt "
                        "to use for a vector database search. Completely strip out all instructions "
                        "like 'find an article', 'summarize', 'provide the title', 'list exactly 3', "
                        "or 'which article'. Return ONLY the raw keywords separated by spaces."
                        "Use only keywords that appear in the question."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=1.0
        )
        search_keywords = query_optimizer.choices[0].message.content.strip()
        
        # Embed the clean keywords instead of the raw conversational question
        embed_response = llmod_client.embeddings.create(
            input=search_keywords, 
            model="4UHRUIN-text-embedding-3-small"
        )
        query_vector = embed_response.data[0].embedding

        # Over-fetch from Pinecone to find raw matches
        pc_response = index.query(
            vector=query_vector,
            top_k=OVERFETCH_K,
            include_metadata=True
        )
        
        # Deduplicate matches to guarantee distinct articles
        seen_articles = set()
        context_texts = []
        output_context = []
        
        for match in pc_response.matches:
            metadata = match.metadata or {}
            
            article_id = str(metadata.get("article_id", match.id))
            
            if article_id in seen_articles:
                continue
                
            seen_articles.add(article_id)
            chunk_text = metadata.get("chunk", "")
            
            context_texts.append(
                f"Article ID: {article_id}\n"
                f"Title: {metadata.get('title', 'Unknown')}\n"
                f"Authors: {metadata.get('authors', 'Unknown')}\n"
                f"Timestamp: {metadata.get('timestamp', 'Unknown')}\n"
                f"Tags: {metadata.get('tags', 'Unknown')}\n"
                f"Content: {chunk_text}"
            )
            
            # Format the item for the mandatory context array output
            output_context.append({
                "article_id": article_id,
                "title": metadata.get("title", "Unknown"),
                "chunk": chunk_text,
                "score": match.score
            })
            
            # Stop once the diverse context window is filled
            if len(output_context) >= TOP_K:
                break
                
        combined_context = "\n\n---\n\n".join(context_texts)
        
        # Construct System and User prompts
        system_prompt = (
            "You are a Medium-article assistant that answers questions strictly and only "
            "based on the Medium articles dataset context provided to you (metadata "
            "and article passages). You must not use any external knowledge, the open "
            "internet, or information that is not explicitly contained in the retrieved "
            "context. If the answer cannot be determined from the provided context, "
            "respond: 'I don't know based on the provided Medium articles data.' "
            "Always explain your answer using the given context, quoting or "
            "paraphrasing the relevant article passage or metadata when helpful."
        )
        
        user_prompt = f"Context:\n{combined_context}\n\nQuestion: {question}"
        
        # Call Chat Model
        chat_response = llmod_client.chat.completions.create(
            model="4UHRUIN-gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0
        )
        
        final_answer = chat_response.choices[0].message.content
        
        return {
            "response": final_answer,
            "context": output_context,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))