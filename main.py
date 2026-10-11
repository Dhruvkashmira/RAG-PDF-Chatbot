#importing files 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import Query
from qdrant_client.models import Distance, VectorParams
import pymupdf
import uuid
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from groq import Groq

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
if not client.collection_exists("pdf_chunks"):
    client.create_collection(
        collection_name="pdf_chunks",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
app = FastAPI()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file"
        )

    pdf_bytes = await file.read()

    try:
        pdf = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = ""
        for page in pdf:
            text += page.get_text() + "\n"

        pdf.close()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="PDF could not be read"
        )

    # Text chunking
    chunk_size = 500
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Create Qdrant points
    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "filename": file.filename,
                    "chunk_index": i,
                    "text": chunk
                }
            )
        )

    # Store vectors in Qdrant
    if points:
        client.upsert(
            collection_name="pdf_chunks",
            points=points
        )

    return {
        "filename": file.filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
        "stored_in_qdrant": len(points)
    }
#2nd endpoint

@app.get("/qdrant-test")
def qdrant_test():
    return {
        "connected": client.collection_exists("pdf_chunks")
    }

#ask end point 


@app.get("/ask")
def ask_question(question: str = Query(..., min_length=1)):
    # 1. embedding of questuons
    question_embedding = model.encode(question).tolist()

    # 2. Qdrant pdf chunk retreave
    results = client.query_points(
        collection_name="pdf_chunks",
        query=question_embedding,
        limit=3
    ).points

    if not results:
        return {
            "question": question,
            "answer": "PDF mein relevant information nahi mili."
        }

    # 3. Retrieved chunks convert into context
    context = "\n\n".join(
        point.payload.get("text", "")
        for point in results
        if point.payload
    )

    # 4. Context 
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful PDF assistant. "
                    "Answer using only the provided context. "
                    "If the answer is not in the context, "
                    "say that the PDF does not contain "
                    "enough information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": len(results)
    }
