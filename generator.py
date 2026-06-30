from groq import Groq
from query import retrieve

client = Groq()

def build_prompt(question, results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        block = f"File: {meta['file']} (lines {meta['start_line']} -{meta['end_line']})\n{doc}"
        context_blocks.append(block)
        
    context = "\n\n---\n\n".join(context_blocks)
    
    prompt = f"""You are a helpful assistant answering questions about a codebase.
    Use ONLY the context below to answer. If the answer isnt in context, say you do not know.
    
    Context: {context}
    Question: {question}
    Answer:"""
    return prompt

def ask(question):
    results = retrieve(question)
    prompt = build_prompt(question, results)
    
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user","content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "How does the app fetch data from NASA's API?"
    answer = ask(question)
    print(answer)