"""
RevAI Chatbot Application

This Streamlit application provides a chat interface for users to interact with their documents 
using AI-powered semantic search and natural language processing. The application uses 
OpenAI's embeddings and chat models to provide relevant responses based on document content.

Features:
- Document-based Q&A using semantic search
- User-specific document repositories
- Interactive chat interface
"""

# Standard library imports
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize environment variables
load_dotenv()

# Initialize OpenAI client
client = None

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def init_openai_client():
    """
    Initialize the OpenAI client with API key from environment variables.
    
    Returns:
        OpenAI: Initialized OpenAI client object
    """
    global client
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.error("OpenAI API key not found in environment variables")
    return client

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

def sanitize_customer_id(customer_id):
    """
    Sanitize customer ID to prevent SQL injection and other security issues.
    
    Args:
        customer_id (str): The raw customer ID input
        
    Returns:
        str or None: Sanitized customer ID if valid, None otherwise
    """
    
    if re.match(r"^[a-zA-Z0-9_-]+$", customer_id):
        return customer_id
    logger.warning(f"Invalid customer ID: {customer_id}")
    return None

def get_customer_table(customer_id):
    """
    Get the database table name associated with a customer ID.
    
    Args:
        customer_id (str): The customer ID to look up
        
    Returns:
        str or None: The table name if found, None otherwise
    """
    sanitized_customer_id = sanitize_customer_id(customer_id)
    if not sanitized_customer_id:
        logger.error("Customer ID is invalid or potentially malicious")
        return None

    try:
        logger.info(f"Fetching table for customer ID: {sanitized_customer_id}")
        response = (
            supabase.table("customer_mappings")
            .select("table_name")
            .eq("customer_id", sanitized_customer_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            table_name = response.data[0]["table_name"]
            logger.info(f"Found table '{table_name}' for customer ID: {customer_id}")
            return table_name
        logger.warning(f"No table found for customer ID: {customer_id}")
        return None
    except Exception as e:
        logger.error(f"Error finding customer table: {str(e)}")
        return None

def get_embeddings(text):
    """
    Generate embeddings for the provided text using OpenAI's embedding model.
    
    Args:
        text (str): The text to generate embeddings for
        
    Returns:
        list: The embedding vector if successful, empty list otherwise
    """
    try:
        logger.info("Generating embeddings for text")
        if not text or not text.strip():
            logger.warning("Empty or invalid text provided for embeddings")
            return []

        response = client.embeddings.create(
            model="text-embedding-3-large", input=text
        )
        embedding = response.data[0].embedding
        logger.info("Embeddings generated successfully")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return []

def perform_similarity_search(customer_id, query_embedding):
    """
    Perform similarity search using the query embedding against customer's documents.
    
    Args:
        customer_id (str): The customer ID to search documents for
        query_embedding (list): The embedding vector for the query
        
    Returns:
        dict or None: Search results containing matches and count, None if error occurs
    """
    try:
        logger.info(f"Performing similarity search for customer ID: {customer_id}")
        table_name = get_customer_table(customer_id)
        if not table_name:
            logger.warning(f"No table found for customer ID: {customer_id}")
            return None

        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": 5,
                "table_name": table_name,
            },
        ).execute()

        if result.data and len(result.data) > 0:
            logger.info(f"Similarity search returned {len(result.data)} results")
            # Serialize the results to JSON for debugging
            output_file = "search_results.json"
            with open(output_file, "w") as f:
                json.dump(result.data, f, indent=2)
            logger.info(f"Search results saved to {output_file}")
            return {"matches": result.data, "count": len(result.data)}
        logger.info("Similarity search returned no results")
        return {"matches": [], "count": 0}
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return None

def call_inference_api(prompt, user_id):
    """
    Process user query by generating embeddings and performing similarity search.
    
    Args:
        prompt (str): The user's query
        user_id (str): The user's ID for document access
        
    Returns:
        dict: Results or error information
    """
    try:
        query_embedding = get_embeddings(prompt)
        if not query_embedding:
            return {"error": "Failed to generate embeddings for query"}

        search_results = perform_similarity_search(user_id, query_embedding)
        if search_results is None:
            return {"error": f"No data found for customer ID: {user_id}"}

        results = []
        for match in search_results.get("matches", []):
            results.append({
                "chunk_content": match.get("content", ""),
                "extra_key_data": match.get("metadata", {}).get("extra_key_data", {}),
                "document_name": match.get("metadata", {}).get("document_name", "Unknown document"),
            })
        return {"results": results}
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": f"Error during inference: {str(e)}"}

def format_chunks(chunks):
    """
    Format the retrieved document chunks into a readable text format.
    
    Args:
        chunks (list): List of document chunks from similarity search
        
    Returns:
        str: Formatted text containing content from retrieved chunks
    """
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        content = chunk.get("chunk_content", "No content")
        extra_data = chunk.get("extra_key_data", "")
        doc_name = chunk.get("document_name", "Unknown document")
        chunks_text += f"Source {i+1} ({doc_name}):\nContent: {content}\n"
        if extra_data:
            chunks_text += f"Summary: {extra_data}\n"
        chunks_text += "\n"
    return chunks_text

def prepare_messages(chunks_text):
    """
    Prepare the messages for the OpenAI API with chat history and context.
    
    Args:
        chunks_text (str): Formatted document chunks to provide as context
        
    Returns:
        list: Messages formatted for OpenAI chat completion API
    """
    messages = [{"role": "system", "content": f"You are a helpful assistant. Use the following information to answer the user's question:\n\n{chunks_text}"}]
    for msg in st.session_state.messages[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages

def generate_ai_response(messages):
    """
    Generate a response using the OpenAI API.
    
    Args:
        messages (list): Formatted messages for the OpenAI chat completion API
        
    Returns:
        str: AI-generated response or error message
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return f"Error generating response: {str(e)}"

def get_response(prompt, user_id):
    """
    Main function to handle the user query and generate a response.
    
    Args:
        prompt (str): The user's query
        user_id (str): The user's ID for document access
        
    Returns:
        tuple: (AI-generated response or error message, turnaround time)
    """
    start_time = time.time()
    
    # Display status: Calling inference API
    with st.spinner("Fetching relevant document..."):
        data = call_inference_api(prompt, user_id)
    
    if "error" in data:
        logger.error(f"Error in inference API: {data['error']}")
        response = data["error"]
    elif "results" in data:
        # Display status: Formatting document chunks
        with st.spinner("Formatting document chunks..."):
            chunks_text = format_chunks(data["results"])
        
        # Display status: Preparing messages for OpenAI
        with st.spinner("Preparing messages for OpenAI..."):
            messages = prepare_messages(chunks_text)
        
        # Display status: Generating AI response
        with st.spinner("Generating AI response..."):
            response = generate_ai_response(messages)
    else:
        response = data.get("response", "No response received from server")
    
    turnaround_time = time.time() - start_time
    return response, turnaround_time

def chat_page():
    """
    Render the chat interface page in the Streamlit app.
    
    Gets user_id from query parameters if available and uses it for document access.
    If user_id is not in query parameters, shows access denied message.
    """
    st.title("Personlized AI. Project Insights")
    
    # Get user_id from query parameters if available
    query_params = st.query_params
    user_id = ""
    
    if "user_id" in query_params and query_params["user_id"]:
        user_id = query_params["user_id"]
        logger.info(f"User ID found in query parameters: {user_id}")
        
        st.markdown("### Ask anything about your document!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input that processes on Enter key
        user_query = st.chat_input("Type your message here...")
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Get and display assistant response
            response, turnaround_time = get_response(user_query, user_id)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                st.write(response)
                st.caption(f"Response time: {turnaround_time:.2f} seconds")
    else:
        # Show access denied message if user_id is not in query parameters
        st.error("Access Denied: You need a valid user ID to access this chatbot.")
        st.info("Please use a URL with a valid user ID parameter: `https://[app-url]?user_id=YOUR_USER_ID`")
        
        # Optionally, add a placeholder for the URL to redirect users
        st.markdown("### Contact your administrator for access")
        st.write("If you believe you should have access to this chatbot, please contact your administrator.")

def main():
    """
    Main function to run the Streamlit application.
    """
    init_openai_client()
    chat_page()

if __name__ == "__main__":
    main()
