# app.py
from flask import Flask, render_template, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import google.generativeai as genai
from llama_index.llms.google_genai import GoogleGenAI  # Updated import
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# from llama_index.embeddings import HuggingFaceeEmbedding  # Added for local embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the LLM using updated Google GenAI wrapper
llm = GoogleGenAI(model="gemini-1.5-flash", schema_type="json")

# Use local embeddings instead of OpenAI
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model  # Configure the embedding model globally

pdf_reader = PDFReader()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Define pydantic models for function schemas


class InvoiceRetrieverInput(BaseModel):
    query: str = Field(..., description="Query string to search in invoice text")


class ExtractInvoiceDetailsInput(BaseModel):
    # No inputs expected here, so empty model
    pass


def extract_invoice_info(nodes):
    # Combine text from nodes for processing
    text = " ".join([node.text for node in nodes])

    # Use Gemini to extract structured information
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(
        """
        Extract the following information from this invoice text. Return as a JSON object with these fields:
        - invoice_number
        - invoice_date
        - due_date
        - vendor_name
        - vendor_address
        - customer_name
        - customer_address
        - total_amount
        - tax_amount
        - items (array of {description, quantity, unit_price, amount})
        
        If any field is not found, use null.
        
        Invoice text:
        """
        + text
    )

    try:
        response_text = response.text
        start_index = response_text.find("{")
        end_index = response_text.rfind("}") + 1

        if start_index >= 0 and end_index > start_index:
            json_str = response_text[start_index:end_index]
            return json.loads(json_str)
        else:
            return {"raw_extraction": response_text}
    except Exception as e:
        return {"error": str(e), "raw_response": response.text}


def process_pdf(file_path):
    try:
        # Load the PDF documents
        docs = pdf_reader.load_data(file_path)

        # Parse into nodes for indexing
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(docs)

        # Create vector index and retriever
        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=3)
        # Define tools with proper pydantic schemas
        tools = [
            FunctionTool.from_defaults(
                name="invoice_retriever",
                fn=lambda query: [
                    {"text": node.text} for node in retriever.retrieve(query)
                ],
                description="Retrieves information from the invoice PDF",
                fn_schema=InvoiceRetrieverInput,
            ),
            FunctionTool.from_defaults(
                name="extract_invoice_details",
                fn=lambda: extract_invoice_info(nodes),
                description="Extracts key details from the invoice like invoice number, date, amount, vendor name, etc.",
                fn_schema=ExtractInvoiceDetailsInput,
            ),
        ]

        # Create agent worker with the LLM
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools, llm=llm, verbose=True
        )

        # Create agent runner
        agent = AgentRunner(agent_worker)

        return agent

    except Exception as e:
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Create unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            # Process the PDF
            agent = process_pdf(file_path)
            # Extract invoice details
            invoice_details = agent.query(
                "Extract all important invoice details including invoice number, date, amounts, vendor, and line items"
            )
            return jsonify(
                {
                    "success": True,
                    "filename": filename,
                    "invoice_data": invoice_details.response,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400


@app.route("/query", methods=["POST"])
def query_document():
    data = request.get_json()

    if not data or "filename" not in data or "query" not in data:
        return jsonify({"error": "Missing filename or query"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], data["filename"])

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        agent = process_pdf(file_path)
        response = agent.query(data["query"])

        return jsonify({"response": response.response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
