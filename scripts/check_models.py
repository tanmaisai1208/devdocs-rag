from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    # Load API credentials from the project's .env file
    load_dotenv()

    # 1. Test Gemini generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )

    response = llm.invoke(
        "In one sentence, explain what a vector embedding is."
    )

    print("\n--- Gemini API test ---")
    print(response.content)

    # 2. Test local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector = embeddings.embed_query(
        "Python database connection error"
    )

    print("\n--- Local embedding test ---")
    print("Vector dimensions:", len(vector))
    print("First five values:", vector[:5])


if __name__ == "__main__":
    main()