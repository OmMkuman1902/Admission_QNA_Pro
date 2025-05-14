"""from speech_to_text import speech_to_text
from text_to_speech import speak_text
from Lanchainback import get_qa_chain

def main():
    # Get spoken question
    question = speech_to_text()

    # Load the QA chain
    chain = get_qa_chain()

    # Ask question
    response = chain.invoke({"query": question})
    answer = response["result"]

    # Speak answer
    speak_text(answer)

if __name__ == "__main__":
    main()
"""