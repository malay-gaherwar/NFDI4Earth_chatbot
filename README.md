# NFDI4Earth Chatbot

This repository provides a Streamlit-based chatbot that leverages OpenAI's large language models (LLMs) to answer your questions based on the FAQ section of the NFDI4Earth Living Handbook.

**Features:**

*   **OpenAI Integration:** Utilizes OpenAI's LLM 'gpt-4o-mini' for informative responses.
*   **Streamlit Interface:** Offers a user-friendly interface using Streamlit for interacting with the chatbot.
*   **Source Management:** Provides the option to remove outdated or irrelevant sources from `Sources.py`.
*   **Temporary Chat History:** Temporarily stores your conversation history.

**Setup**

1.  **Prerequisites:**

      - Docker: Ensure you have Docker installed on your system. 

2.  **Building the Docker Image:**

      - Navigate to the project directory in your terminal.

      - Run the following command to build the Docker image:

        ```bash
        docker-compose up --build
        ```

    This command utilizes Docker Compose to build the image based on the specifications in the `dockerfile` file. 

**Usage**

1.  **Running the Chatbot:**

      - You can launch the app via the local port as of now by running the docker image.
         ```bash
        docker run chatbot
        ``` 
      - And then click on the link for local deployment. (Online Deployment  is still being figured out)

    This will launch the chatbot container, and the Streamlit app will be accessible in your web browser.

2.  **OpenAI API Key:**

      - The chatbot currently requires your OpenAI API key to function. You can obtain your API key from your OpenAI account ([https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)).
      - Enter your API key in the Streamlit interface where it is prompted.

3.  **Asking Questions:**

      - Type your question in the designated text area on the Streamlit app.
      - Click the "Submit" button to send your question to the chatbot.
      - The chatbot will process your question using the FAQ links provided and generate a response based on the retrieved information.

4.  **Chat History:**

      - The chatbot temporarily stores your conversation history for the current session with the most recent question coming on the top. This allows you to see the previous questions and responses as you interact with the chatbot. When you close the app the chat history will no longer be accessible.

5.  **FAQ Links:**

      - The `data_links.yaml` file contains the links to the relevant FAQ sections of the NFDI4Earth Living Handbook.

6.  **Source Management:**

      - The `Sources.py` file manages the FAQ links used by the chatbot.
      - You can navigate to the 'Sources' page within the Streamlit app to remove outdated or irrelevant sources.

