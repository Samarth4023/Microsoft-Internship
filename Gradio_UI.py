import gradio as gr
import os

class GradioUI:
    def __init__(self, agent):
        self.agent = agent

    def process_input(self, query: str, pdf_file: gr.File = None):
        """
        Processes user input, which can include a text query and optionally a PDF file.
        If a PDF is uploaded, it assumes the query is a question about it and calls
        the Document Q&A tool directly. Otherwise, it sends the query to the agent.run() method.

        Args:
            query (str): The text query from the user.
            pdf_file (gr.File, optional): The uploaded PDF file object provided by Gradio. Defaults to None.

        Returns:
            str: The response from the agent or tool.
        """
        # Check if a PDF file is uploaded AND there is a text query (assumed to be the question)
        if pdf_file is not None and query and query.strip():
            print(f"PDF file uploaded: {pdf_file.name}")
            print(f"Query (assumed question for PDF): {query}")
            try:
                # --- Call the Document Q&A tool directly ---
                # We assume the document_qna_tool is at index 4 as per your app.py initialization
                # Gradio's gr.File object has a .name attribute which is the file path
                # The document_qna_tool expects a file path string.
                print("Detected PDF upload and query. Calling document_qna_tool...")
                # The tool signature is document_qna_tool(pdf_path: str, question: str)
                response = self.agent.tools[4](pdf_file.name, query)
                print("Document Q&A tool finished.")
                # Optional: Clean up the uploaded file after processing
                # import os
                # try:
                #     if os.path.exists(pdf_file.name):
                #         os.remove(pdf_file.name)
                #         print(f"Cleaned up file: {pdf_file.name}")
                # except Exception as e:
                #     print(f"Error cleaning up file {pdf_file.name}: {e}")
                return response

            except IndexError:
                 return "Error: Document Q&A tool not found at the expected index (4). Check agent tool setup in app.py."
            except Exception as e:
                print(f"Error during Document Q&A tool execution: {e}")
                # Optional: Clean up the uploaded file on error too
                # import os
                # try:
                #     if os.path.exists(pdf_file.name):
                #         os.remove(pdf_file.name)
                #         print(f"Cleaned up file on error: {pdf_file.name}")
                # except Exception as e:
                #     print(f"Error cleaning up file {pdf_file.name} on error: {e}")
                return f"An error occurred during Document Q&A: {str(e)}"

        # If no PDF file, or query is empty when file is present, handle as a general agent query
        elif query and query.strip():
            print(f"No PDF file or query is for general task. Processing with agent.run(): {query}")
            try:
                # --- Call the agent's run method for general queries ---
                response = self.agent.run(query)
                print("Agent.run finished for general query.")
                return response
            except Exception as e:
                print(f"Error during agent.run: {e}")
                return f"An error occurred while processing your request: {str(e)}"

        # Handle cases where only a PDF is uploaded without a question, or no input at all
        elif pdf_file is not None:
             # Optional: Clean up the uploaded file if no question was provided
            # import os
            # try:
            #     if os.path.exists(pdf_file.name):
            #         os.remove(pdf_file.name)
            #         print(f"Cleaned up file: {pdf_file.name}")
            # except Exception as e:
            #     print(f"Error cleaning up file {pdf_file.name}: {e}")
            return "Please enter a question in the textbox to ask about the uploaded PDF."
        else:
            return "Please enter a request or upload a document for analysis."


    def launch(self):
        """
        Launches the Gradio user interface with input components stacked vertically.
        """
        with gr.Blocks() as demo:
            gr.Markdown("# Multi-Tool AI Agent with Document Upload")
            gr.Markdown(
                "Enter your request and optionally upload a PDF document. "
                "If you upload a PDF, the text box should contain your question about the document. "
                "Otherwise, use the text box for general requests (weather, search, etc.)."
            )

            # Stack components vertically by default within gr.Blocks
            # Text input for the user's query or question
            query_input = gr.Textbox(
                label="Enter your request or question:",
                placeholder="e.g., What is the weather in London? Search for the latest news about AI. What does this document say about [topic]? Summarize the document.",
                lines=3, # Gives height to the textbox
                interactive=True
            )

            # File upload component for PDF documents - placed below the textbox
            pdf_upload = gr.File(
                label="Upload PDF (Optional - for Document Q&A)",
                file_types=[".pdf"], # Restrict file types to PDF
                interactive=True,
                # The vertical size of this component is somewhat fixed by Gradio
            )

            # Button to trigger the processing function - placed below the file upload
            submit_btn = gr.Button("Submit")

            # Output field below the button
            agent_output = gr.Textbox(
                label="Agent's Response:",
                interactive=False,
                lines=10,
                autoscroll=True
            )

            # Link the button click to the process_input function
            # Pass both query_input (text) and pdf_upload (file) as inputs
            submit_btn.click(
                fn=self.process_input,
                inputs=[query_input, pdf_upload], # This list specifies all inputs
                outputs=agent_output # This specifies the output where the return value goes
            )

            # Optional examples - adjust these based on your tools and whether you have default files
            # examples = [
            #     ["What is the time in Berlin?", None], # Example with only text
            #     ["Generate an image of a robot cooking pasta.", None], # Example with only text
            #     # Example for document Q&A - requires a default file path accessible by the tool
            #     # and needs to be in the same directory or accessible path
            #     # ["Summarize the introduction section", "sample_document.pdf"] # Example with text and file
            # ]
            # gr.Examples(examples=examples, inputs=[query_input, pdf_upload]) # Examples take same inputs as function


        # Launch the Gradio interface
        # Setting share=True creates a public URL (useful for demos, be mindful of security/costs)
        # Setting inline=False opens the app in a new browser tab
        demo.launch(share=False, inline=False)