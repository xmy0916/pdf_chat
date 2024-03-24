import gradio as gr
from parse_pdf import create_db, search_most_similarity_content

db = None

def upload(
    file
) -> str:
    global db
    pdfs = [f.name for f in file]
    db = create_db(pdfs)

    return "success!"

def ask(question):
    global db
    prompt = search_most_similarity_content(db, question)
    return prompt


title = 'PDF GPT'
description = """ PDF GPT allows you to chat with your PDF file using Universal Sentence Encoder and Open AI. It gives hallucination free response than other tools as the embeddings are better than OpenAI. The returned response can even cite the page number in square brackets([]) where the information is located, adding credibility to the responses and helping to locate pertinent information quickly."""

with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            file = gr.Files(
                label='Upload your PDFs/ push commad(on mac) or control(on windows) to select multi files.', file_types=['.pdf']
            )
            question = gr.Textbox(label='Enter your question here')
            btn_upload = gr.Button(value='Submit')
            btn_upload.style(full_width=True)
            btn_ask = gr.Button(value='Submit')
            btn_ask.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn_upload.click(
            upload,
            inputs=[file],
            # outputs=[answer],
        )
        btn_ask.click(
            ask,
            inputs=[question],
            outputs=[answer],
        )


demo.launch(server_port=7860, enable_queue=True)