import gradio as gr
from parse_pdf import create_db, search_most_similarity_content
from llm import init_model, get_answer

model, tokenizer = init_model()
db = None

def upload(
    file
) -> str:
    global db
    pdfs = [f.name for f in file]
    db = create_db(pdfs)

    print("db create sucess!!!")

    return pdfs[0]

def ask(question):
    global db
    prompt = search_most_similarity_content(db, question)

    print(prompt)

    answer = get_answer(prompt, model, tokenizer)

    return answer


title = 'PDF GPT'
description = """这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行这里写啥都行"""
with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            file = gr.Files(
                label='Upload your PDFs/ push commad(on mac) or control(on windows) to select multi files.', file_types=['.pdf']
            )
            btn_upload = gr.Button(value='上传', elem_id="btn_upload")
            # btn_upload.style(full_width=True)
            gr.Markdown("<style> #btn_upload { width: 100%; } </style>")
            question = gr.Textbox(label='Enter your question here')
            btn_ask = gr.Button(value='提问', elem_id="btn_ask")
            gr.Markdown("<style> #btn_ask { width: 100%; } </style>")
            # btn_ask.style(full_width=True)
            answer = gr.Textbox(label='The answer to your question is :')

        btn_upload.click(
            upload,
            inputs=[file],
            outputs=[file],
        )
        btn_ask.click(
            ask,
            inputs=[question],
            outputs=[answer],
        )


demo.launch(server_port=7860, enable_queue=True, share=True)