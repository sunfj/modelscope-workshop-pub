import gradio as gr
import copy
import re
from threading import Thread
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _remove_image_special(text):
    text = text.replace("<ref>", "").replace("</ref>", "")
    return re.sub(r"<box>.*?(</box>|$)", "", text)


def is_video_file(filename):
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg"]
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message["content"]:
            if "image" in item:
                new_item = {"type": "image", "image": item["image"]}
            elif "text" in item:
                new_item = {"type": "text", "text": item["text"]}
            elif "video" in item:
                new_item = {"type": "video", "video": item["video"]}
            else:
                continue
            new_content.append(new_item)

        new_message = {"role": message["role"], "content": new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def make_demo(model, processor, model_name="Qwen3_VL"):
    def call_local_model(model, processor, messages):
        messages = transform_messages(messages)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        tokenizer = processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=3600.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {"max_new_tokens": 512, "streamer": streamer, **inputs}

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

    def predict(history):
        """Build multi-turn messages from chat history and stream a response."""
        messages = []

        for msg in history:
            role = msg["role"]
            raw_content = msg.get("content")

            # Gradio 6.9+ preprocesses chatbot content into a list of typed dicts:
            # [{'type': 'text', 'text': '...'}, {'type': 'file', 'file': {...}}, ...]
            # Older Gradio passes plain strings or dicts.
            if isinstance(raw_content, list):
                content_items = raw_content
            elif isinstance(raw_content, str):
                content_items = [{"type": "text", "text": raw_content}]
            elif isinstance(raw_content, dict) and "path" in raw_content:
                content_items = [{"type": "file", "file": raw_content}]
            else:
                continue

            if role == "user":
                built = []
                for item in content_items:
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text_val = item.get("text", "")
                        if text_val:
                            built.append({"text": text_val})
                    elif item_type == "file":
                        file_info = item.get("file", {})
                        filepath = file_info.get("path", "") if isinstance(file_info, dict) else str(file_info)
                        if filepath:
                            if is_video_file(filepath):
                                built.append({"video": f"file://{filepath}"})
                            else:
                                built.append({"image": f"file://{filepath}"})
                    # Legacy: plain dict with "path" key (older Gradio)
                    elif "path" in item:
                        filepath = item["path"]
                        if is_video_file(filepath):
                            built.append({"video": f"file://{filepath}"})
                        else:
                            built.append({"image": f"file://{filepath}"})
                if built:
                    messages.append({"role": "user", "content": built})

            elif role == "assistant":
                text_parts = [i.get("text", "") for i in content_items if i.get("type") == "text"]
                text = " ".join(t for t in text_parts if t)
                if text:
                    messages.append({"role": "assistant", "content": [{"text": text}]})

        # Remove last assistant (the empty placeholder we'll fill)
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()

        if not messages:
            return

        for response in call_local_model(model, processor, messages):
            history[-1]["content"] = [{"type": "text", "text": _remove_image_special(response)}]
            yield history

    def add_text(history, text):
        history = history or []
        if text and text.strip():
            history.append({"role": "user", "content": [{"type": "text", "text": text}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        return history, ""

    def add_file(history, file):
        history = history or []
        filepath = file if isinstance(file, str) else file.name
        history.append({"role": "user", "content": [{"type": "file", "file": {"path": filepath}}]})
        return history

    def reset_state():
        return []

    def regenerate(history):
        if not history:
            return history
        # Remove last assistant message
        while history and history[-1]["role"] == "assistant":
            history.pop()
        if not history:
            return history
        # Add empty assistant placeholder
        history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        yield from predict(history)

    with gr.Blocks() as demo:
        gr.Markdown(f"""<center><font size=8>{model_name} OpenVINO 演示</center>""")

        chatbot = gr.Chatbot(label=model_name, height=500)
        query = gr.Textbox(lines=2, label="输入")

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 上传文件", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🤔️ 重新生成")
            empty_bin = gr.Button("🧹 清除历史")

        submit_btn.click(add_text, [chatbot, query], [chatbot, query]).then(
            predict, [chatbot], [chatbot], show_progress=True
        )
        empty_bin.click(reset_state, [], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, addfile_btn], [chatbot], show_progress=True)

    return demo
