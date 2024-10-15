import os
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import openai
import base64
from io import BytesIO
import re
import time
import pdb

class ParaSeedMultiClient:
    def __init__(self, base_url, model="default-lora"):
        self.client = openai.OpenAI(
            api_key='EMPTY',  # 如果不需要认证，保持为空
            base_url=base_url  # 服务器 B 的 IP 和监听端口
        )
        self.model = model

    def encode_image(self, image_path):
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f'data:image/jpeg;base64,{encoded_image}'

    def call_paraseed_multi(self, image_paths=None, question=None, history=None):
        images_content = [{"type": "image_url", "image_url": {"url": self.encode_image(img_path)}}
                          for img_path in image_paths]

        all_messages = history if history else []
        all_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                *images_content
            ]
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            seed=42
        )

        history.append({
            "role": "user",
            "content": question
        })
        history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return response.choices[0].message.content, history

# 设置头像图标路径
user_avatar_path = "./1.png"  # 用户头像路径
model_avatar_path = "./2.jpg"  # 模型头像路径

# 加载用户和模型头像
user_avatar = Image.open(user_avatar_path).resize((40, 40))
model_avatar = Image.open(model_avatar_path).resize((40, 40))

# 去除 <image> 标签的辅助函数
def clean_history(history):
    cleaned_history = []
    for entry in history:
        user_query = re.sub(r'<image>', '', entry[0])  # 去除用户问题中的 <image> 标签
        assistant_response = re.sub(r'<image>', '', entry[1])  # 去除助手回复中的 <image> 标签
        cleaned_history.append([user_query, assistant_response])
    return cleaned_history

# Streamlit 页面
st.title("ParaSeed 多模态大模型")

# 初始化 session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'cropped_images' not in st.session_state:
    st.session_state.cropped_images = []

if 'history' not in st.session_state:
    st.session_state.history = []  # 初始化 history 为空
    st.session_state.model_history = []  # 初始化 history 为空

# 定义一个临时目录用于保存裁剪后的图片
output_folder = "./tmp_cropped_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 图片上传部分
uploaded_files = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# 保存所有上传的图片并获取它们的路径
uploaded_image_paths = []
for uploaded_file in st.session_state.uploaded_files:
    temp_image_path = os.path.join(output_folder, uploaded_file.name)
    with open(temp_image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    uploaded_image_paths.append(temp_image_path)

# 显示所有上传的图片并缩放至合理大小
st.write("上传的图片：")
# 仅当有上传的图片时显示列布局
if uploaded_image_paths:
    # 设置每行最多显示3张图片
    num_columns = min(3, len(uploaded_image_paths))  
    columns = st.columns(num_columns)

    for idx, image_path in enumerate(uploaded_image_paths):
        img = Image.open(image_path)
        # 根据列数动态显示图片
        with columns[idx % num_columns]:  
            st.image(img, caption=f"图片 {idx+1}", use_column_width=True)
else:
    st.write("请上传图片以查看。")

# 初始化按钮点击状态
if 'show_annotation' not in st.session_state:
    st.session_state.show_annotation = False

# “进行标注”按钮，点击后设置状态
if st.button("进行标注/标注隐藏"):
    st.session_state.show_annotation = not st.session_state.show_annotation


# 展示和标注页面
if st.session_state.show_annotation and st.session_state.uploaded_files:
    selected_image = st.selectbox("选择图片进行标注", options=[file.name for file in st.session_state.uploaded_files])
    current_file = next(file for file in st.session_state.uploaded_files if file.name == selected_image)

    try:
        image = Image.open(current_file)
    except Exception as e:
        st.error(f"无法加载图像: {e}")


    st.write("在图片上框选区域进行裁剪:")
    temp_cropped_images = []
    st.image(image)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#000000",
        background_image=image.convert("RGBA"),
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key=f"canvas_{selected_image}",
    )

    if canvas_result.json_data is not None:
        shapes = canvas_result.json_data["objects"]
        st.session_state.cropped_images = []

        if shapes:
            for idx, shape in enumerate(shapes):
                left = int(shape["left"])
                top = int(shape["top"])
                width = int(shape["width"])
                height = int(shape["height"])

                cropped_image = image.crop((left, top, left + width, top + height))

                if cropped_image.mode == 'RGBA':
                    cropped_image = cropped_image.convert('RGB')

                cropped_image_path = os.path.join(output_folder, f"cropped_{selected_image}_{idx}.jpg")
                cropped_image.save(cropped_image_path, format="JPEG")
                temp_cropped_images.append(cropped_image_path)

            st.session_state.cropped_images = temp_cropped_images
        else:
            st.warning("没有选择任何裁剪区域。")

# 显示裁剪后的图片
if st.session_state.cropped_images:
    st.write("裁剪后的图片：")
    for img_path in st.session_state.cropped_images:
        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)





# 添加 CSS 样式控制
st.markdown(
    """
    <style>
    .chat-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .chat-avatar {
        margin-right: 10px;
    }
    .chat-message-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        max-width: 80%;
        word-wrap: break-word;
    }
    .chat-message {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
        max-width: 70%;
        word-wrap: break-word;
    }
    .user-chat {
        text-align: right;
        
        justify-content: flex-end;
        color: #00796b;  /* 用户输入框的字体颜色 */
    }
    .model-chat {
        text-align: left;
        justify-content: flex-start;
        color: #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 辅助函数将头像转换为 base64 字符串
def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

if 'history' not in st.session_state:
    st.session_state.history = []  # 初始化 history 为空
# 创建聊天记录显示框
history_display = st.empty()

# 显示会话历史的函数
def display_chat_history():
    with history_display.container():
        #history_display.empty()
        for chat in st.session_state.history:
            # 用户问题
            st.markdown(
                f"""
                <div class="chat-container user-chat">
                    <div class="chat-message-box">**用户**: {chat['question'].strip()}</div>
                    <img class="chat-avatar" src="data:image/png;base64,{image_to_base64(user_avatar)}">
                </div>
                """,
                unsafe_allow_html=True,
            )
            # 模型回答
            st.markdown(
                f"""
                <div class="chat-container model-chat">
                    <img class="chat-avatar" src="data:image/png;base64,{image_to_base64(model_avatar)}" width="40" height="40">
                    <div class="chat-message">**模型**: {chat['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 调用聊天记录显示，无论是否在标注模式下
display_chat_history()

# 输入问题框在最底部
query = st.text_area("输入问题进行推理:", key="input_query")
if st.session_state.uploaded_files:
    

    if st.button("发送问题"):
        if query:
            
            
            st.session_state.history.append({"question": query, "answer": "正在生成回答..."})
            display_chat_history()  # 仅显示新问题，无重复

            
            # 构建推理输入，合并上传的图片和裁剪后的图片路径（仅当裁剪图像不为空时）
            images_to_infer = uploaded_image_paths
            if st.session_state.cropped_images:
                images_to_infer += st.session_state.cropped_images

            # 在 query 的末尾添加 <image> 标签，数量与 images_to_infer 一致
            # query += "\n" + "<image>\n" * len(images_to_infer)
            
            
            # 模型推理
            # 初始化模型和LoRA权重
            #os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'0,1,2,3' #'0'
            with st.spinner("模型正在生成回答，请稍候..."):
                client = ParaSeedMultiClient(base_url='http://101.43.67.130:9001/v1', model="default-lora")
                response = "发生错误"
                try:
                    response, history = client.call_paraseed_multi(
                        image_paths=images_to_infer,
                        question=query,
                        history=st.session_state.model_history
                    )
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")
                

                # 更新会话历史
                st.session_state.history[-1]['answer'] = response
                st.session_state.model_history = history

                # 刷新历史记录，仅更新新增的问答
                display_chat_history()
                st.session_state.query = ""  # 清空输入框
            
            # 清空输入框
        # st.session_state.input_query = ""  # 清空输入框
            # # 显示模型回复在左侧
            # _, model_col = st.columns([3, 1])
            # with model_col:
            #     st.markdown(
            #         f"""
            #         <div class="chat-container">
            #             <img class="chat-avatar" src="data:image/png;base64,{st.image_to_base64(model_avatar)}" width="40" height="40">
            #             <div class="chat-message">**模型**: {clean_response}</div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True,
            #     )

            # st.write(f"模型回复: {response}")
            # st.write(f"推理历史记录: {history}")

    # # 显示历史会话记录
    # for chat in reversed(st.session_state.history):
    #     if chat["role"] == "user":
    #         user_col, _ = st.columns([1, 3])
    #         with user_col:
    #             st.markdown(
    #                 f"""
    #                 <div class="chat-container" style="text-align: right;">
    #                     <img class="chat-avatar" src="data:image/png;base64,{st.image_to_base64(user_avatar)}" width="40" height="40">
    #                     <div class="chat-message">{chat[0].strip()}</div>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True,
    #             )
    #     else:
    #         _, model_col = st.columns([3, 1])
    #         with model_col:
    #             st.markdown(
    #                 f"""
    #                 <div class="chat-container">
    #                     <img class="chat-avatar" src="data:image/png;base64,{st.image_to_base64(model_avatar)}" width="40" height="40">
    #                     <div class="chat-message">: {chat[1]}</div>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True,
    #             )
    # 清空已裁剪的图像
    
    if st.button("清空裁剪"):
        st.session_state.cropped_images = []




